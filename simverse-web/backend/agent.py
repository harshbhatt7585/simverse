from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


def _load_env_from_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_from_file(Path(__file__).resolve().parent / ".env")
_load_env_from_file(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class AgentTurn:
    status: str
    reply: str


@dataclass
class BuildState:
    env_dir: Path | None = None
    generated_files: dict[str, str] = field(default_factory=dict)
    awaiting_build_confirmation: bool = False


@dataclass
class LoopState:
    step: int = 0
    max_steps: int = 100


@dataclass
class ClientResponse:
    output_text: str


class PersonalClient:
    def __init__(
        self,
        provider: str = "custom",
        *,
        custom_api_url: str = "http://127.0.0.1:9000/codex",
        custom_timeout_s: int = 120,
    ) -> None:
        normalized = provider.strip().lower()
        self.provider = normalized if normalized in {"openai", "custom"} else "custom"
        self.custom_api_url = custom_api_url
        self.custom_timeout_s = int(custom_timeout_s)
        self.openai_client = OpenAI() if self.provider == "openai" else None

    def create_response(
        self,
        *,
        model: str,
        input_payload: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> ClientResponse:
        if self.provider == "openai":
            if self.openai_client is None:
                raise RuntimeError("OpenAI provider selected but openai client is unavailable")
            response = self.openai_client.responses.create(
                model=model,
                input=input_payload,
                reasoning={"effort": "minimal"},
                max_output_tokens=max_output_tokens,
            )
            return ClientResponse(output_text=(getattr(response, "output_text", "") or ""))
        return self._create_custom_response(input_payload=input_payload)

    def _create_custom_response(self, *, input_payload: list[dict[str, Any]]) -> ClientResponse:
        prompt_parts: list[str] = []
        for message in input_payload:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            prompt_parts.append(f"[{role}]\\n{content}")
        prompt = "\n\n".join(prompt_parts)

        body = {
            "prompt": prompt,
            "timeout_s": self.custom_timeout_s,
            "include_events": False,
            "include_raw_output": True,
            "extra_args": ["--skip-git-repo-check"],
        }
        candidate_urls = self._candidate_custom_urls(self.custom_api_url)
        last_error: Exception | None = None
        payload_text: str | None = None
        attempted: list[str] = []
        for url in candidate_urls:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.custom_timeout_s + 5) as resp:
                    payload_text = resp.read().decode("utf-8")
                    break
            except urllib.error.HTTPError as exc:
                attempted.append(f"{url} -> HTTP {exc.code}")
                last_error = exc
                continue
            except urllib.error.URLError as exc:
                attempted.append(f"{url} -> {exc}")
                last_error = exc
                continue
        if payload_text is None:
            detail = "; ".join(attempted) if attempted else str(last_error)
            raise RuntimeError(f"Custom API request failed. Tried: {detail}")

        parsed = self._parse_custom_payload(payload_text)
        return ClientResponse(output_text=parsed)

    def _candidate_custom_urls(self, configured_url: str) -> list[str]:
        base = configured_url.rstrip("/")
        if not base:
            return [
                "http://127.0.0.1:9000/codex",
                "http://127.0.0.1:9000/codex/",
                "http://127.0.0.1:9000",
            ]
        if base.endswith("/codex"):
            parent = base[: -len("/codex")] or base
            return [base, f"{base}/", parent]
        return [f"{base}/codex", f"{base}/codex/", base]

    def _parse_custom_payload(self, payload_text: str) -> str:
        text = payload_text.strip()
        if not text:
            return ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text

        if isinstance(data, dict):
            for key in (
                "final_text",
                "output_text",
                "text",
                "response",
                "result",
                "content",
                "stdout_text",
                "stderr_text",
            ):
                value = data.get(key)
                if isinstance(value, str):
                    return self._strip_code_fence(value)
            if isinstance(data.get("output"), str):
                return self._strip_code_fence(data["output"])
            return json.dumps(data)
        if isinstance(data, list):
            joined: list[str] = []
            for item in data:
                if isinstance(item, dict):
                    for key in ("text", "content", "message"):
                        value = item.get(key)
                        if isinstance(value, str):
                            joined.append(self._strip_code_fence(value))
                            break
                elif isinstance(item, str):
                    joined.append(self._strip_code_fence(item))
            return "\n".join(joined).strip()
        return str(data)

    def _strip_code_fence(self, text: str) -> str:
        stripped = text.strip()
        match = re.match(r"^```(?:python|json)?\n(.*)\n```$", stripped, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return stripped


class SimpleTerminalAgent:
    def __init__(
        self,
        name: str,
        workspace: Path,
        model: str = "gpt-5-nano",
    ) -> None:
        self.name = name
        self.workspace = workspace
        self.repo_root = self.workspace.parents[1]
        self.model = model
        self.provider = os.getenv("SIMVERSE_CLIENT_PROVIDER", "custom").strip().lower()
        self.client = self._build_client()
        self.history: list[dict[str, str]] = []
        self.build_state = BuildState()
        self.loop_state = LoopState()
        self.abstract_context = self._build_abstract_context()
        self.farmtila_examples = self._build_farmtila_examples()
        self.system_prompt = self._build_system_prompt()

    def _build_client(self) -> PersonalClient | None:
        provider = self.provider if self.provider in {"openai", "custom"} else "custom"
        if provider == "openai":
            if OpenAI is None:
                return None
            if not os.getenv("OPENAI_API_KEY"):
                return None
        custom_url = os.getenv("SIMVERSE_CUSTOM_API_URL", "http://127.0.0.1:9000/codex")
        timeout_s = int(os.getenv("SIMVERSE_CUSTOM_TIMEOUT_S", "120"))
        return PersonalClient(
            provider=provider,
            custom_api_url=custom_url,
            custom_timeout_s=timeout_s,
        )

    def handle_user_message(self, user_input: str) -> AgentTurn:
        loop = AgenticLoop(self)
        return loop.run_once(user_input)

    def observe(self, user_input: str) -> dict[str, Any]:
        text = user_input.strip()
        return {
            "user_input": text,
            "history_tail": self.history[-20:],
            "has_client": self.client is not None,
            "has_env": self.build_state.env_dir is not None,
            "generated_files": sorted(self.build_state.generated_files.keys()),
        }

    def decide(self, obs: dict[str, Any]) -> dict[str, Any]:
        text = str(obs.get("user_input", "")).strip()
        lowered = text.lower()

        if not text:
            return {"action": "chat", "status": "idle", "reply": "Say something and I will reply."}

        if not obs.get("has_client"):
            return {
                "action": "error",
                "status": "error",
                "reply": (
                    "Client unavailable. For OpenAI, install `openai` and set `OPENAI_API_KEY`. "
                    "For custom, set `SIMVERSE_CLIENT_PROVIDER=custom`."
                ),
            }

        if lowered == "train":
            return {"action": "run_train", "status": "run_train"}

        if self._is_yes_intent(lowered) and self.build_state.awaiting_build_confirmation:
            return {"action": "build_files", "status": "build"}

        if self._is_build_intent(lowered):
            return {"action": "build_files", "status": "build"}

        messages = (
            [
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            ]
            + self.history[-20:]
            + [{"role": "user", "content": text}]
        )
        response = self._create_response(input_payload=messages, max_output_tokens=1400)
        reply = (getattr(response, "output_text", "") or "").strip()
        if not reply:
            return {
                "action": "chat",
                "status": "error",
                "reply": "I could not generate a response. Please try again.",
            }
        return {"action": "chat", "status": "ok", "reply": reply}

    def act(self, decision: dict[str, Any]) -> AgentTurn:
        action = str(decision.get("action", "chat"))

        if action == "error":
            return AgentTurn(status="error", reply=str(decision.get("reply", "Unknown error.")))
        if action == "run_train":
            return self._run_training()
        if action == "build_files":
            self.build_state.awaiting_build_confirmation = False
            return self._build_files_one_by_one()
        if action == "finish":
            return AgentTurn(status="done", reply=str(decision.get("reply", "Done.")))

        turn = AgentTurn(
            status=str(decision.get("status", "ok")),
            reply=str(decision.get("reply", "OK")),
        )
        self.build_state.awaiting_build_confirmation = self._asks_user_to_build(turn.reply)
        return turn

    def check(self, turn: AgentTurn) -> AgentTurn:
        if not turn.reply.strip():
            return AgentTurn(status="error", reply="Empty response generated.")
        return turn

    def update(self, user_input: str, turn: AgentTurn) -> None:
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": turn.reply})
        self.loop_state.step += 1

    def _build_files_one_by_one(self) -> AgentTurn:
        out_dir = self._new_env_dir()
        order = ["config.py", "agent.py", "env.py", "render.py", "train.py"]
        instructions = {
            "config.py": "Create config dataclasses and defaults for this environment.",
            "agent.py": "Create agent class compatible with SimVerse abstractions.",
            "env.py": "Create the RL environment using SimVerse-style abstractions.",
            "render.py": "Create a simple renderer for environment state.",
            "train.py": "Create runnable training entrypoint using env/config/agent.",
        }

        generated: dict[str, str] = {}
        self.build_state.env_dir = out_dir
        self.build_state.generated_files = {}
        for filename in order:
            print(f"Agent: generating {filename}...")
            try:
                code = self._generate_single_file(
                    filename=filename,
                    instruction=instructions[filename],
                    generated_so_far=generated,
                )
                file_path = out_dir / filename
                file_path.write_text(code.rstrip() + "\n", encoding="utf-8")
                self._format_generated_file(file_path)
                generated[filename] = code
                self.build_state.generated_files[filename] = code
                print(f"Agent: saved {file_path}")
            except Exception as exc:
                partial = ", ".join(generated.keys()) if generated else "(none)"
                return AgentTurn(
                    status="error",
                    reply=(
                        "Build failed while generating files.\n"
                        f"Failed file: {filename}\n"
                        f"Saved so far: {partial}\n"
                        f"Output folder: {out_dir}\n"
                        f"Error: {exc}"
                    ),
                )

        self.build_state.generated_files = generated
        files_written = "\n".join(f"- {out_dir / name}" for name in order)
        reply = (
            "Generated files one-by-one and saved to a new env directory:\n"
            f"{files_written}\n"
            "Type `train` to run the generated `train.py`."
        )
        return AgentTurn(status="built", reply=reply)

    def _generate_single_file(
        self,
        *,
        filename: str,
        instruction: str,
        generated_so_far: dict[str, str],
    ) -> str:
        history_context = self._clip(self._history_text(), max_chars=5000)
        existing_context = self._clip(self._generated_context(generated_so_far), max_chars=12000)
        example_context = self._example_for_file(filename)

        prompt = (
            "You must return ONLY JSON with tool calls. No markdown, no prose.\n\n"
            "Tool schema:\n"
            "{\n"
            '  "tool_calls": [\n'
            "    {\n"
            '      "tool": "write_file",\n'
            '      "path": "target filename",\n'
            '      "content": "full python file content"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Return exactly one tool call.\n"
            f"- `path` MUST be exactly `{filename}`.\n"
            "- `content` must be complete file code.\n"
            "- No extra keys outside schema.\n\n"
            f"Target file: {filename}\n"
            f"Task: {instruction}\n\n"
            "Formatting and quality requirements:\n"
            "- Output valid Python 3.10+ syntax.\n"
            "- Keep lines <= 100 chars.\n"
            "- Use clear type hints on public functions.\n"
            "- Use clean imports (no unused imports).\n"
            "- Keep names consistent and descriptive.\n"
            "- No TODO placeholders or pseudo code.\n"
            "- Ensure file is immediately runnable/importable.\n\n"
            "Required framework constraints:\n"
            "- Use SimVerse abstractions correctly.\n"
            "- Use `simverse.abstractor.*` and `simverse.envs.<new_env>.*` import patterns.\n"
            "- Ensure generated files are consistent with each other.\n"
            "- `train.py` should call `run_ppo_training`.\n\n"
            "SimVerse abstraction reference snippets:\n"
            f"{self._context_for_file(filename)}\n\n"
            "User conversation context:\n"
            f"{history_context}\n\n"
            "Already generated files:\n"
            f"{existing_context}\n\n"
            "Reference example (trimmed):\n"
            f"{example_context}\n"
        )

        response = self._create_response(
            input_payload=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=3000,
        )
        raw = (getattr(response, "output_text", "") or "").strip()
        call = self._extract_write_file_tool_call(raw=raw, filename=filename)
        code = str(call["content"]).strip()
        if len(code) >= 120:
            return code

        retry_prompt = (
            f"Regenerate `{filename}` with full implementation. "
            "Remember: return JSON tool_calls only, exactly one write_file call."
        )
        retry_response = self._create_response(
            input_payload=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": prompt},
                {"role": "user", "content": retry_prompt},
            ],
            max_output_tokens=3000,
        )
        retry_raw = (getattr(retry_response, "output_text", "") or "").strip()
        retry_call = self._extract_write_file_tool_call(raw=retry_raw, filename=filename)
        return str(retry_call["content"]).strip()

    def _build_abstract_context(self) -> dict[str, str]:
        base = self.repo_root / "src" / "simverse"
        return {
            "simenv": self._clip(
                self._read_text(base / "abstractor" / "simenv.py"),
                max_chars=6000,
            ),
            "simagent": self._clip(
                self._read_text(base / "abstractor" / "agent.py"),
                max_chars=3500,
            ),
            "train_utils": self._clip(
                self._read_text(base / "abstractor" / "train_utils.py"),
                max_chars=7000,
            ),
            "farmtila_env": self._clip(
                self._read_text(base / "envs" / "farmtila" / "env.py"),
                max_chars=5000,
            ),
            "farmtila_agent": self._clip(
                self._read_text(base / "envs" / "farmtila" / "agent.py"),
                max_chars=3500,
            ),
            "farmtila_config": self._clip(
                self._read_text(base / "envs" / "farmtila" / "config.py"),
                max_chars=2500,
            ),
            "farmtila_train": self._clip(
                self._read_text(base / "envs" / "farmtila" / "train.py"),
                max_chars=5000,
            ),
        }

    def _build_farmtila_examples(self) -> dict[str, str]:
        base = self.repo_root / "src" / "simverse" / "envs" / "farmtila"
        return {
            "env.py": self._read_text(base / "env.py"),
            "config.py": self._read_text(base / "config.py"),
            "agent.py": self._read_text(base / "agent.py"),
            "train.py": self._read_text(base / "train.py"),
        }

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.name}. You are an expert SimVerse RL environment builder.\n"
            "Ask concise follow-up questions when details are missing.\n"
            "When user asks to build, generate complete and consistent files.\n"
            "Follow SimVerse abstractions exactly.\n"
            "When generating files, obey the tool-call JSON schema exactly.\n"
            "Never return prose when tool calls are requested.\n"
        )

    def _context_for_file(self, filename: str) -> str:
        if filename == "env.py":
            return (
                "SimEnv interface:\n"
                f"{self.abstract_context['simenv']}\n\n"
                "Reference env pattern:\n"
                f"{self.abstract_context['farmtila_env']}"
            )
        if filename == "agent.py":
            return (
                "SimAgent interface:\n"
                f"{self.abstract_context['simagent']}\n\n"
                "Reference agent pattern:\n"
                f"{self.abstract_context['farmtila_agent']}"
            )
        if filename == "config.py":
            return "Reference config pattern:\n" f"{self.abstract_context['farmtila_config']}"
        if filename == "train.py":
            return (
                "run_ppo_training usage and helpers:\n"
                f"{self.abstract_context['train_utils']}\n\n"
                "Reference train pattern:\n"
                f"{self.abstract_context['farmtila_train']}"
            )
        return (
            "Env rendering should work with generated env state and use simple terminal output.\n"
            "Keep render dependencies minimal."
        )

    def _run_training(self) -> AgentTurn:
        if self.build_state.env_dir is None:
            return AgentTurn(
                status="error",
                reply="No generated environment yet. Type `build` first.",
            )
        train_file = self.build_state.env_dir / "train.py"
        if not train_file.exists():
            return AgentTurn(status="error", reply="`train.py` not found in generated env folder.")

        proc: CompletedProcess[str] = run(
            [os.sys.executable, str(train_file)],
            cwd=str(self.build_state.env_dir),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            error_text = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"
            return AgentTurn(status="error", reply=f"Training failed:\n{error_text}")

        output = proc.stdout.strip() or "Training finished with no output."
        return AgentTurn(status="trained", reply=f"Training complete:\n{output}")

    def _new_env_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = self.workspace / "generated_envs" / f"env_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir

    def _is_build_intent(self, lowered_text: str) -> bool:
        if lowered_text in {"build", "generate", "create env", "create environment"}:
            return True
        has_build_word = re.search(r"\b(build|generate|create|scaffold)\b", lowered_text)
        has_target_word = re.search(r"\b(env|environment|files|code|rl)\b", lowered_text)
        return bool(has_build_word and has_target_word)

    def _is_yes_intent(self, lowered_text: str) -> bool:
        cleaned = lowered_text.strip()
        return cleaned in {"yes", "y", "ok", "okay", "sure", "yep", "do it", "go ahead"}

    def _asks_user_to_build(self, reply: str) -> bool:
        lowered = reply.lower()
        return (
            "type `build`" in lowered
            or "type build" in lowered
            or "say build" in lowered
            or "run build" in lowered
        )

    def _history_text(self) -> str:
        lines: list[str] = []
        for msg in self.history[-30:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _generated_context(self, generated_so_far: dict[str, str]) -> str:
        if not generated_so_far:
            return "(none yet)"
        blocks: list[str] = []
        for name, code in generated_so_far.items():
            blocks.append(f"# {name}\n{self._clip(code, max_chars=3000)}")
        return "\n\n".join(blocks)

    def _example_for_file(self, filename: str) -> str:
        if filename in {"env.py", "config.py", "agent.py", "train.py"}:
            return self.farmtila_examples.get(filename, "")
        if filename == "render.py":
            return "Keep render simple and terminal-friendly."
        return ""

    def _extract_write_file_tool_call(self, *, raw: str, filename: str) -> dict[str, str]:
        payload = self._parse_json_object(raw)
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise ValueError(f"Expected exactly one tool call for {filename}")

        call = tool_calls[0]
        if not isinstance(call, dict):
            raise ValueError(f"Invalid tool call object for {filename}")
        if call.get("tool") != "write_file":
            raise ValueError(f"Unsupported tool `{call.get('tool')}` for {filename}")
        if call.get("path") != filename:
            raise ValueError(f"Tool call path must be exactly `{filename}`")

        content = call.get("content")
        if not isinstance(content, str) or len(content.strip()) < 40:
            raise ValueError(f"Tool call content is too short for {filename}")
        return {"tool": "write_file", "path": filename, "content": content}

    def _parse_json_object(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            fenced = re.findall(r"```(?:json)?\n(.*?)```", text, flags=re.DOTALL)
            if fenced:
                text = fenced[0].strip()

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model did not return JSON object")
        candidate = text[start : end + 1]
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("Model JSON payload is not an object")
        return payload

    def _clip(self, text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return f"(missing: {path})"
        return path.read_text(encoding="utf-8")

    def _create_response(
        self,
        *,
        input_payload: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> Any:
        if self.client is None:
            raise RuntimeError("Personal client unavailable")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.client.create_response(
                    model=self.model,
                    input_payload=input_payload,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Model request failed after retries: {last_error}")

    def _format_generated_file(self, path: Path) -> None:
        proc: CompletedProcess[str] = run(
            ["ruff", "format", str(path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ruff format failed for {path.name}: {proc.stderr.strip() or proc.stdout.strip()}"
            )


class AgenticLoop:
    def __init__(self, agent: SimpleTerminalAgent) -> None:
        self.agent = agent

    def run_once(self, user_input: str) -> AgentTurn:
        obs = self.agent.observe(user_input)
        try:
            decision = self.agent.decide(obs)
        except Exception as exc:
            return AgentTurn(status="error", reply=f"Request failed: {exc}")
        turn = self.agent.act(decision)
        validated_turn = self.agent.check(turn)
        self.agent.update(user_input, validated_turn)
        return validated_turn


def create_agent(name: str, workspace: Path, model: str = "gpt-5-nano") -> SimpleTerminalAgent:
    return SimpleTerminalAgent(name=name, workspace=workspace, model=model)


def run_cli() -> None:
    agent = create_agent(name="SimVerse Assistant", workspace=Path(__file__).resolve().parent)
    print("SimVerse Builder Agent")
    print("Chat to describe your environment.")
    print("Type `build` to generate files one-by-one into a new folder.")
    print("Type `train` to run generated training.")
    print("Type `exit` to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        lowered = user_input.lower()
        if lowered == "train":
            print("Agent: starting training run...")
        elif (
            lowered in {"build", "generate", "create env", "create environment"}
            or ("build" in lowered and "env" in lowered)
            or ("generate" in lowered and "env" in lowered)
        ):
            print("Agent: starting environment creation (files will be generated one-by-one)...")

        turn = agent.handle_user_message(user_input)
        print(f"Agent ({turn.status}): {turn.reply}\n")


if __name__ == "__main__":
    run_cli()
