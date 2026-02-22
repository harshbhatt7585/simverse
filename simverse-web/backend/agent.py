from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any

from openai import OpenAI


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


@dataclass
class LoopState:
    step: int = 0
    max_steps: int = 100


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
        self.client = self._build_client()
        self.history: list[dict[str, str]] = []
        self.build_state = BuildState()
        self.loop_state = LoopState()
        self.abstract_context = self._build_abstract_context()
        self.farmtila_examples = self._build_farmtila_examples()
        self.system_prompt = self._build_system_prompt()

    def _build_client(self) -> OpenAI | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return OpenAI()

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
                "reply": "OpenAI client unavailable. Install `openai` and set `OPENAI_API_KEY`.",
            }

        if lowered == "train":
            return {"action": "run_train", "status": "run_train"}

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
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            reasoning={"effort": "minimal"},
            max_output_tokens=1400,
        )
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
            return self._build_files_one_by_one()
        if action == "finish":
            return AgentTurn(status="done", reply=str(decision.get("reply", "Done.")))

        return AgentTurn(
            status=str(decision.get("status", "ok")),
            reply=str(decision.get("reply", "OK")),
        )

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
        for filename in order:
            code = self._generate_single_file(
                filename=filename,
                instruction=instructions[filename],
                generated_so_far=generated,
            )
            (out_dir / filename).write_text(code.rstrip() + "\n", encoding="utf-8")
            generated[filename] = code

        self.build_state.env_dir = out_dir
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
            "Return ONLY raw Python code for the requested file. "
            "No markdown, no explanation.\n\n"
            f"Target file: {filename}\n"
            f"Task: {instruction}\n\n"
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

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": "minimal"},
            max_output_tokens=5000,
        )
        raw = (getattr(response, "output_text", "") or "").strip()
        code = self._extract_code(raw)
        if len(code) >= 120:
            return code

        retry_prompt = (
            f"Regenerate `{filename}` with full implementation. "
            "Return only code and ensure imports/classes/functions are complete."
        )
        retry_response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": prompt},
                {"role": "user", "content": retry_prompt},
            ],
            reasoning={"effort": "minimal"},
            max_output_tokens=5000,
        )
        retry_raw = (getattr(retry_response, "output_text", "") or "").strip()
        return self._extract_code(retry_raw)

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
            "Follow SimVerse abstractions exactly.\n\n"
            "SimVerse abstraction contracts (must follow):\n\n"
            "### simverse/abstractor/simenv.py\n"
            f"{self.abstract_context['simenv']}\n\n"
            "### simverse/abstractor/agent.py\n"
            f"{self.abstract_context['simagent']}\n\n"
            "### simverse/abstractor/train_utils.py\n"
            f"{self.abstract_context['train_utils']}\n\n"
            "Farmtila reference code (use as implementation pattern):\n\n"
            "### farmtila/config.py\n"
            f"{self.farmtila_examples['config.py']}\n\n"
            "### farmtila/agent.py\n"
            f"{self.farmtila_examples['agent.py']}\n\n"
            "### farmtila/env.py\n"
            f"{self.farmtila_examples['env.py']}\n\n"
            "### farmtila/train.py\n"
            f"{self.farmtila_examples['train.py']}\n"
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

    def _extract_code(self, raw: str) -> str:
        if raw.startswith("```"):
            matches = re.findall(r"```(?:python)?\n(.*?)```", raw, flags=re.DOTALL)
            if matches:
                return matches[0].strip()
        return raw.strip()

    def _clip(self, text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return f"(missing: {path})"
        return path.read_text(encoding="utf-8")


class AgenticLoop:
    def __init__(self, agent: SimpleTerminalAgent) -> None:
        self.agent = agent

    def run_once(self, user_input: str) -> AgentTurn:
        obs = self.agent.observe(user_input)
        decision = self.agent.decide(obs)
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

        turn = agent.handle_user_message(user_input)
        print(f"Agent ({turn.status}): {turn.reply}\n")


if __name__ == "__main__":
    run_cli()
