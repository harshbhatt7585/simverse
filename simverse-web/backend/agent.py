from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

REQUIRED_FIELDS = (
    "environment_name",
    "objective",
    "action_space",
    "observation_space",
    "reward_function",
    "termination_conditions",
)

FIELD_ALIASES = {
    "environment_name": {"name", "environment_name", "environment", "env_name"},
    "objective": {"objective", "goal", "task"},
    "action_space": {"action_space", "actions", "agent_actions"},
    "observation_space": {"observation_space", "observations", "state_space", "state"},
    "reward_function": {"reward_function", "reward", "rewards"},
    "termination_conditions": {
        "termination_conditions",
        "termination",
        "episode_end",
        "done_condition",
    },
    "difficulty_modes": {"difficulty_modes", "difficulty", "modes"},
    "baseline_policy": {"baseline_policy", "baseline"},
    "evaluation_metrics": {"evaluation_metrics", "metrics"},
}

FIELD_QUESTIONS = {
    "environment_name": "What should the environment be called?",
    "objective": "What is the primary objective of the agent?",
    "action_space": "What actions can the agent take?",
    "observation_space": "What observations/states does the agent receive?",
    "reward_function": "How should rewards be computed?",
    "termination_conditions": "When should an episode terminate?",
}

ALLOWED_ACTIONS = {"ask", "build", "train", "build_and_train"}

PLANNER_SYSTEM_PROMPT = """
You produce the next step for an RL-environment-building agent.
Return strict JSON with this schema:
{
  "assistant_reply": "string",
  "details_update": {
    "environment_name": "optional string",
    "objective": "optional string",
    "action_space": "optional string",
    "observation_space": "optional string",
    "reward_function": "optional string",
    "termination_conditions": "optional string",
    "difficulty_modes": "optional string",
    "baseline_policy": "optional string",
    "evaluation_metrics": "optional string"
  },
  "action": "ask | build | train | build_and_train"
}
Rules:
- If required fields are missing, choose action "ask".
- Choose "build" only when required fields are complete.
- Choose "train" only when user asks to train and files already exist.
- Choose "build_and_train" only when user asks to train and build is still needed.
- Keep assistant_reply short and actionable.
- Output JSON only, no markdown.
""".strip()


def _build_system_prompt(name: str) -> str:
    return textwrap.dedent(
        f"""
        You are {name}, an expert RL environment builder.
        Ask the user for missing environment details, then build the environment files.
        Build these files: env.py (environment), render.py (renderer), train.py (training loop).
        Continue until the environment scaffold is generated and training can run.
        After build is complete, ask if the user wants training.
        """
    ).strip()


def create_agent(
    name: str,
    workspace: str | Path | None = None,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
) -> "OpenAIRLBuilderAgent":
    return OpenAIRLBuilderAgent(
        name=name,
        workspace=workspace or Path(__file__).resolve().parent,
        model=model,
        api_key=api_key,
    )


@dataclass
class EnvironmentDetails:
    environment_name: Optional[str] = None
    objective: Optional[str] = None
    action_space: Optional[str] = None
    observation_space: Optional[str] = None
    reward_function: Optional[str] = None
    termination_conditions: Optional[str] = None
    difficulty_modes: Optional[str] = None
    baseline_policy: Optional[str] = None
    evaluation_metrics: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        return {key: (value or "") for key, value in asdict(self).items()}

    def missing_fields(self) -> list[str]:
        return [field for field in REQUIRED_FIELDS if not getattr(self, field)]

    def apply_updates(self, updates: dict[str, str]) -> None:
        for key, value in updates.items():
            if key not in FIELD_ALIASES:
                continue
            cleaned = str(value).strip()
            if cleaned:
                setattr(self, key, cleaned)


@dataclass
class PlannerDecision:
    assistant_reply: str
    details_update: dict[str, str]
    action: str


@dataclass
class TrainingResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


@dataclass
class AgentTurn:
    status: str
    reply: str
    missing_fields: list[str]
    created_files: list[str]
    training_result: Optional[TrainingResult] = None


class OpenAIRLBuilderAgent:
    def __init__(
        self,
        name: str,
        workspace: str | Path,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        output_subdir: str = "generated_env",
    ):
        if OpenAI is None:
            raise RuntimeError(
                "openai package is not installed. Install it with: "
                "`python3 -m pip install openai`."
            )

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=resolved_key)
        self.name = name
        self.model = model
        self.workspace = Path(workspace).resolve()
        self.output_dir = self.workspace / output_subdir
        self.system_prompt = _build_system_prompt(name)
        self.messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.details = EnvironmentDetails()
        self.created_files: list[str] = []

    def handle_user_message(self, message: str) -> AgentTurn:
        text = message.strip()
        if not text:
            reply = "Please share the environment requirements."
            return AgentTurn(
                status="awaiting_input",
                reply=reply,
                missing_fields=self.details.missing_fields(),
                created_files=self.created_files.copy(),
            )

        self.messages.append({"role": "user", "content": text})

        direct_updates = _parse_key_values(text)
        self.details.apply_updates(direct_updates)

        decision = self._plan_next_step(user_message=text)
        self.details.apply_updates(decision.details_update)

        missing = self.details.missing_fields()
        action = decision.action if decision.action in ALLOWED_ACTIONS else "ask"

        if missing and action != "ask":
            action = "ask"

        if _is_train_intent(text) and action == "ask" and not missing:
            action = "train" if self._has_generated_files() else "build_and_train"

        reply_parts: list[str] = []
        if decision.assistant_reply:
            reply_parts.append(decision.assistant_reply.strip())

        created_files = self.created_files.copy()
        training_result: Optional[TrainingResult] = None

        if action in {"build", "build_and_train"} and not missing:
            if self._has_generated_files():
                if not reply_parts:
                    reply_parts.append("Environment scaffold is already generated.")
            else:
                created_files = self.build_environment_files()
                reply_parts.append(f"Generated files: {', '.join(created_files)}")

        if action in {"train", "build_and_train"}:
            if missing:
                first_missing = FIELD_QUESTIONS[missing[0]]
                reply_parts.append(first_missing)
                reply_parts.append(
                    f"Missing details: {', '.join(field.replace('_', ' ') for field in missing)}."
                )
                reply = "\n".join(reply_parts).strip()
                self.messages.append({"role": "assistant", "content": reply})
                return AgentTurn(
                    status="collecting_requirements",
                    reply=reply,
                    missing_fields=missing,
                    created_files=self.created_files.copy(),
                )

            if not self._has_generated_files():
                created_files = self.build_environment_files()
                reply_parts.append(f"Generated files: {', '.join(created_files)}")

            training_result = self.train_environment()
            if training_result.success:
                reply_parts.append("Training completed successfully.")
                reply_parts.append(training_result.stdout.strip() or "[no output]")
                status = "trained"
            else:
                reply_parts.append("Training failed.")
                reply_parts.append(f"stdout:\n{training_result.stdout.strip() or '[no output]'}")
                reply_parts.append(f"stderr:\n{training_result.stderr.strip() or '[no output]'}")
                status = "training_failed"

            reply = "\n\n".join(part for part in reply_parts if part).strip()
            self.messages.append({"role": "assistant", "content": reply})
            return AgentTurn(
                status=status,
                reply=reply,
                missing_fields=self.details.missing_fields(),
                created_files=self.created_files.copy(),
                training_result=training_result,
            )

        if missing:
            first_missing = FIELD_QUESTIONS[missing[0]]
            if not reply_parts:
                reply_parts.append(first_missing)
            reply_parts.append(
                f"Missing details: {', '.join(field.replace('_', ' ') for field in missing)}."
            )
            status = "collecting_requirements"
        else:
            if self._has_generated_files():
                reply_parts.append("Scaffold exists. Reply with `train` to start training.")
                status = "ready_to_train"
            else:
                created_files = self.build_environment_files()
                reply_parts.append(f"Generated files: {', '.join(created_files)}")
                reply_parts.append("Reply with `train` to run training.")
                status = "built"

        reply = "\n".join(part for part in reply_parts if part).strip()
        self.messages.append({"role": "assistant", "content": reply})
        return AgentTurn(
            status=status,
            reply=reply,
            missing_fields=self.details.missing_fields(),
            created_files=self.created_files.copy(),
            training_result=training_result,
        )

    def build_environment_files(self) -> list[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        env_path = self.output_dir / "env.py"
        render_path = self.output_dir / "render.py"
        train_path = self.output_dir / "train.py"

        env_path.write_text(self._render_env_file(), encoding="utf-8")
        render_path.write_text(self._render_render_file(), encoding="utf-8")
        train_path.write_text(self._render_train_file(), encoding="utf-8")

        self.created_files = [str(env_path), str(render_path), str(train_path)]
        return self.created_files.copy()

    def train_environment(self) -> TrainingResult:
        command = [sys.executable, "train.py"]
        process: CompletedProcess[str] = run(
            command,
            cwd=self.output_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        return TrainingResult(
            command=command,
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )

    def _plan_next_step(self, user_message: str) -> PlannerDecision:
        context = {
            "current_details": self.details.to_dict(),
            "required_fields": list(REQUIRED_FIELDS),
            "missing_fields": self.details.missing_fields(),
            "files_generated": self._has_generated_files(),
            "user_message": user_message,
            "recent_messages": self.messages[-8:],
        }

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Use the context below and produce the JSON decision.\n"
                        f"{json.dumps(context, indent=2)}"
                    ),
                },
            ],
        )

        raw = completion.choices[0].message.content or "{}"
        parsed = _parse_json(raw)

        reply = str(parsed.get("assistant_reply", "")).strip()
        action = str(parsed.get("action", "ask")).strip().lower()
        details_update = _normalize_details_update(parsed.get("details_update", {}))

        if action not in ALLOWED_ACTIONS:
            action = "ask"

        return PlannerDecision(
            assistant_reply=reply,
            details_update=details_update,
            action=action,
        )

    def _has_generated_files(self) -> bool:
        expected_files = ("env.py", "render.py", "train.py")
        return all((self.output_dir / file_name).exists() for file_name in expected_files)

    def _render_env_file(self) -> str:
        details_json = json.dumps(self.details.to_dict(), indent=2)
        action_list = json.dumps(_normalize_actions(self.details.action_space or ""))
        return textwrap.dedent(
            f"""\
            \"\"\"Auto-generated RL environment scaffold.\"\"\"

            from __future__ import annotations

            import random

            ENV_SPEC = {details_json}
            ACTIONS = {action_list}


            class Environment:
                def __init__(self, goal_position: int = 6, max_steps: int = 25, seed: int = 0):
                    self.goal_position = goal_position
                    self.max_steps = max_steps
                    self.position = 0
                    self.steps = 0
                    self._rng = random.Random(seed)

                @property
                def action_count(self) -> int:
                    return len(ACTIONS)

                def reset(self) -> int:
                    self.position = 0
                    self.steps = 0
                    return self.position

                def step(self, action: int):
                    if action < 0 or action >= self.action_count:
                        raise ValueError(f"Invalid action index: {{action}}")

                    self.steps += 1
                    action_name = ACTIONS[action]
                    if action_name in {{"right", "forward", "up", "east"}}:
                        self.position += 1
                    elif action_name in {{"left", "backward", "down", "west"}}:
                        self.position -= 1
                    else:
                        if self._rng.random() > 0.6:
                            self.position += 1

                    self.position = max(-self.goal_position, min(self.position, self.goal_position))
                    reached_goal = self.position >= self.goal_position
                    timeout = self.steps >= self.max_steps
                    reward = 10.0 if reached_goal else -0.1
                    info = {{"action": action_name}}
                    return self.position, reward, reached_goal, timeout, info
            """
        )

    def _render_render_file(self) -> str:
        return textwrap.dedent(
            """\
            \"\"\"Renderer for the generated environment.\"\"\"

            from __future__ import annotations

            from env import Environment


            def render_position(position: int, goal_position: int) -> str:
                line = ["-"] * (goal_position * 2 + 1)
                origin_index = goal_position
                goal_index = min(len(line) - 1, origin_index + goal_position)
                agent_index = max(0, min(len(line) - 1, origin_index + position))
                line[goal_index] = "G"
                line[origin_index] = "S"
                line[agent_index] = "A"
                return "".join(line)


            if __name__ == "__main__":
                env = Environment()
                observation = env.reset()
                print(render_position(observation, env.goal_position))
            """
        )

    def _render_train_file(self) -> str:
        return textwrap.dedent(
            """\
            \"\"\"Training loop for the generated environment (tabular Q-learning).\"\"\"

            from __future__ import annotations

            import json
            import random
            from pathlib import Path

            from env import Environment


            def _get_q_values(
                table: dict[int, list[float]],
                state: int,
                actions: int,
            ) -> list[float]:
                if state not in table:
                    table[state] = [0.0] * actions
                return table[state]


            def _choose_action(
                q_table: dict[int, list[float]],
                state: int,
                action_count: int,
                epsilon: float,
                rng: random.Random,
            ) -> int:
                if rng.random() < epsilon:
                    return rng.randrange(action_count)
                q_values = _get_q_values(q_table, state, action_count)
                return max(range(action_count), key=lambda idx: q_values[idx])


            def train(episodes: int = 300, alpha: float = 0.25, gamma: float = 0.95) -> dict:
                env = Environment()
                rng = random.Random(0)
                q_table: dict[int, list[float]] = {}
                returns: list[float] = []

                for _ in range(episodes):
                    state = env.reset()
                    total_reward = 0.0
                    terminated = False
                    truncated = False

                    while not (terminated or truncated):
                        action = _choose_action(q_table, state, env.action_count, 0.2, rng)
                        next_state, reward, terminated, truncated, _ = env.step(action)

                        current_q = _get_q_values(q_table, state, env.action_count)
                        next_q = _get_q_values(q_table, next_state, env.action_count)
                        td_target = reward + gamma * max(next_q)
                        current_q[action] += alpha * (td_target - current_q[action])

                        state = next_state
                        total_reward += reward

                    returns.append(total_reward)

                trailing = returns[-20:] if returns else [0.0]
                results = {
                    "episodes": episodes,
                    "average_return": sum(returns) / max(1, len(returns)),
                    "last_20_avg_return": sum(trailing) / len(trailing),
                    "num_states_seen": len(q_table),
                }
                Path("training_results.json").write_text(
                    json.dumps(results, indent=2),
                    encoding="utf-8",
                )
                return results


            if __name__ == "__main__":
                output = train()
                print("Training completed.")
                print(json.dumps(output, indent=2))
            """
        )


def _parse_key_values(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    payload = _parse_json_payload(text)

    for raw_key, value in payload.items():
        canonical = _canonical_field_name(raw_key)
        if canonical and isinstance(value, (str, int, float)):
            parsed[canonical] = str(value).strip()

    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-* ").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        canonical = _canonical_field_name(key)
        if canonical:
            parsed[canonical] = value.strip()

    return parsed


def _parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


def _normalize_details_update(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    updates: dict[str, str] = {}
    for key, value in raw.items():
        canonical = _canonical_field_name(str(key))
        if canonical is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            updates[canonical] = cleaned
    return updates


def _canonical_field_name(raw_key: str) -> Optional[str]:
    key = re.sub(r"[^a-z0-9]+", "_", raw_key.strip().lower()).strip("_")
    for canonical, aliases in FIELD_ALIASES.items():
        if key == canonical or key in aliases:
            return canonical
    return None


def _is_train_intent(text: str) -> bool:
    return bool(re.search(r"\b(train|run training|start training)\b", text, re.IGNORECASE))


def _normalize_actions(action_space: str) -> list[str]:
    if not action_space.strip():
        return ["left", "right", "stay"]
    candidates = re.split(r"[,|/]+", action_space)
    cleaned_actions: list[str] = []
    for candidate in candidates:
        cleaned = re.sub(r"[^A-Za-z0-9_\- ]+", "", candidate).strip().lower()
        if cleaned:
            cleaned_actions.append(cleaned.replace(" ", "_"))
    if not cleaned_actions:
        return ["left", "right", "stay"]

    deduped: list[str] = []
    seen = set()
    for action in cleaned_actions:
        if action not in seen:
            seen.add(action)
            deduped.append(action)
    return deduped
