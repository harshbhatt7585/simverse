from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class AgentTurn:
    status: str
    reply: str


def create_system_prompt(name: str) -> str:
    return (
        "You are a helpful agent who is an expert in building RL environments.\n"
        "You will converse with the user to understand their needs and build the "
        "environment accordingly.\n"
        "When the user provides all the details, use SimVerse abstractions to build "
        "the RL environment in its framework.\n"
        "You will write `env.py` with environment logic, `render.py` with render "
        "code, and `train.py` with training code.\n"
        "You can navigate to the `simverse-web` directory to reference existing code "
        "and build accordingly.\n"
        "Continue until the environment is built and training is working.\n"
        "After the environment is built, ask the user if they want to train the "
        "environment.\n"
        "If they say train, run training, show results, and return the results"
    )


class SimpleTerminalAgent:
    def __init__(
        self,
        name: str,
        workspace: Path,
        model: str = "gpt-5-nano",
    ) -> None:
        self.name = name
        self.workspace = workspace
        self.model = model
        self.client = self._build_client()
        self.history: list[dict[str, str]] = []
        self.system_prompt = (
            f"You are {name}. Keep replies concise, practical, and helpful. "
            "You are chatting with the user in a terminal."
        )

    def _build_client(self) -> OpenAI | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return OpenAI()

    def handle_user_message(self, user_input: str) -> AgentTurn:
        text = user_input.strip()
        if not text:
            return AgentTurn(status="idle", reply="Say something and I will reply.")

        if self.client is None:
            return AgentTurn(
                status="error",
                reply="OpenAI client unavailable. Install `openai` and set `OPENAI_API_KEY`.",
            )

        self.history.append({"role": "user", "content": text})

        messages = [{"role": "system", "content": create_system_prompt(self.name)}] + self.history[
            -20:
        ]
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            reasoning={"effort": "minimal"},
            max_output_tokens=600,
        )
        reply = (getattr(response, "output_text", "") or "").strip()
        if not reply:
            reply = "I could not generate a response. Please try again."
            status = "error"
        else:
            status = "ok"

        self.history.append({"role": "assistant", "content": reply})
        return AgentTurn(status=status, reply=reply)


def create_agent(name: str, workspace: Path, model: str = "gpt-5-nano") -> SimpleTerminalAgent:
    return SimpleTerminalAgent(name=name, workspace=workspace, model=model)


def run_cli() -> None:
    agent = create_agent(name="SimVerse Assistant", workspace=Path(__file__).resolve().parent)
    print("Simple Terminal Agent")
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
