from __future__ import annotations

from pathlib import Path

from agent import create_agent


def run_cli() -> None:
    workspace = Path(__file__).resolve().parent
    agent = create_agent(name="SimVerse RL Builder", workspace=workspace)

    print("SimVerse RL Builder")
    print("Provide environment details using lines like `objective: ...`.")
    print("When scaffold generation is complete, type `train` to run training.")
    print("Type `exit` to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        turn = agent.handle_user_message(user_input)
        print(f"\nAgent ({turn.status}): {turn.reply}\n")


if __name__ == "__main__":
    run_cli()
