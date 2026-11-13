import argparse

from assistant.application import ChatApplication
from assistant.config import load_config
from assistant.context import make_context_manager
from assistant.model import ModelClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--policy", choices=["stateless", "full_history", "recent_window", "managed"])
    args = parser.parse_args()
    config = load_config(args.config)
    model = ModelClient(config.model)
    manager = make_context_manager(
        args.policy or config.context.policy,
        model_call=model.chat,
        recent_turns=config.context.recent_turns,
    )
    app = ChatApplication(config, model, manager)
    from assistant.tui import run_tui

    run_tui(app)


if __name__ == "__main__":
    main()
