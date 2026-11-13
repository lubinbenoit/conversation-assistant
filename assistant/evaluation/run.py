import argparse
import json
from pathlib import Path

from assistant.config import load_config
from assistant.context import make_context_manager
from assistant.context.base import run_maintenance
from assistant.conversation import Conversation
from assistant.evaluation.scenario import load_scenario
from assistant.model import ModelClient
from assistant.maintenance import MaintenanceRecorder


def run_scenario(scenario_path, config_path, policy, output):
    config = load_config(config_path)
    scenario = load_scenario(scenario_path)
    maintenance_thinking = config.model.think in {"maintenance", "all"}
    model = ModelClient(config.model)
    maintenance_recorder = MaintenanceRecorder(
        model.chat, thinking_enabled=maintenance_thinking
    )
    manager = make_context_manager(policy, model_call=maintenance_recorder, recent_turns=config.context.recent_turns)
    conversation = Conversation(config.assistant.system_prompt)
    records = []
    for n, user_text in enumerate(scenario.turns, 1):
        conversation.add_user(user_text)
        context = manager.build(conversation)
        manager_state_before = manager.state()
        model_messages = [
            message.as_model_message() for message in context.messages
        ]
        result = model.chat(model_messages)
        conversation.add_assistant(result.text)
        maintenance_recorder.reset()
        maintenance = run_maintenance(manager, conversation)
        u = result.usage
        records.append({
            "turn": n, "user": user_text, "assistant": result.text,
            "num_ctx": u.num_ctx, "prompt_eval_count": u.prompt_eval_count,
            "eval_count": u.eval_count, "prompt_utilization": u.prompt_utilization,
            "done_reason": result.done_reason,
            "source_ids": [
                message.id for message in context.messages
                if message.id is not None
            ],
            "messages": model_messages,
            "manager_state_before": manager_state_before,
            "manager_state_after": maintenance.state,
            "maintenance_error": maintenance.error,
            "maintenance_calls": maintenance_recorder.summary().as_dict(),
        })
        pct = "?" if u.prompt_utilization is None else f"{100*u.prompt_utilization:.1f}%"
        print(f"[{n}/{len(scenario.turns)}] prompt={u.prompt_eval_count} tokens ({pct})")

    output = Path(output); 
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({
            "scenario": scenario.name, 
            "policy": policy,
            "model": config.model.name,
            "num_ctx": config.model.num_ctx, 
            "recent_turns": config.context.recent_turns,
            "num_predict": config.model.num_predict,
            "temperature": config.model.temperature,
            "seed": config.model.seed,
            "thinking_mode": config.model.think,
            "records": records,
        }, ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    print(f"Wrote {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario", type=Path)
    p.add_argument("--config", default="config.toml")
    p.add_argument("--policy", required=True, choices=["stateless", "full_history", "recent_window", "managed"])
    p.add_argument("--output", type=Path, default=Path("logs/evaluation.json"))
    a = p.parse_args()
    run_scenario(a.scenario, a.config, a.policy, a.output)


if __name__ == "__main__": main()
