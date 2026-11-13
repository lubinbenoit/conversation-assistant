import json
from types import SimpleNamespace

from assistant.context.base import Context
from assistant.conversation import Message
from assistant.evaluation.run import run_scenario
from assistant.evaluation.scenario import Scenario
from assistant.model import ModelResult, Usage


def test_evaluation_log_records_thinking_mode_and_effective_settings(
    tmp_path, monkeypatch
):
    config = SimpleNamespace(
        model=SimpleNamespace(
            name="test",
            num_ctx=4096,
            num_predict=1000,
            temperature=0.2,
            seed=42,
            think="maintenance",
        ),
        assistant=SimpleNamespace(system_prompt="system"),
        context=SimpleNamespace(recent_turns=3),
    )

    class Model:
        def __init__(self, model_config):
            pass

        def chat(self, messages, **kwargs):
            return ModelResult(
                text="answer",
                usage=Usage(10, 5, None, None, None, None, 4096),
                done_reason="stop",
            )

    class Manager:
        def __init__(self, model_call):
            self.model_call = model_call

        def build(self, conversation):
            return Context(
                messages=[Message(id=1, role="user", content="question")],
            )

        def update(self, conversation):
            self.model_call([])
            raise RuntimeError("maintenance unavailable")

        def state(self):
            return {}

    monkeypatch.setattr("assistant.evaluation.run.load_config", lambda path: config)
    monkeypatch.setattr(
        "assistant.evaluation.run.load_scenario",
        lambda path: Scenario("test scenario", "", ["question"]),
    )
    monkeypatch.setattr("assistant.evaluation.run.ModelClient", Model)
    monkeypatch.setattr(
        "assistant.evaluation.run.make_context_manager",
        lambda *args, **kwargs: Manager(kwargs["model_call"]),
    )
    output = tmp_path / "evaluation.json"

    run_scenario("scenario.yaml", "config.toml", "managed", output)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["model"] == "test"
    assert data["num_ctx"] == 4096
    assert data["num_predict"] == 1000
    assert data["temperature"] == 0.2
    assert data["seed"] == 42
    assert data["thinking_mode"] == "maintenance"
    assert data["recent_turns"] == 3
    assert "answer_thinking_enabled" not in data["records"][0]
    assert "maintenance_thinking_enabled" not in data["records"][0]
    assert "context_metadata" not in data["records"][0]
    assert data["records"][0]["maintenance_error"] == (
        "RuntimeError: maintenance unavailable"
    )
    assert data["records"][0]["maintenance_calls"] == {
        "call_count": 1,
        "successful_call_count": 1,
        "failed_call_count": 0,
        "prompt_eval_count": 10,
        "eval_count": 5,
        "calls": [{
            "prompt_eval_count": 10,
            "eval_count": 5,
            "thinking_enabled": True,
            "done_reason": "stop",
            "error": None,
        }],
    }
    
