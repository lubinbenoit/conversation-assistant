import json
from types import SimpleNamespace

import pytest

from assistant.application import ChatApplication, truncation_notice
from assistant.context import StatelessContext
from assistant.inspector import format_inspector
from assistant.model import ModelResult, Usage


def model_result(text: str = "response") -> ModelResult:
    return ModelResult(
        text=text,
        usage=Usage(
            prompt_eval_count=10,
            eval_count=5,
            prompt_eval_duration=None,
            eval_duration=None,
            total_duration=None,
            load_duration=None,
            num_ctx=4096,
        ),
        done_reason="stop",
    )


def config(tmp_path):
    return SimpleNamespace(
        model=SimpleNamespace(
            name="test-model",
            host="http://localhost:11434",
            num_ctx=4096,
            num_predict=384,
            temperature=0.2,
            seed=7,
            think="off",
            timeout_seconds=30,
        ),
        assistant=SimpleNamespace(system_prompt="system"),
        context=SimpleNamespace(policy="stateless", recent_turns=3),
        trace=SimpleNamespace(directory=tmp_path),
    )


class SuccessfulModel:
    def chat(self, messages, **kwargs):
        return model_result()

    def chat_stream(self, messages, on_chunk):
        on_chunk("response")
        return model_result()


class CapturingModel(SuccessfulModel):
    def __init__(self):
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return super().chat(messages)

class FailingOnceModel:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("model unavailable")
        return model_result()


class FailingOnceContext(StatelessContext):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def build(self, conversation):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("cannot build context")
        return super().build(conversation)


class FailingMaintenanceContext(StatelessContext):
    def update(self, conversation):
        raise RuntimeError("maintenance unavailable")


class CallingMaintenanceContext(StatelessContext):
    def update(self, conversation):
        self.model_call([])
        self.model_call([])


@pytest.mark.parametrize(
    ("model", "manager", "message"),
    [
        (SuccessfulModel(), FailingOnceContext(), "cannot build context"),
        (FailingOnceModel(), StatelessContext(), "model unavailable"),
    ],
)
def test_failed_turn_rolls_back_and_can_be_retried(
    tmp_path, model, manager, message
):
    app = ChatApplication(config(tmp_path), model, manager)

    with pytest.raises((RuntimeError, ValueError), match=message):
        app.handle_message("first attempt")

    assert [m.id for m in app.conversation.messages] == [0]
    assert [
        record["content"]
        for line in app.trace.path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line))["type"] == "message"
    ] == ["system"]
    assert app.turn == 0
    assert app.last_inspection is None

    assert app.handle_message("retry") == "response"
    assert [m.id for m in app.conversation.messages] == [0, 1, 2]
    assert app.turn == 1
    assert app.last_inspection is not None
    assert app.last_inspection.turn == 1


def test_reset_starts_new_trace_session(tmp_path):
    app = ChatApplication(config(tmp_path), SuccessfulModel(), StatelessContext())
    app.handle_message("before reset")
    previous_trace_path = app.trace.path

    app.reset()

    assert [m.id for m in app.conversation.messages] == [0]
    assert app.turn == 0
    assert app.trace.path != previous_trace_path
    assert app.last_inspection is None
    assert previous_trace_path.exists()


def test_context_ids_are_removed_only_at_model_boundary(tmp_path):
    model = CapturingModel()
    app = ChatApplication(config(tmp_path), model, StatelessContext())

    app.handle_message("hello")

    assert model.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]
    assert app.last_inspection.messages == [
        {"id": 0, "role": "system", "content": "system"},
        {"id": 1, "role": "user", "content": "hello"},
    ]


def test_rapid_resets_use_unique_trace_paths(tmp_path):
    app = ChatApplication(config(tmp_path), SuccessfulModel(), StatelessContext())
    paths = {app.trace.path}

    for _ in range(5):
        app.reset()
        paths.add(app.trace.path)

    assert len(paths) == 6


def test_streamed_turn_is_committed_and_traced(tmp_path):
    app = ChatApplication(config(tmp_path), SuccessfulModel(), StatelessContext())
    chunks = []

    response = app.handle_message_stream("hello\n🌍", chunks.append)

    assert response == "response"
    assert chunks == ["response"]
    assert [message.content for message in app.conversation.messages] == [
        "system", "hello\n🌍", "response"
    ]
    assert app.last_inspection is not None
    assert [
        message["id"] for message in app.last_inspection.history_messages
    ] == [0, 1]
    assert [
        message["content"] for message in app.last_inspection.history_messages
    ] == ["system", "hello\n🌍"]
    rendered_inspector = format_inspector(app.last_inspection)
    assert "· assistant · dropped" not in rendered_inspector
    assert "response" not in rendered_inspector

    records = [
        json.loads(line)
        for line in app.trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0] == {
        "type": "session",
        "timestamp": records[0]["timestamp"],
        "schema_version": 1,
        "model": {
            "name": "test-model",
            "host": "http://localhost:11434",
            "num_ctx": 4096,
            "temperature": 0.2,
            "seed": 7,
            "timeout_seconds": 30,
        },
        "context": {"policy": "stateless", "recent_turns": 3},
        "initial_settings": {"thinking_mode": "off", "num_predict": 384},
    }
    messages = [record for record in records if record["type"] == "message"]
    assert [(record["id"], record["role"], record["content"]) for record in messages] == [
        (0, "system", "system"),
        (1, "user", "hello\n🌍"),
        (2, "assistant", "response"),
    ]
    assert all(
        set(record) == {"type", "timestamp", "id", "role", "content"}
        for record in messages
    )


def test_streamed_turn_signals_policy_maintenance_phase(tmp_path):
    app = ChatApplication(config(tmp_path), SuccessfulModel(), StatelessContext())
    phases = []

    app.handle_message_stream(
        "hello", lambda chunk: None, on_maintenance=phases.append
    )

    assert phases == [True, False]


def test_maintenance_failure_preserves_answer_trace_and_next_turn(tmp_path):
    app = ChatApplication(
        config(tmp_path), SuccessfulModel(), FailingMaintenanceContext()
    )
    phases = []

    assert app.handle_message_stream(
        "first", lambda chunk: None, on_maintenance=phases.append
    ) == "response"

    assert phases == [True, False]
    assert [message.content for message in app.conversation.messages] == [
        "system", "first", "response"
    ]
    assert app.last_inspection.maintenance_error == (
        "RuntimeError: maintenance unavailable"
    )

    assert app.handle_message("second") == "response"
    assert app.turn == 2
    assert [message.content for message in app.conversation.messages][-3:] == [
        "response", "second", "response"
    ]
    records = [
        json.loads(line)
        for line in app.trace.path.read_text(encoding="utf-8").splitlines()
    ]
    messages = [record for record in records if record["type"] == "message"]
    assert [record["id"] for record in messages] == [0, 1, 2, 3, 4]
    assert [record["content"] for record in messages] == [
        "system", "first", "response", "second", "response"
    ]


def test_truncation_notice_only_for_length_completion():
    assert truncation_notice(384, "stop") is None
    assert truncation_notice(384, None) is None
    assert truncation_notice(384, "length") == (
        "Output stopped at the configured limit of 384 tokens. "
        "Increase the runtime output limit or [model].num_predict in config.toml."
    )


def test_runtime_settings_are_recorded_per_turn(tmp_path):
    app = ChatApplication(config(tmp_path), SuccessfulModel(), StatelessContext())
    app.set_think("maintenance")
    app.set_num_predict(700)

    app.handle_message("hello")

    assert app.last_inspection.num_predict == 700
    records = [
        json.loads(line)
        for line in app.trace.path.read_text(encoding="utf-8").splitlines()
    ]
    settings = [record for record in records if record["type"] == "settings"]
    assert settings == [
        {
            "type": "settings",
            "timestamp": settings[0]["timestamp"],
            "thinking_mode": "maintenance",
            "num_predict": 384,
        },
        {
            "type": "settings",
            "timestamp": settings[1]["timestamp"],
            "thinking_mode": "maintenance",
            "num_predict": 700,
        },
    ]


def test_application_records_observed_maintenance_calls(tmp_path):
    maintenance_model = SuccessfulModel()
    manager = CallingMaintenanceContext(model_call=maintenance_model.chat)
    app = ChatApplication(config(tmp_path), SuccessfulModel(), manager)
    app.set_think("maintenance")

    app.handle_message("hello")

    summary = app.last_inspection.maintenance_calls
    assert summary.call_count == 2
    assert summary.prompt_eval_count == 20
    assert summary.eval_count == 10
    assert [call.thinking_enabled for call in summary.calls] == [True, True]


def test_runtime_setting_overrides_remain_validated(tmp_path):
    app = ChatApplication(config(tmp_path), SuccessfulModel(), StatelessContext())

    with pytest.raises(ValueError, match="invalid thinking mode"):
        app.set_think("sometimes")
    with pytest.raises(ValueError, match="greater than zero"):
        app.set_num_predict(0)

    assert app.active_think == "off"
    assert app.active_num_predict == 384
