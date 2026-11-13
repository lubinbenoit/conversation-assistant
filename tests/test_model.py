import json

import pytest

from assistant.config import ModelConfig
from assistant.model import ModelClient


class FakeResponse:
    def __init__(self, body): self.body = json.dumps(body).encode()
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return self.body


class FakeStreamResponse:
    def __init__(self, events):
        self.lines = [f"{json.dumps(event)}\n".encode() for event in events]

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __iter__(self): return iter(self.lines)


def model_config():
    return ModelConfig(
        "test", "http://localhost:11434", 400, 50, 0.0, 42, "off", 10
    )


def test_ollama_usage_is_exposed(monkeypatch):
    body = {"message": {"role": "assistant", "content": "hello"}, "done_reason": "stop",
            "prompt_eval_count": 100, "eval_count": 20, "prompt_eval_duration": 1,
            "eval_duration": 2, "total_duration": 3, "load_duration": 0}
    monkeypatch.setattr("assistant.model.urlopen", lambda *a, **k: FakeResponse(body))
    cfg = ModelConfig("test", "http://localhost:11434", 400, 50, 0.0, 42, "off", 10)
    result = ModelClient(cfg).chat([{"role": "user", "content": "hi"}])
    assert result.text == "hello"
    assert result.usage.prompt_eval_count == 100
    assert result.usage.eval_count == 20
    assert result.usage.prompt_utilization == 0.25


def test_request_sets_ollama_context_window(monkeypatch):
    captured = {}
    body = {"message": {"role": "assistant", "content": "ok"}, "prompt_eval_count": 10, "eval_count": 2}
    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(body)
    monkeypatch.setattr("assistant.model.urlopen", fake_urlopen)
    cfg = ModelConfig("test", "http://localhost:11434", 2048, 123, 0.1, 7, "off", 10)
    ModelClient(cfg).chat([{"role": "user", "content": "hi"}])
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["num_ctx"] == 2048
    assert captured["payload"]["options"]["num_predict"] == 123


def test_runtime_settings_change_subsequent_request_payloads(monkeypatch):
    captured = {}
    body = {"message": {"role": "assistant", "content": "ok"}}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(body)

    monkeypatch.setattr("assistant.model.urlopen", fake_urlopen)
    client = ModelClient(model_config())
    client.set_think(True)
    client.set_num_predict(750)

    client.chat([{"role": "user", "content": "hi"}])

    assert captured["payload"]["think"] is True
    assert captured["payload"]["options"]["num_predict"] == 750


def test_non_streaming_call_can_override_thinking_for_maintenance(monkeypatch):
    captured = {}
    body = {"message": {"role": "assistant", "content": "ok"}}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(body)

    monkeypatch.setattr("assistant.model.urlopen", fake_urlopen)
    client = ModelClient(model_config())

    client.chat([{"role": "user", "content": "hi"}], think=True)

    assert captured["payload"]["think"] is True


def test_connection_timeout_is_normalized(monkeypatch):
    def time_out(*args, **kwargs):
        raise TimeoutError("connection timed out")

    monkeypatch.setattr("assistant.model.urlopen", time_out)

    with pytest.raises(RuntimeError, match="Cannot reach Ollama") as error:
        ModelClient(model_config()).chat([{"role": "user", "content": "hi"}])

    assert isinstance(error.value.__cause__, TimeoutError)


def test_response_read_timeout_is_normalized(monkeypatch):
    class TimedOutResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            raise TimeoutError("response timed out")

    monkeypatch.setattr(
        "assistant.model.urlopen", lambda *args, **kwargs: TimedOutResponse()
    )

    with pytest.raises(RuntimeError, match="Cannot reach Ollama") as error:
        ModelClient(model_config()).chat([{"role": "user", "content": "hi"}])

    assert isinstance(error.value.__cause__, TimeoutError)


def test_streaming_chat_emits_chunks_and_returns_usage(monkeypatch):
    events = [
        {"message": {"content": "Hello "}, "done": False},
        {
            "message": {"content": "world"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 2,
        },
    ]
    monkeypatch.setattr(
        "assistant.model.urlopen", lambda *args, **kwargs: FakeStreamResponse(events)
    )
    chunks = []

    result = ModelClient(model_config()).chat_stream(
        [{"role": "user", "content": "hi"}], chunks.append
    )

    assert chunks == ["Hello ", "world"]
    assert result.text == "Hello world"
    assert result.usage.prompt_eval_count == 12
    assert result.done_reason == "stop"


def test_streaming_chat_emits_thinking_separately(monkeypatch):
    events = [
        {"message": {"thinking": "Working "}, "done": False},
        {"message": {"thinking": "it out"}, "done": False},
        {
            "message": {"content": "Answer"},
            "done": True,
            "done_reason": "stop",
        },
    ]
    monkeypatch.setattr(
        "assistant.model.urlopen", lambda *args, **kwargs: FakeStreamResponse(events)
    )
    answers = []
    thoughts = []

    result = ModelClient(model_config()).chat_stream(
        [], answers.append, on_thinking=thoughts.append
    )

    assert thoughts == ["Working ", "it out"]
    assert answers == ["Answer"]
    assert result.text == "Answer"


def test_streaming_chat_surfaces_midstream_error(monkeypatch):
    events = [
        {"message": {"content": "partial"}, "done": False},
        {"error": "generation failed"},
    ]
    monkeypatch.setattr(
        "assistant.model.urlopen", lambda *args, **kwargs: FakeStreamResponse(events)
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        ModelClient(model_config()).chat_stream([], lambda chunk: None)
