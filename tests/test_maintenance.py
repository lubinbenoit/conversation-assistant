import pytest

from assistant.maintenance import MaintenanceRecorder
from assistant.model import ModelResult, Usage


def result(prompt_tokens=10, output_tokens=5, done_reason="stop"):
    return ModelResult(
        text="state",
        usage=Usage(
            prompt_tokens,
            output_tokens,
            None,
            None,
            None,
            None,
            4096,
        ),
        done_reason=done_reason,
    )


def test_recorder_distinguishes_no_call_from_one_observed_call():
    captured = {}

    def model_call(*args, **kwargs):
        captured.update(kwargs)
        return result()

    recorder = MaintenanceRecorder(model_call, thinking_enabled=True)

    assert recorder.summary().call_count == 0

    recorder([], format="json")
    summary = recorder.summary()

    assert summary.call_count == 1
    assert summary.successful_call_count == 1
    assert summary.failed_call_count == 0
    assert summary.prompt_eval_count == 10
    assert summary.eval_count == 5
    assert summary.calls[0].thinking_enabled is True
    assert summary.calls[0].done_reason == "stop"
    assert captured == {"format": "json", "think": True}


def test_recorder_aggregates_multiple_calls_and_resets_per_turn():
    results = iter((result(10, 2), result(15, 3, "length")))
    recorder = MaintenanceRecorder(lambda *args, **kwargs: next(results))

    recorder([])
    recorder.set_think(True)
    recorder([])
    summary = recorder.summary()

    assert summary.call_count == 2
    assert summary.prompt_eval_count == 25
    assert summary.eval_count == 5
    assert [call.thinking_enabled for call in summary.calls] == [False, True]
    assert [call.done_reason for call in summary.calls] == ["stop", "length"]

    recorder.reset()
    assert recorder.summary().call_count == 0


def test_recorder_preserves_and_records_a_failed_call():
    def fail(*args, **kwargs):
        raise RuntimeError("Ollama unavailable")

    recorder = MaintenanceRecorder(fail)

    with pytest.raises(RuntimeError, match="Ollama unavailable"):
        recorder([])

    summary = recorder.summary()
    assert summary.call_count == 1
    assert summary.successful_call_count == 0
    assert summary.failed_call_count == 1
    assert summary.prompt_eval_count is None
    assert summary.eval_count is None
    assert summary.calls[0].thinking_enabled is False
    assert summary.calls[0].error == "RuntimeError: Ollama unavailable"
