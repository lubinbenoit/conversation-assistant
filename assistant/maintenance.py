from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MaintenanceCall:
    prompt_eval_count: int | None
    eval_count: int | None
    thinking_enabled: bool | None
    done_reason: str | None
    error: str | None = None


@dataclass(frozen=True)
class MaintenanceSummary:
    calls: tuple[MaintenanceCall, ...] = ()

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def successful_call_count(self) -> int:
        return sum(call.error is None for call in self.calls)

    @property
    def failed_call_count(self) -> int:
        return sum(call.error is not None for call in self.calls)

    @staticmethod
    def _total(calls: tuple[MaintenanceCall, ...], field: str) -> int | None:
        successful = [call for call in calls if call.error is None]
        values = [getattr(call, field) for call in successful]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    @property
    def prompt_eval_count(self) -> int | None:
        return self._total(self.calls, "prompt_eval_count")

    @property
    def eval_count(self) -> int | None:
        return self._total(self.calls, "eval_count")

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "successful_call_count": self.successful_call_count,
            "failed_call_count": self.failed_call_count,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "calls": [asdict(call) for call in self.calls],
        }


class MaintenanceRecorder:
    """Record calls made through a context manager's maintenance API."""

    def __init__(
        self,
        model_call: Callable[..., Any] | None,
        *,
        thinking_enabled: bool = False,
    ):
        self._model_call = model_call
        self.thinking_enabled = thinking_enabled
        self._calls: list[MaintenanceCall] = []

    def set_think(self, enabled: bool) -> None:
        self.thinking_enabled = enabled

    def reset(self) -> None:
        self._calls.clear()

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        format: str | dict[str, Any] | None = None,
    ) -> Any:
        if self._model_call is None:
            raise RuntimeError("No maintenance model call is configured.")
        try:
            result = self._model_call(
                messages,
                format=format,
                think=self.thinking_enabled,
            )
        except Exception as exc:
            self._calls.append(
                MaintenanceCall(
                    prompt_eval_count=None,
                    eval_count=None,
                    thinking_enabled=self.thinking_enabled,
                    done_reason=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

        usage = getattr(result, "usage", None)
        self._calls.append(
            MaintenanceCall(
                prompt_eval_count=getattr(usage, "prompt_eval_count", None),
                eval_count=getattr(usage, "eval_count", None),
                thinking_enabled=self.thinking_enabled,
                done_reason=getattr(result, "done_reason", None),
            )
        )
        return result

    def summary(self) -> MaintenanceSummary:
        return MaintenanceSummary(tuple(self._calls))
