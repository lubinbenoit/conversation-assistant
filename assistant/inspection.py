from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from assistant.maintenance import MaintenanceSummary


@dataclass(frozen=True)
class InspectionRecord:
    """In-memory details for explaining the most recent model call."""

    turn: int
    policy: str
    num_ctx: int
    prompt_eval_count: int | None
    eval_count: int | None
    done_reason: str | None
    history_messages: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    manager_state_before: dict[str, Any]
    manager_state_after: dict[str, Any]
    num_predict: int | None = None
    maintenance_error: str | None = None
    maintenance_calls: MaintenanceSummary = field(
        default_factory=MaintenanceSummary
    )

    @property
    def manager_state(self) -> dict[str, Any]:
        return self.manager_state_after
