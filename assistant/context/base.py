from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from assistant.conversation import Conversation, Message

ModelCall = Callable[..., Any]


@dataclass(frozen=True)
class Context:
    messages: list[Message]


@dataclass(frozen=True)
class MaintenanceOutcome:
    state: dict[str, Any]
    error: str | None = None


class ContextManager(ABC):
    name: str

    def __init__(self, model_call: ModelCall | None = None, recent_turns: int = 3):
        """Initialize a context policy.

        ``model_call`` is the non-streaming ``ModelClient.chat`` method supplied
        for optional context-maintenance work. It accepts model messages plus
        an optional ``format`` argument and returns a ``ModelResult``. The
        application supplies the active maintenance-thinking setting.
        The application owns the separate call that generates the user-visible
        assistant response.
        """
        if recent_turns < 0:
            raise ValueError("recent_turns must be non-negative")
        self.model_call = model_call
        self.recent_turns = recent_turns

    @abstractmethod
    def build(self, conversation: Conversation) -> Context:
        """Select the messages to send on the next model call."""
        raise NotImplementedError

    def update(self, conversation: Conversation) -> None:
        """Update persistent in-session state after an assistant turn."""

    def reset(self) -> None:
        """Clear state associated with the current conversation."""

    def state(self) -> dict[str, Any]:
        return {}

def run_maintenance(
    manager: ContextManager, conversation: Conversation
) -> MaintenanceOutcome:
    """Run post-answer policy maintenance without invalidating the answer."""
    error = None
    try:
        manager.update(conversation)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    try:
        state = manager.state()
    except Exception as exc:
        state_error = f"{type(exc).__name__}: {exc}"
        error = (
            state_error
            if error is None
            else f"{error}; state inspection failed: {state_error}"
        )
        state = {}
    return MaintenanceOutcome(state, error)
