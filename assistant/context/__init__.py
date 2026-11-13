from assistant.context.base import Context, ContextManager, ModelCall
from assistant.context.full_history import FullHistoryContext
from assistant.context.managed import ManagedContext
from assistant.context.recent_window import RecentWindowContext
from assistant.context.stateless import StatelessContext

POLICIES = {
    "stateless": StatelessContext,
    "full_history": FullHistoryContext,
    "recent_window": RecentWindowContext,
    "managed": ManagedContext,
}


def make_context_manager(name: str, *, model_call=None, recent_turns: int = 3):
    try:
        cls = POLICIES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(POLICIES))
        raise ValueError(f"Unknown context policy {name!r}. Choose one of: {valid}") from exc
    return cls(model_call=model_call, recent_turns=recent_turns)
