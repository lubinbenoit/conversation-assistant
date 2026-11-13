from assistant.context.base import Context, ContextManager
from assistant.conversation import Conversation


class RecentWindowContext(ContextManager):
    """Return stored ``Message`` objects for the required window."""

    name = "recent_window"

    def build(self, conversation: Conversation) -> Context:
        raise NotImplementedError("Implement recent-turn policy.")
