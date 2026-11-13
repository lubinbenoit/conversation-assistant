from assistant.context.base import Context, ContextManager
from assistant.conversation import Conversation


class FullHistoryContext(ContextManager):
    """Return every stored ``Message`` in chronological order."""

    name = "full_history"

    def build(self, conversation: Conversation) -> Context:
        raise NotImplementedError("Implement full-history policy.")
