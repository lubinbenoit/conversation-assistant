from assistant.context.base import Context, ContextManager
from assistant.conversation import Conversation


class FullHistoryContext(ContextManager):
    """Return every stored ``Message`` in chronological order."""

    name = "full_history"

    def build(self, conversation: Conversation) -> Context:
        history = conversation.messages

        if len(history) < 2 or history[-1].role != "user":
            raise ValueError("Context must be built immediately after a user message.")
        
        return Context(messages=history)
