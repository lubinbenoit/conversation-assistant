from assistant.context.base import Context, ContextManager
from assistant.conversation import Conversation


class RecentWindowContext(ContextManager):
    """Return stored ``Message`` objects for the required window."""

    name = "recent_window"

    def build(self, conversation: Conversation) -> Context:
        history = conversation.messages

        if len(history) < 2 or history[-1].role != "user":
            raise ValueError("Context must be built immediately after a user message.")

        start = max(1, len(history) - 2 * self.recent_turns - 1)
        selected = [history[0]] + history[start:]

        return Context(messages=selected)
