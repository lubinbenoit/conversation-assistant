from assistant.context.base import Context, ContextManager
from assistant.conversation import Conversation


class StatelessContext(ContextManager):
    name = "stateless"

    def build(self, conversation: Conversation) -> Context:
        history = conversation.messages

        if len(history) < 2 or history[-1].role != "user":
            raise ValueError("Context must be built immediately after a user message.")
        
        selected = [history[0], history[-1]]
        
        return Context(
            messages=selected,
        )
