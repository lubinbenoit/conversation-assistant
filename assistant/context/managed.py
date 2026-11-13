from assistant.context.base import Context, ContextManager
from assistant.conversation import Conversation, Message


class ManagedContext(ContextManager):
    """Preserve useful older information without replaying full history.

    You may change this class freely. A managed policy may keep internal state,
    implement ``update()`` and ``reset()``, expose inspectable data through
    ``state()``, and optionally use ``self.model_call`` for context maintenance.
    That callable is the non-streaming
    ``model_call(messages, format=...)`` API and returns a ``ModelResult``. The
    application applies the runtime maintenance-thinking setting. This
    maintenance call is separate from the main call that the application makes
    for the visible assistant response. Any state belonging to one conversation
    must be cleared by ``reset()``.

    ``Context.messages`` contains ``Message`` objects, not model-request
    dictionaries. Preserve a stored message's ID when returning it, including
    when changing its content. Use ``id=None`` only for a new message that has
    no counterpart in the transcript. The application removes IDs immediately
    before calling Ollama.

    Let maintenance exceptions propagate from ``update()``. The application
    catches and reports them without discarding the completed assistant answer.
    """

    name = "managed"

    def build(self, conversation: Conversation) -> Context:
        raise NotImplementedError("Implement managed policy.")
