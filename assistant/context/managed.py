from assistant.context.base import Context, ContextManager
from assistant.conversation import Conversation, Message
import json

MAX_ITEMS = 10
MAX_VALUE_CHARS = 300

MAINTENANCE_PROMPT = """You maintain a short memory of durable user instructions for a study assistant.
You do not answer the user. You only decide how the memory must change.

INPUT
- "Current memory": a JSON object mapping keys to instructions currently in force.
- "Latest user message": the newest message from the user.

WHAT TO KEEP
Keep only instructions that the user wants applied to FUTURE responses:
restrictions, preferences about form or style, notation conventions, required markers.

WHAT TO IGNORE
Questions, exercises, topics being studied, one-off requests that apply only to the
current answer, and any statement that does not change how future answers should be written.

OPERATIONS
- add: a new durable instruction that no existing key covers.
- replace: the user changes an instruction that already exists. Reuse the EXISTING key
  and give the full new value. Never add a second key for the same subject.
- remove: the user cancels or withdraws an existing instruction. Use its existing key.
If nothing changes, return {"operations": []}. This is the most common case.

VALUE RULES
- Write each value as a self-contained instruction addressed to the assistant.
- Keep the condition under which it applies (for example "When the user asks for ...").
- Keep exact symbols, markers and words chosen by the user.
- Keys are short snake_case names describing the subject, e.g. "code_language".
- Do not refer to previous values (avoid words like "instead" or "anymore").

EXAMPLES

Current memory: {}
Latest user message: "Please always write vectors in bold, like **v**."
{"operations": [{"op": "add", "key": "vector_notation", "value": "Write vectors in bold, like **v**."}]}

Current memory: {"vector_notation": "Write vectors in bold, like **v**."}
Latest user message: "Actually, write vectors with an arrow on top instead of bold."
{"operations": [{"op": "replace", "key": "vector_notation", "value": "Write vectors with an arrow on top instead of bold."}]}

Current memory: {"answer_language": "When answering, reply in French."}
Latest user message: "You can go back to answering in English."
{"operations": [{"op": "remove", "key": "answer_language"}]}

Current memory: {"vector_notation": "Write vectors in bold, like **v**."}
Latest user message: "What is the difference between a norm and a dot product?"
{"operations": []}
"""

MAINTENANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "replace", "remove"]},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["op", "key"],
            },
        }
    },
    "required": ["operations"],
}


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

    def __init__(self, model_call=None, recent_turns=3):
        super().__init__(model_call=model_call, recent_turns=recent_turns)
        self.memory = {}
        self.last_operations = []

    def build(self, conversation: Conversation) -> Context:
        history = conversation.messages
                
        if len(history) < 2 or history[-1].role != "user":
            raise ValueError("Context must be built immediately after a user message.")

        system = history[0]

        if self.memory:
            lines = "\n".join(f"- {value}" for value in self.memory.values())
            system = Message(
                id=system.id,
                role="system",
                content=system.content
                + "\n\nPersistent user instructions (still in force):\n" + lines,
            )

        start = max(1, len(history) - 2 * self.recent_turns - 1)
        selected = [system] + history[start:]
        return Context(messages=selected)

    
    def update(self, conversation: Conversation) -> None:
        if self.model_call is None:
            return

        history = conversation.messages
        user_messages = [m for m in history if m.role == "user"]
        if not user_messages:
            return
        latest = user_messages[-1]

        data = (
            "Current memory: " + json.dumps(self.memory, ensure_ascii=False)
            + "\nLatest user message: " + json.dumps(latest.content, ensure_ascii=False)
        )
        result = self.model_call(
            [
                {"role": "system", "content": MAINTENANCE_PROMPT},
                {"role": "user", "content": data},
            ],
            format=MAINTENANCE_SCHEMA,
        )

        if result.done_reason == "length":
            raise RuntimeError("Maintenance output was truncated by num_predict.")
        operations = json.loads(result.text)["operations"]  # ValueError/KeyError → propagées

        self.last_operations = []   # pour l'inspection : ce qui a réellement été appliqué
        for op in operations:
            self._apply(op)


    def _apply(self, op: dict) -> None:
        kind = op.get("op")
        key = str(op.get("key", "")).strip()
        value = str(op.get("value", "")).strip()[:MAX_VALUE_CHARS]
        if not key:
            return

        if kind in ("add", "replace"):
            if not value:
                return                                   # add/replace sans valeur : ignoré
            if key not in self.memory and len(self.memory) >= MAX_ITEMS:
                self.last_operations.append({"op": "rejected_full", "key": key})
                return                                   # borne atteinte : on refuse l'ajout
            applied = "replace" if key in self.memory else "add"
            self.memory[key] = value
            self.last_operations.append({"op": applied, "key": key, "value": value})

        elif kind == "remove":
            if key in self.memory:
                del self.memory[key]
                self.last_operations.append({"op": "remove", "key": key})
            # remove sur une clé inconnue : ignoré

    def state(self) -> dict:
        return {
            "memory": dict(self.memory),
            "last_operations": list(self.last_operations),
        }

    def reset(self) -> None:
        self.memory = {}
        self.last_operations = []