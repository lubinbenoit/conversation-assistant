from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    id: int | None
    role: Role
    content: str

    def as_model_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Conversation:
    def __init__(self, system_prompt: str):
        self._messages = [Message(id=0, role="system", content=system_prompt)]
        self._next_id = 1

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def last_message(self) -> Message:
        return self._messages[-1]

    def add_user(self, content: str) -> Message:
        return self._append("user", content)

    def add_assistant(self, content: str) -> Message:
        return self._append("assistant", content)

    def reset(self) -> None:
        system = self._messages[0]
        self._messages = [system]
        self._next_id = 1

    def discard_last(self, expected: Message) -> None:
        if self._messages[-1] != expected:
            raise RuntimeError("Cannot discard: conversation has advanced.")
        if expected.id == 0:
            raise RuntimeError("Cannot discard the system message.")
        self._messages.pop()
        self._next_id = expected.id

    def _append(self, role: Role, content: str) -> Message:
        message = Message(id=self._next_id, role=role, content=content)
        self._messages.append(message)
        self._next_id += 1
        return message
