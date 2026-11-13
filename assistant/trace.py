from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from assistant.conversation import Message


class TraceWriter:
    def __init__(
        self,
        directory: Path,
        session_id: str,
        system_message: Message,
        metadata: dict[str, Any],
    ):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{session_id}.jsonl"
        self._write(
            {
                "type": "session",
                "timestamp": self._timestamp(),
                "schema_version": 1,
                **metadata,
            }
        )
        self.append(system_message)

    def append(self, message: Message) -> None:
        self._write(
            {
                "type": "message",
                "timestamp": self._timestamp(),
                "id": message.id,
                "role": message.role,
                "content": message.content,
            }
        )

    def append_turn(self, user: Message, assistant: Message) -> None:
        self.append(user)
        self.append(assistant)

    def append_settings(self, *, thinking_mode: str, num_predict: int) -> None:
        self._write(
            {
                "type": "settings",
                "timestamp": self._timestamp(),
                "thinking_mode": thinking_mode,
                "num_predict": num_predict,
            }
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
