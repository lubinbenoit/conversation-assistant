from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from assistant.config import ModelConfig


@dataclass(frozen=True)
class Usage:
    prompt_eval_count: int | None
    eval_count: int | None
    prompt_eval_duration: int | None
    eval_duration: int | None
    total_duration: int | None
    load_duration: int | None
    num_ctx: int

    @property
    def prompt_utilization(self) -> float | None:
        if self.prompt_eval_count is None or self.num_ctx <= 0:
            return None
        return self.prompt_eval_count / self.num_ctx


@dataclass(frozen=True)
class ModelResult:
    text: str
    usage: Usage
    done_reason: str | None


class ModelClient:
    """Minimal client for Ollama's native /api/chat endpoint."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.url = config.host.rstrip("/") + "/api/chat"
        self.think = config.think in {"answers", "all"}
        self.num_predict = config.num_predict

    def set_think(self, enabled: bool) -> None:
        self.think = enabled

    def set_num_predict(self, value: int) -> None:
        if value <= 0:
            raise ValueError("num_predict must be greater than zero")
        self.num_predict = value

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        format: str | dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> ModelResult:
        payload: dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            "stream": False,
            "think": self.think if think is None else think,
            "options": {
                "num_ctx": self.config.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.config.temperature,
                "seed": self.config.seed,
            },
        }
        if format is not None:
            payload["format"] = format

        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.config.host}. "
                "Check that `ollama serve` is running."
            ) from exc

        try:
            text = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama chat response: {body!r}") from exc

        return ModelResult(
            text=text,
            usage=self._usage(body),
            done_reason=body.get("done_reason"),
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], None],
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResult:
        payload = {
            "model": self.config.name,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "options": {
                "num_ctx": self.config.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.config.temperature,
                "seed": self.config.seed,
            },
        }
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        parts: list[str] = []
        final: dict[str, Any] | None = None
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    event = json.loads(raw_line.decode("utf-8"))
                    if "error" in event:
                        raise RuntimeError(f"Ollama stream failed: {event['error']}")
                    message = event.get("message", {})
                    thinking = message.get("thinking", "")
                    if thinking and on_thinking is not None:
                        on_thinking(thinking)
                    content = message.get("content", "")
                    if content:
                        parts.append(content)
                        on_chunk(content)
                    if event.get("done"):
                        final = event
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.config.host}. "
                "Check that `ollama serve` is running."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Ollama returned an invalid streaming response.") from exc

        if final is None:
            raise RuntimeError("Ollama stream ended without a final response.")
        return ModelResult(
            text="".join(parts),
            usage=self._usage(final),
            done_reason=final.get("done_reason"),
        )

    def _usage(self, body: dict[str, Any]) -> Usage:
        return Usage(
            prompt_eval_count=body.get("prompt_eval_count"),
            eval_count=body.get("eval_count"),
            prompt_eval_duration=body.get("prompt_eval_duration"),
            eval_duration=body.get("eval_duration"),
            total_duration=body.get("total_duration"),
            load_duration=body.get("load_duration"),
            num_ctx=self.config.num_ctx,
        )
