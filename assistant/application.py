from datetime import datetime
from typing import Callable
from uuid import uuid4

from assistant.context import ContextManager
from assistant.context.base import run_maintenance
from assistant.conversation import Conversation
from assistant.inspection import InspectionRecord
from assistant.maintenance import MaintenanceRecorder
from assistant.model import ModelClient
from assistant.trace import TraceWriter


def truncation_notice(num_predict: int, done_reason: str | None) -> str | None:
    if done_reason != "length":
        return None
    return (
        f"Output stopped at the configured limit of {num_predict} tokens. "
        "Increase the runtime output limit or [model].num_predict in config.toml."
    )


def maintenance_notice(error: str | None) -> str | None:
    if error is None:
        return None
    return (
        "The answer was saved, but conversation-state maintenance failed. "
        "See the context inspector for details."
    )


class ChatApplication:
    def __init__(self, config, model: ModelClient, context_manager: ContextManager):
        self.config = config
        self.model = model
        self.context_manager = context_manager
        self.maintenance_recorder = MaintenanceRecorder(
            context_manager.model_call
        )
        if context_manager.model_call is not None:
            context_manager.model_call = self.maintenance_recorder
        model_config = getattr(config, "model", None)
        self.active_think = getattr(model_config, "think", "off")
        self.active_num_predict = getattr(model_config, "num_predict", 384)
        self.conversation = Conversation(config.assistant.system_prompt)
        self.last_inspection: InspectionRecord | None = None
        self.trace = self._new_trace()
        self.turn = 0
        self._apply_runtime_settings()

    def _apply_runtime_settings(self) -> None:
        if hasattr(self.model, "set_think"):
            self.model.set_think(self.active_think in {"answers", "all"})
        self.maintenance_recorder.set_think(
            self.active_think in {"maintenance", "all"}
        )
        if hasattr(self.model, "set_num_predict"):
            self.model.set_num_predict(self.active_num_predict)

    def set_think(self, mode: str) -> None:
        if mode not in {"off", "answers", "maintenance", "all"}:
            raise ValueError("invalid thinking mode")
        if mode == self.active_think:
            return
        if hasattr(self.model, "set_think"):
            self.model.set_think(mode in {"answers", "all"})
        self.maintenance_recorder.set_think(
            mode in {"maintenance", "all"}
        )
        self.active_think = mode
        self.trace.append_settings(
            thinking_mode=self.active_think,
            num_predict=self.active_num_predict,
        )

    def set_num_predict(self, value: int) -> None:
        if value <= 0:
            raise ValueError("num_predict must be greater than zero")
        if value == self.active_num_predict:
            return
        if hasattr(self.model, "set_num_predict"):
            self.model.set_num_predict(value)
        self.active_num_predict = value
        self.trace.append_settings(
            thinking_mode=self.active_think,
            num_predict=self.active_num_predict,
        )

    def _trace_metadata(self) -> dict:
        model = getattr(self.config, "model", None)
        return {
            "model": {
                key: getattr(model, key, None)
                for key in (
                    "name",
                    "host",
                    "num_ctx",
                    "temperature",
                    "seed",
                    "timeout_seconds",
                )
            },
            "context": {
                "policy": self.context_manager.name,
                "recent_turns": self.context_manager.recent_turns,
            },
            "initial_settings": {
                "thinking_mode": self.active_think,
                "num_predict": self.active_num_predict,
            },
        }

    def _new_trace(self) -> TraceWriter:
        timestamp = datetime.now().strftime("chat-%Y%m%d-%H%M%S-%f")
        return TraceWriter(
            self.config.trace.directory,
            f"{timestamp}-{uuid4().hex[:8]}",
            self.conversation.messages[0],
            self._trace_metadata(),
        )

    def handle_message(self, text: str) -> str:
        return self._handle_message(text, None)

    def handle_message_stream(
        self,
        text: str,
        on_chunk: Callable[[str], None],
        on_thinking: Callable[[str], None] | None = None,
        on_maintenance: Callable[[bool], None] | None = None,
    ) -> str:
        return self._handle_message(
            text, on_chunk, on_thinking, on_maintenance
        )

    def _handle_message(
        self,
        text: str,
        on_chunk: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None = None,
        on_maintenance: Callable[[bool], None] | None = None,
    ) -> str:
        pending_user = self.conversation.add_user(text)
        inspection_history = self.conversation.messages
        try:
            context = self.context_manager.build(self.conversation)
            manager_state_before = self.context_manager.state()
            model_messages = [
                message.as_model_message() for message in context.messages
            ]
            if on_chunk is None:
                result = self.model.chat(model_messages)
            elif on_thinking is None:
                result = self.model.chat_stream(model_messages, on_chunk)
            else:
                result = self.model.chat_stream(
                    model_messages, on_chunk, on_thinking=on_thinking
                )
        except Exception:
            self.conversation.discard_last(pending_user)
            raise

        self.turn += 1
        assistant = self.conversation.add_assistant(result.text)
        self.trace.append_turn(pending_user, assistant)
        if on_maintenance is not None:
            on_maintenance(True)
        self.maintenance_recorder.reset()
        try:
            maintenance = run_maintenance(
                self.context_manager, self.conversation
            )
        finally:
            if on_maintenance is not None:
                on_maintenance(False)
        self.last_inspection = InspectionRecord(
            turn=self.turn,
            policy=self.context_manager.name,
            num_ctx=result.usage.num_ctx,
            prompt_eval_count=result.usage.prompt_eval_count,
            eval_count=result.usage.eval_count,
            done_reason=result.done_reason,
            history_messages=[
                {
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                }
                for message in inspection_history
            ],
            messages=[
                {
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                }
                for message in context.messages
            ],
            num_predict=self.active_num_predict,
            manager_state_before=manager_state_before,
            manager_state_after=maintenance.state,
            maintenance_error=maintenance.error,
            maintenance_calls=self.maintenance_recorder.summary(),
        )
        return result.text

    def reset(self):
        self.conversation.reset()
        self.context_manager.reset()
        self.last_inspection = None
        self.trace = self._new_trace()
        self.turn = 0
