from __future__ import annotations

from time import monotonic
from typing import Callable

from rich.markdown import Markdown as RichMarkdown
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Select,
    Static,
    TextArea,
)

from assistant.application import ChatApplication, maintenance_notice, truncation_notice
from assistant.inspection import InspectionRecord
from assistant.inspector import format_inspector_content
from assistant.rendering import render_assistant_markdown


_STREAM_BATCH_MAX_DELAY = 0.05
_STREAM_BATCH_MAX_CHARS = 256
_THINKING_BATCH_MAX_DELAY = 0.2
_THINKING_BATCH_MAX_CHARS = 1024


class _ChunkBatcher:
    """Batch chunks within one worker; never shared with the UI thread."""

    def __init__(
        self,
        sink: Callable[[str], None],
        *,
        max_delay: float = _STREAM_BATCH_MAX_DELAY,
        max_chars: int = _STREAM_BATCH_MAX_CHARS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._sink = sink
        self._max_delay = max_delay
        self._max_chars = max_chars
        self._clock = clock
        self._chunks: list[str] = []
        self._char_count = 0
        self._has_flushed = False
        self._last_flush = clock()

    def add(self, chunk: str) -> None:
        if not chunk:
            return
        if not self._has_flushed:
            self._sink(chunk)
            self._has_flushed = True
            self._last_flush = self._clock()
            return

        self._chunks.append(chunk)
        self._char_count += len(chunk)
        if (
            self._char_count >= self._max_chars
            or self._clock() - self._last_flush >= self._max_delay
        ):
            self.flush()

    def flush(self) -> None:
        if not self._chunks:
            return
        batch = "".join(self._chunks)
        self._chunks.clear()
        self._char_count = 0
        self._sink(batch)
        self._has_flushed = True
        self._last_flush = self._clock()


class ThinkingContent(Static):
    """Single-widget Markdown view for complete, scrollable thinking text."""

    def __init__(self, *, classes: str):
        super().__init__("", classes=classes)
        self.markdown_source = ""

    def append_markdown(self, chunk: str) -> None:
        self.set_markdown(self.markdown_source + chunk)

    def set_markdown(self, content: str) -> None:
        self.markdown_source = content
        self.update(RichMarkdown(content))


class ContextInspector(Vertical):
    """Structured latest-turn view with an independently scrolling transcript."""

    def compose(self) -> ComposeResult:
        initial = format_inspector_content(None)
        yield Markdown(initial.summary, id="inspector-summary")
        transcript = Markdown("", id="inspector-transcript")
        transcript_scroll = VerticalScroll(
            transcript, classes="inspector-transcript-scroll"
        )
        box = Collapsible(
            transcript_scroll,
            title="Stored transcript",
            collapsed=True,
            id="inspector-transcript-box",
        )
        box.display = False
        yield box
        yield Markdown(initial.details, id="inspector-details")

    async def show_record(self, record: InspectionRecord | None) -> None:
        content = format_inspector_content(record)
        await self.query_one("#inspector-summary", Markdown).update(content.summary)
        details = self.query_one("#inspector-details", Markdown)
        box = self.query_one("#inspector-transcript-box", Collapsible)
        if content.transcript is None:
            box.display = False
            await self.query_one("#inspector-transcript", Markdown).update("")
            await details.update(content.details)
            return

        box.display = True
        await self.query_one("#inspector-transcript", Markdown).update(
            content.transcript
        )
        await details.update(content.details)
        self._scroll_transcript_to_bottom()

    async def show_error(self, message: str) -> None:
        await self.query_one("#inspector-summary", Markdown).update(
            f"# Context inspector\n\n**Generation error:** {message}"
        )
        self.query_one("#inspector-transcript-box", Collapsible).display = False
        await self.query_one("#inspector-details", Markdown).update("")

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        if event.collapsible.id == "inspector-transcript-box":
            self._scroll_transcript_to_bottom()

    def _scroll_transcript_to_bottom(self) -> None:
        scroll = self.query_one(
            ".inspector-transcript-scroll", VerticalScroll
        )
        scroll.call_after_refresh(scroll.scroll_end, animate=False)


class NumPredictDialog(ModalScreen[int | None]):
    CSS = """
    NumPredictDialog { align: center middle; }
    #num-predict-dialog {
        width: 48;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    #num-predict-actions { height: 3; align-horizontal: right; }
    #num-predict-actions Button { margin-left: 1; }
    """

    def __init__(self, current: int):
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="num-predict-dialog"):
            yield Label("Maximum generated tokens for subsequent calls")
            yield Input(value=str(self.current), type="integer", id="num-predict")
            with Horizontal(id="num-predict-actions"):
                yield Button("Cancel", id="cancel-num-predict")
                yield Button("Apply", id="apply-num-predict", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#num-predict", Input).focus()

    def _submit(self) -> None:
        raw = self.query_one("#num-predict", Input).value
        try:
            value = int(raw)
            if value <= 0:
                raise ValueError
        except ValueError:
            self.notify("Enter a positive integer.", severity="error")
            return
        self.dismiss(value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-num-predict":
            self._submit()
        elif event.button.id == "cancel-num-predict":
            self.dismiss(None)


class ThinkingModeDialog(ModalScreen[str | None]):
    CSS = """
    ThinkingModeDialog { align: center middle; }
    #thinking-mode-dialog {
        width: 38;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    #thinking-mode-actions { height: 3; align-horizontal: right; }
    #thinking-mode-actions Button { margin-left: 1; }
    """

    OPTIONS = [
        ("Off", "off"),
        ("Answers only", "answers"),
        ("Maintenance only", "maintenance"),
        ("All calls", "all"),
    ]

    def __init__(self, current: str):
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="thinking-mode-dialog"):
            yield Label("Thinking mode")
            yield Select(
                self.OPTIONS,
                value=self.current,
                allow_blank=False,
                id="thinking-mode",
            )
            with Horizontal(id="thinking-mode-actions"):
                yield Button("Cancel", id="cancel-thinking-mode")
                yield Button("Apply", id="apply-thinking-mode", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#thinking-mode", Select).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-thinking-mode":
            value = self.query_one("#thinking-mode", Select).value
            self.dismiss(str(value))
        elif event.button.id == "cancel-thinking-mode":
            self.dismiss(None)


class ChatInput(TextArea):
    """Composer where Enter submits and Shift+Enter inserts a line break."""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            await self.app.action_send_message()
            return
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            start, end = self.selection
            self.replace("\n", start, end, maintain_selection_offset=False)
            return
        await super()._on_key(event)


class ChatTUI(App[None]):
    TITLE = "Conversational Assistant"
    SUB_TITLE = "Ollama"
    # Textual reserves Ctrl+P for its command palette by default. This app uses
    # that shortcut for the more relevant runtime output-limit control.
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen { layout: vertical; }
    #session { height: 1; padding: 0 1; color: $text-muted; }
    #maintenance-status {
        width: 1fr;
        height: 3;
        padding: 0 1;
        content-align: left middle;
        color: $warning;
        text-style: bold;
    }
    #workspace { height: 1fr; }
    #chat { width: 3fr; padding: 1 2; }
    #inspector-pane {
        width: 2fr;
        min-width: 40;
        overflow-x: hidden;
        border-left: solid $accent;
    }
    #inspector { width: 100%; height: auto; }
    #inspector-summary, #inspector-details {
        width: 100%;
        padding: 1 2;
        text-wrap: wrap;
    }
    #inspector-transcript-box { margin: 0 1; }
    .inspector-transcript-scroll {
        height: auto;
        max-height: 14;
        overflow-x: hidden;
        overflow-y: auto;
    }
    #inspector-transcript { width: 100%; padding: 0 1; text-wrap: wrap; }
    .user-label { color: $accent; text-style: bold; margin-top: 1; }
    .user-message { margin-bottom: 1; }
    .assistant-label { color: $success; text-style: bold; }
    .assistant-thinking { margin-bottom: 1; }
    .thinking-scroll {
        height: 10;
        max-height: 10;
        overflow-x: hidden;
        overflow-y: scroll;
    }
    .thinking-content {
        color: $text-muted;
        padding: 0 1;
        width: 100%;
    }
    .assistant-message { margin-bottom: 1; }
    #composer { height: 9; padding: 0 1; }
    #input { height: 6; border: round $accent; }
    #actions { height: 3; align-horizontal: right; }
    #actions Button { margin-left: 1; min-width: 10; }
    """

    BINDINGS = [
        Binding("ctrl+up", "previous_input", "Previous input", show=False),
        Binding("ctrl+down", "next_input", "Next input", show=False),
        Binding("ctrl+r", "reset_chat", "Reset"),
        Binding("ctrl+i", "toggle_inspector", "Inspector"),
        Binding("ctrl+t", "set_thinking_mode", "Thinking"),
        Binding("ctrl+p", "set_num_predict", "Output limit"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, chat: ChatApplication):
        super().__init__()
        self.chat = chat
        self.input_history: list[str] = []
        self.history_index = 0
        self.busy = False
        self.maintenance_busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(self._session_text(), id="session")
        with Horizontal(id="workspace"):
            yield VerticalScroll(id="chat")
            with VerticalScroll(id="inspector-pane"):
                yield ContextInspector(id="inspector")
        with Vertical(id="composer"):
            yield ChatInput(id="input", language=None, show_line_numbers=False)
            with Horizontal(id="actions"):
                status = Static(
                    "Maintenance in progress…", id="maintenance-status"
                )
                status.display = False
                yield status
                yield Button("Reset", id="reset")
                yield Button("Send", id="send", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input", TextArea).focus()

    def _session_text(self) -> str:
        return (
            f"Model: {self.chat.config.model.name}  •  "
            f"Policy: {self.chat.context_manager.name}  •  "
            f"Context: {self.chat.config.model.num_ctx}  •  "
            f"Thinking: {self.chat.active_think}  •  "
            f"Output limit: {self.chat.active_num_predict}"
        )

    def _refresh_session_text(self) -> None:
        self.query_one("#session", Static).update(self._session_text())

    def _set_maintenance_busy(self, active: bool) -> None:
        self.maintenance_busy = active
        self.query_one("#maintenance-status", Static).display = active

    def action_set_thinking_mode(self) -> None:
        if self.busy:
            self.notify("Wait for the current response.", severity="warning")
            return

        def apply(value: str | None) -> None:
            if value is None:
                return
            self.chat.set_think(value)
            self._refresh_session_text()
            self.notify(f"Thinking mode set to {value}.")

        self.push_screen(ThinkingModeDialog(self.chat.active_think), apply)

    def action_set_num_predict(self) -> None:
        if self.busy:
            self.notify("Wait for the current response.", severity="warning")
            return

        def apply(value: int | None) -> None:
            if value is None:
                return
            self.chat.set_num_predict(value)
            self._refresh_session_text()
            self.notify(f"Output limit set to {value} tokens.")

        self.push_screen(NumPredictDialog(self.chat.active_num_predict), apply)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            await self.action_send_message()
        elif event.button.id == "reset":
            await self.action_reset_chat()

    async def action_send_message(self) -> None:
        if self.busy:
            return
        editor = self.query_one("#input", TextArea)
        text = editor.text.strip()
        if not text:
            return
        editor.text = ""
        self.input_history.append(text)
        self.history_index = len(self.input_history)
        await self._start_turn(text)

    async def action_reset_chat(self) -> None:
        if self.busy:
            self.notify("Wait for the current response before resetting.", severity="warning")
            return
        self.chat.reset()
        await self.query_one("#chat", VerticalScroll).remove_children()
        await self._update_inspector(None)
        self.notify("Conversation reset; a new trace session has started.")

    async def _update_inspector(self, record: InspectionRecord | None) -> None:
        await self.query_one("#inspector", ContextInspector).show_record(record)
        self.query_one("#inspector-pane", VerticalScroll).scroll_home(animate=False)

    async def _show_inspector_error(self, message: str) -> None:
        await self.query_one("#inspector", ContextInspector).show_error(message)
        self.query_one("#inspector-pane", VerticalScroll).scroll_home(animate=False)

    def action_toggle_inspector(self) -> None:
        pane = self.query_one("#inspector-pane", VerticalScroll)
        pane.display = not pane.display

    def action_previous_input(self) -> None:
        if not self.input_history:
            return
        self.history_index = max(0, self.history_index - 1)
        self.query_one("#input", TextArea).text = self.input_history[self.history_index]

    def action_next_input(self) -> None:
        if not self.input_history:
            return
        self.history_index = min(len(self.input_history), self.history_index + 1)
        text = "" if self.history_index == len(self.input_history) else self.input_history[self.history_index]
        self.query_one("#input", TextArea).text = text

    async def _prepare_turn(
        self, text: str
    ) -> tuple[Markdown, Collapsible, ThinkingContent]:
        log = self.query_one("#chat", VerticalScroll)
        await log.mount(Label("You", classes="user-label"))
        await log.mount(Static(text, classes="user-message"))
        await log.mount(Label("Assistant", classes="assistant-label"))
        thinking_content = ThinkingContent(classes="thinking-content")
        thinking_scroll = VerticalScroll(
            thinking_content, classes="thinking-scroll"
        )
        thinking = Collapsible(
            thinking_scroll,
            title="Thinking",
            collapsed=False,
            classes="assistant-thinking",
        )
        thinking.display = False
        await log.mount(thinking)
        response = Markdown("_Thinking…_", classes="assistant-message")
        await log.mount(response)
        log.scroll_end(animate=False)
        return response, thinking, thinking_content

    async def _start_turn(self, text: str) -> None:
        response, thinking, thinking_content = await self._prepare_turn(text)
        await response.update("")
        stream = Markdown.get_stream(response)
        self.query_one("#chat", VerticalScroll).anchor()
        self.busy = True
        self.query_one("#send", Button).disabled = True
        self._stream_response(
            text,
            response,
            thinking,
            thinking_content,
            stream,
        )

    def _write_thinking_chunk(
        self, box: Collapsible, widget: ThinkingContent, chunk: str
    ) -> None:
        box.display = True
        box.collapsed = False
        widget.append_markdown(chunk)
        scroll = box.query_one(".thinking-scroll", VerticalScroll)
        scroll.call_after_refresh(scroll.scroll_end, animate=False)

    def _show_thinking(
        self, box: Collapsible, widget: ThinkingContent, content: str
    ) -> None:
        widget.set_markdown(content)
        box.display = True
        box.collapsed = False

    def _collapse_thinking(self, box: Collapsible) -> None:
        if box.display:
            scroll = box.query_one(".thinking-scroll", VerticalScroll)
            scroll.scroll_home(animate=False)
            box.collapsed = True
            scroll.call_after_refresh(scroll.scroll_home, animate=False)

    @work(thread=True, exclusive=True)
    def _stream_response(
        self,
        text: str,
        response: Markdown,
        thinking: Collapsible,
        thinking_content: ThinkingContent,
        stream,
    ) -> None:
        succeeded = False
        final_markdown: str | None = None
        thought_chunks: list[str] = []
        thinking_finalized = False
        answer_started = False
        answer_batches = _ChunkBatcher(
            lambda batch: self.call_from_thread(stream.write, batch)
        )
        thinking_batches = _ChunkBatcher(
            lambda batch: self.call_from_thread(
                self._write_thinking_chunk,
                thinking,
                thinking_content,
                batch,
            ),
            max_delay=_THINKING_BATCH_MAX_DELAY,
            max_chars=_THINKING_BATCH_MAX_CHARS,
        )

        def finalize_thinking() -> None:
            nonlocal thinking_finalized
            if thinking_finalized:
                return
            thinking_finalized = True
            thinking_batches.flush()
            thought = "".join(thought_chunks)
            if thought:
                self.call_from_thread(
                    self._show_thinking,
                    thinking,
                    thinking_content,
                    render_assistant_markdown(thought),
                )

        def on_chunk(chunk: str) -> None:
            nonlocal answer_started
            if not answer_started:
                answer_started = True
                finalize_thinking()
                self.call_from_thread(self._collapse_thinking, thinking)
            answer_batches.add(chunk)

        def on_thinking(chunk: str) -> None:
            if answer_started:
                return
            thought_chunks.append(chunk)
            thinking_batches.add(chunk)

        def on_maintenance(active: bool) -> None:
            if active:
                # The model stream is complete at this boundary. Publish any
                # short trailing batch before maintenance can block the worker.
                answer_batches.flush()
            self.call_from_thread(self._set_maintenance_busy, active)

        try:
            result_text = self.chat.handle_message_stream(
                text, on_chunk, on_thinking, on_maintenance
            )
            succeeded = True
            notice = truncation_notice(
                self.chat.active_num_predict,
                self.chat.last_inspection.done_reason,
            )
            final_markdown = render_assistant_markdown(result_text)
            if notice:
                final_markdown += (
                    f"\n\n> ⚠️ **Output truncated.** {notice}"
                )
            warning = maintenance_notice(
                self.chat.last_inspection.maintenance_error
            )
            if warning:
                final_markdown += f"\n\n> ⚠️ **Maintenance failed.** {warning}"
        except Exception as exc:
            final_markdown = f"**Error:** {exc}"
            self.call_from_thread(
                self._show_inspector_error,
                str(exc),
            )
        finally:
            finalize_thinking()
            self.call_from_thread(self._collapse_thinking, thinking)
            answer_batches.flush()
            self.call_from_thread(stream.stop)
            if final_markdown is not None:
                self.call_from_thread(response.update, final_markdown)
            self.call_from_thread(self._finish_turn, succeeded)

    async def _finish_turn(self, succeeded: bool = True) -> None:
        self._set_maintenance_busy(False)
        self.busy = False
        self.query_one("#send", Button).disabled = False
        if succeeded:
            await self._update_inspector(self.chat.last_inspection)
        self.query_one("#input", TextArea).focus()


def run_tui(chat: ChatApplication) -> None:
    ChatTUI(chat).run()
