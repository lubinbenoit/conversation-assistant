import asyncio
from threading import Event
from types import SimpleNamespace

from rich.markdown import Markdown as RichMarkdown
from textual.containers import VerticalScroll
from textual.widgets import Button, Collapsible, Input, Markdown, Select, Static, TextArea

from assistant.application import ChatApplication
from assistant.context import StatelessContext
from assistant.model import ModelResult, Usage
from assistant.tui import ChatTUI, ThinkingContent, _ChunkBatcher


def model_result(text="streamed response", done_reason="stop"):
    return ModelResult(
        text=text,
        usage=Usage(10, 2, None, None, None, None, 4096),
        done_reason=done_reason,
    )


class StreamingModel:
    def __init__(self, done_reason="stop"):
        self.done_reason = done_reason

    def chat_stream(self, messages, on_chunk, on_thinking=None):
        on_chunk("streamed ")
        on_chunk("response")
        return model_result(done_reason=self.done_reason)


class MathStreamingModel:
    def chat_stream(self, messages, on_chunk, on_thinking=None):
        on_chunk("Result: $x^")
        on_chunk("2 + \\alpha$")
        return model_result(r"Result: $x^2 + \alpha$")


class ThinkingStreamingModel:
    def chat_stream(self, messages, on_chunk, on_thinking=None):
        on_thinking("Considering the question")
        on_chunk("Final answer")
        return model_result("Final answer")


class MarkdownThinkingStreamingModel:
    def chat_stream(self, messages, on_chunk, on_thinking=None):
        for chunk in ("## Plan\n\n", "Use **care** ", r"with $\alpha$."):
            on_thinking(chunk)
        on_chunk("Final answer")
        return model_result("Final answer")


class LongThinkingStreamingModel:
    thinking = "\n\n".join(f"Paragraph {index}" for index in range(150))

    def chat_stream(self, messages, on_chunk, on_thinking=None):
        on_thinking(self.thinking)
        on_chunk("Final answer")
        return model_result("Final answer")


class BurstStreamingModel:
    def __init__(self, chunks=1000):
        self.chunks = chunks

    def chat_stream(self, messages, on_chunk, on_thinking=None):
        for _ in range(self.chunks):
            on_chunk("x")
        return model_result("x" * self.chunks)


def config(tmp_path):
    return SimpleNamespace(
        model=SimpleNamespace(name="test", num_ctx=4096, num_predict=384),
        assistant=SimpleNamespace(system_prompt="system"),
        trace=SimpleNamespace(directory=tmp_path),
    )


def test_footer_prioritizes_runtime_controls_over_input_history():
    visibility = {binding.description: binding.show for binding in ChatTUI.BINDINGS}

    assert visibility["Thinking"]
    assert visibility["Output limit"]
    assert not visibility["Previous input"]
    assert not visibility["Next input"]


def test_worker_owned_chunk_batcher_flushes_by_size_time_and_force():
    now = [0.0]
    writes = []
    batcher = _ChunkBatcher(
        writes.append,
        max_delay=1.0,
        max_chars=5,
        clock=lambda: now[0],
    )

    batcher.add("a")
    batcher.add("bc")
    batcher.add("def")
    assert writes == ["a", "bcdef"]

    batcher.add("g")
    now[0] = 1.1
    batcher.add("h")
    assert writes == ["a", "bcdef", "gh"]

    batcher.add("i")
    batcher.flush()
    assert writes == ["a", "bcdef", "gh", "i"]


def test_output_limit_shortcut_wins_over_textual_command_palette(tmp_path):
    async def exercise():
        chat = ChatApplication(config(tmp_path), StreamingModel(), StatelessContext())
        app = ChatTUI(chat)
        async with app.run_test():
            binding = app.screen.active_bindings["ctrl+p"][1]
            assert binding.action == "set_num_predict"
            assert binding.description == "Output limit"

    asyncio.run(exercise())


def test_tui_sends_streamed_message_and_keeps_input_history(tmp_path):
    async def exercise():
        chat = ChatApplication(config(tmp_path), StreamingModel(), StatelessContext())
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            editor = app.query_one("#input", TextArea)
            editor.text = "hello\nfrom two lines"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert [message.content for message in chat.conversation.messages] == [
                "system",
                "hello\nfrom two lines",
                "streamed response",
            ]
            assert app.input_history == ["hello\nfrom two lines"]
            assert app.query_one(".assistant-message", Markdown)._markdown == (
                "streamed response"
            )
            summary = app.query_one("#inspector-summary", Markdown)._markdown
            details = app.query_one("#inspector-details", Markdown)._markdown
            assert "# Turn 1" in summary
            assert "## Context for this answer" in summary
            assert "### Model input" in details
            assert "10 prompt tokens · 2 generated tokens" in summary
            assert "Prompt use" in summary
            assert "Prompt + generation use" in summary

            app.action_previous_input()
            assert editor.text == "hello\nfrom two lines"
            app.action_next_input()
            assert editor.text == ""

    asyncio.run(exercise())


def test_tui_converts_latex_but_stores_the_original_response(tmp_path):
    async def exercise():
        chat = ChatApplication(config(tmp_path), MathStreamingModel(), StatelessContext())
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            app.query_one("#input", TextArea).text = "formula"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert chat.conversation.messages[-1].content == r"Result: $x^2 + \alpha$"
            rendered = app.query_one(".assistant-message", Markdown)._markdown
            assert "𝑥² + α" in rendered

    asyncio.run(exercise())


def test_tui_writes_chunks_through_textual_markdown_stream(tmp_path, monkeypatch):
    writes = []

    class SpyStream:
        def __init__(self, name):
            self.name = name

        def write(self, chunk):
            writes.append((self.name, chunk))

        def stop(self):
            writes.append((self.name, "STOP"))

    monkeypatch.setattr(
        Markdown,
        "get_stream",
        lambda markdown: SpyStream(
            "thinking" if markdown.has_class("thinking-content") else "answer"
        ),
    )

    async def exercise():
        chat = ChatApplication(config(tmp_path), StreamingModel(), StatelessContext())
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            app.query_one("#input", TextArea).text = "hello"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert writes == [
                ("answer", "streamed "),
                ("answer", "response"),
                ("answer", "STOP"),
            ]
            assert app.query_one(".assistant-message", Markdown)._markdown == (
                "streamed response"
            )

    asyncio.run(exercise())


def test_tui_batches_bursty_answer_chunks_before_ui_handoff(
    tmp_path, monkeypatch
):
    writes = {"answer": [], "thinking": []}

    class SpyStream:
        def __init__(self, name):
            self.name = name

        def write(self, chunk):
            writes[self.name].append(chunk)

        def stop(self):
            pass

    monkeypatch.setattr(
        Markdown,
        "get_stream",
        lambda markdown: SpyStream(
            "thinking" if markdown.has_class("thinking-content") else "answer"
        ),
    )

    async def exercise():
        chat = ChatApplication(
            config(tmp_path), BurstStreamingModel(), StatelessContext()
        )
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            app.query_one("#input", TextArea).text = "hello"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert "".join(writes["answer"]) == "x" * 1000
            assert len(writes["answer"]) <= 5

    asyncio.run(exercise())


def test_tui_keeps_thinking_in_a_collapsed_box_without_storing_it(tmp_path):
    async def exercise():
        chat = ChatApplication(
            config(tmp_path), ThinkingStreamingModel(), StatelessContext()
        )
        app = ChatTUI(chat)
        shown = []
        original = app._show_thinking

        def capture(box, widget, content):
            shown.append(content)
            original(box, widget, content)

        app._show_thinking = capture
        async with app.run_test() as pilot:
            app.query_one("#input", TextArea).text = "question"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert shown == ["Considering the question"]
            thinking = app.query_one(".assistant-thinking", Collapsible)
            assert thinking.display
            assert thinking.collapsed
            scroll = app.query_one(".thinking-scroll", VerticalScroll)
            assert scroll.scroll_y == 0
            content = app.query_one(".thinking-content", ThinkingContent)
            assert content.markdown_source == "Considering the question"
            thinking.collapsed = False
            await pilot.pause()
            assert not thinking.collapsed
            assert chat.conversation.messages[-1].content == "Final answer"
            assert "Considering" not in chat.trace.path.read_text()

    asyncio.run(exercise())


def test_tui_batches_thinking_updates_and_renders_markdown(tmp_path):
    async def exercise():
        chat = ChatApplication(
            config(tmp_path), MarkdownThinkingStreamingModel(), StatelessContext()
        )
        app = ChatTUI(chat)
        shown = []
        original = app._show_thinking

        def capture(box, widget, content):
            shown.append(content)
            original(box, widget, content)

        app._show_thinking = capture
        async with app.run_test() as pilot:
            app.query_one("#input", TextArea).text = "question"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            content = app.query_one(".thinking-content", ThinkingContent)
            assert content.markdown_source == "## Plan\n\nUse **care** with α."
            assert isinstance(content.content, RichMarkdown)
            assert shown == ["## Plan\n\nUse **care** with α."]

    asyncio.run(exercise())


def test_tui_keeps_long_thinking_in_a_scrollable_viewport(tmp_path):
    async def exercise():
        chat = ChatApplication(
            config(tmp_path), LongThinkingStreamingModel(), StatelessContext()
        )
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            app.query_one("#input", TextArea).text = "question"
            await app.action_send_message()
            await app.workers.wait_for_complete()

            thinking = app.query_one(".assistant-thinking", Collapsible)
            scroll = app.query_one(".thinking-scroll", VerticalScroll)
            assert thinking.collapsed
            assert scroll.scroll_y == 0
            thinking.collapsed = False
            await pilot.pause()

            content = app.query_one(".thinking-content", ThinkingContent)
            assert content.markdown_source == LongThinkingStreamingModel.thinking
            assert scroll.size.height == 10
            assert scroll.max_scroll_y > 0
            assert scroll.scroll_y == 0

    asyncio.run(exercise())


def test_new_turn_scrolls_updated_inspector_to_top(tmp_path):
    async def exercise():
        chat = ChatApplication(config(tmp_path), StreamingModel(), StatelessContext())
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            editor = app.query_one("#input", TextArea)
            editor.text = "first"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            pane = app.query_one("#inspector-pane", VerticalScroll)
            pane.scroll_end(animate=False)
            await pilot.pause()
            assert pane.scroll_y > 0

            editor.text = "second"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert pane.scroll_y == 0

    asyncio.run(exercise())


def test_transcript_has_bounded_scroll_and_stays_at_bottom_on_update(tmp_path):
    async def exercise():
        chat = ChatApplication(config(tmp_path), StreamingModel(), StatelessContext())
        for index in range(15):
            chat.conversation.add_user(f"older question {index}")
            chat.conversation.add_assistant(f"older answer {index}")

        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            editor = app.query_one("#input", TextArea)
            editor.text = "newest question"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            box = app.query_one("#inspector-transcript-box", Collapsible)
            scroll = app.query_one(
                ".inspector-transcript-scroll", VerticalScroll
            )
            assert box.display
            assert box.collapsed

            box.collapsed = False
            await pilot.pause()
            assert scroll.region.height <= 14
            assert scroll.max_scroll_y > 0
            assert scroll.scroll_y == scroll.max_scroll_y

            box.collapsed = True
            editor.text = "another question"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert box.collapsed

            box.collapsed = False
            await pilot.pause()
            assert scroll.scroll_y == scroll.max_scroll_y

    asyncio.run(exercise())


def test_runtime_controls_update_thinking_and_output_limit(tmp_path):
    async def exercise():
        chat = ChatApplication(config(tmp_path), StreamingModel(), StatelessContext())
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            app.action_set_thinking_mode()
            await pilot.pause()
            dialog_select = app.screen.query_one("#thinking-mode", Select)
            dialog_select.value = "maintenance"
            await pilot.click("#apply-thinking-mode")
            await pilot.pause()

            assert chat.active_think == "maintenance"
            assert chat.maintenance_recorder.thinking_enabled is True
            assert "Thinking: maintenance" in str(
                app.query_one("#session", Static).content
            )

            app.action_set_num_predict()
            await pilot.pause()
            dialog_input = app.screen.query_one("#num-predict", Input)
            dialog_input.value = "700"
            await pilot.click("#apply-num-predict")
            await pilot.pause()

            assert chat.active_num_predict == 700
            assert "Output limit: 700" in str(
                app.query_one("#session", Static).content
            )

    asyncio.run(exercise())


def test_tui_shows_a_dedicated_maintenance_activity_row(tmp_path):
    async def exercise():
        chat = ChatApplication(config(tmp_path), StreamingModel(), StatelessContext())
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            app._set_maintenance_busy(True)
            await pilot.pause()
            status = app.query_one("#maintenance-status", Static)
            assert status.display
            assert str(status.content) == "Maintenance in progress…"
            assert status.region.y == app.query_one("#send", Button).region.y

            app._set_maintenance_busy(False)
            assert not status.display

    asyncio.run(exercise())


def test_complete_answer_is_visible_while_policy_update_blocks(tmp_path):
    class BlockingContext(StatelessContext):
        def __init__(self):
            super().__init__()
            self.entered = Event()
            self.release = Event()

        def update(self, conversation):
            self.entered.set()
            self.release.wait(timeout=2)

    async def exercise():
        manager = BlockingContext()
        chat = ChatApplication(config(tmp_path), StreamingModel(), manager)
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            app.query_one("#input", TextArea).text = "question"
            await app.action_send_message()

            for _ in range(50):
                if manager.entered.is_set():
                    break
                await asyncio.sleep(0.01)
            await pilot.pause()

            status = app.query_one("#maintenance-status", Static)
            response = app.query_one(".assistant-message", Markdown)
            assert manager.entered.is_set()
            assert status.display
            assert response._markdown == "streamed response"

            manager.release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not status.display

    asyncio.run(exercise())


def test_tui_keeps_answer_when_maintenance_fails(tmp_path):
    class FailingMaintenanceContext(StatelessContext):
        def update(self, conversation):
            raise RuntimeError("maintenance unavailable")

    async def exercise():
        chat = ChatApplication(
            config(tmp_path), StreamingModel(), FailingMaintenanceContext()
        )
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            app.query_one("#input", TextArea).text = "question"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            response = app.query_one(".assistant-message", Markdown)._markdown
            assert response.startswith("streamed response")
            assert "Maintenance failed" in response
            assert not response.startswith("**Error:**")
            assert not app.query_one("#maintenance-status", Static).display
            summary = app.query_one("#inspector-summary", Markdown)._markdown
            assert "**Completion:** completed" in summary
            assert "**Maintenance:** ⚠️ failed" in summary

    asyncio.run(exercise())


def test_enter_sends_and_shift_enter_inserts_a_line_break(tmp_path):
    async def exercise():
        chat = ChatApplication(config(tmp_path), StreamingModel(), StatelessContext())
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            editor = app.query_one("#input", TextArea)
            editor.text = "first line"
            editor.move_cursor((0, len(editor.text)))

            await pilot.press("shift+enter")
            await pilot.press("s", "e", "c", "o", "n", "d")
            assert editor.text == "first line\nsecond"
            assert chat.conversation.messages[-1].content == "system"

            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert editor.text == ""
            assert chat.conversation.messages[1].content == "first line\nsecond"

    asyncio.run(exercise())


def test_tui_marks_length_limited_response_as_truncated(tmp_path):
    async def exercise():
        settings = config(tmp_path)
        settings.model.num_predict = 32
        chat = ChatApplication(
            settings, StreamingModel(done_reason="length"), StatelessContext()
        )
        app = ChatTUI(chat)
        async with app.run_test() as pilot:
            editor = app.query_one("#input", TextArea)
            editor.text = "write a long answer"
            await app.action_send_message()
            await app.workers.wait_for_complete()
            await pilot.pause()

            response = app.query_one(".assistant-message", Markdown)._markdown
            assert "**Output truncated.**" in response
            assert "configured limit of 32 tokens" in response
            assert "output limit reached" in app.query_one(
                "#inspector-summary", Markdown
            )._markdown

    asyncio.run(exercise())
