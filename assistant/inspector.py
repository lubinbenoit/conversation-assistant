from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from assistant.inspection import InspectionRecord


_MESSAGE_PREVIEW_LIMIT = 200
_TRANSCRIPT_DETAIL_LIMIT = 12


@dataclass(frozen=True)
class InspectorContent:
    summary: str
    transcript: str | None
    details: str


def _escape(value: object) -> str:
    """Render arbitrary values without letting their text become Markdown."""
    text = str(value)
    for character in "\\`*_{}[]<>#":
        text = text.replace(character, f"\\{character}")
    return text


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _escape(value).replace("\n", " ↵ ")
    if isinstance(value, Mapping) and not value:
        return "empty mapping"
    if _is_sequence(value) and not value:
        return "empty list"
    return _escape(value)


def _tree(value: Any, depth: int = 0) -> list[str]:
    indent = "    " * depth
    if isinstance(value, Mapping):
        if not value:
            return [f"{indent}- _empty mapping_"]
        lines = []
        for key in sorted(value, key=str):
            child = value[key]
            label = _escape(key)
            nonempty_container = (
                isinstance(child, Mapping) and bool(child)
            ) or (_is_sequence(child) and bool(child))
            if nonempty_container:
                lines.append(f"{indent}- **{label}**")
                lines.extend(_tree(child, depth + 1))
            else:
                lines.append(f"{indent}- **{label}:** {_scalar(child)}")
        return lines
    if _is_sequence(value):
        if not value:
            return [f"{indent}- _empty list_"]
        lines = []
        for index, child in enumerate(value):
            nonempty_container = (
                isinstance(child, Mapping) and bool(child)
            ) or (_is_sequence(child) and bool(child))
            if nonempty_container:
                lines.append(f"{indent}- **\\[{index}\\]**")
                lines.extend(_tree(child, depth + 1))
            else:
                lines.append(f"{indent}- **\\[{index}\\]:** {_scalar(child)}")
        return lines
    return [f"{indent}- {_scalar(value)}"]


def _previewed_transcript_ids(
    history: Sequence[Mapping[str, Any]],
) -> set[int]:
    if len(history) <= _TRANSCRIPT_DETAIL_LIMIT:
        return {message["id"] for message in history}
    newest_count = _TRANSCRIPT_DETAIL_LIMIT - 1
    visible = {message["id"] for message in history[-newest_count:]}
    visible.add(history[0]["id"])
    return visible


def _content(content: Any) -> list[str]:
    text = str(content)
    truncated = len(text) > _MESSAGE_PREVIEW_LIMIT
    if truncated:
        text = text[:_MESSAGE_PREVIEW_LIMIT].rstrip()
    lines = text.splitlines() or [""]
    rendered = [f"> {_escape(line)}" if line else ">" for line in lines]
    if truncated:
        rendered.extend([">", "> …", ">", "> _Preview truncated._"])
    return rendered


def _message(
    label: str,
    role: Any,
    status: str,
    content: Any,
    *,
    show_preview: bool = True,
) -> list[str]:
    return [
        f"#### {label} · {_escape(role)} · {status}",
        "",
        *(_content(content) if show_preview else ["> _Preview omitted._"]),
        "",
    ]


def _input_message(
    position: int,
    role: Any,
    status: str,
    content: Any | None = None,
) -> list[str]:
    lines = [f"{position}. {_escape(role)} · {status}", ""]
    if content is not None:
        lines.extend(f"   {line}" for line in _content(content))
        lines.append("")
    return lines


def format_inspector_content(record: InspectionRecord | None) -> InspectorContent:
    if record is None:
        return InspectorContent(
            "# Context inspector\n\nSend a message to inspect its context.",
            None,
            "",
        )

    utilization = (
        "unknown"
        if record.prompt_eval_count is None or record.num_ctx <= 0
        else f"{100 * record.prompt_eval_count / record.num_ctx:.1f}%"
    )
    turn_utilization = (
        "unknown"
        if record.prompt_eval_count is None
        or record.eval_count is None
        or record.num_ctx <= 0
        else f"{100 * (record.prompt_eval_count + record.eval_count) / record.num_ctx:.1f}%"
    )
    completion = (
        "⚠️ output limit reached"
        if record.done_reason == "length"
        else (
            "completed"
            if record.done_reason == "stop"
            else (record.done_reason or "unknown")
        )
    )
    selected_ids = {
        message.get("id")
        for message in record.messages
        if message.get("id") is not None
    }
    transcript = []
    previewed_transcript_ids = _previewed_transcript_ids(record.history_messages)
    omitted_ids = [
        message["id"]
        for message in record.history_messages
        if message["id"] not in previewed_transcript_ids
    ]
    if omitted_ids:
        transcript.extend(
            [
                "_Preview of older messages omitted. See trace for full content._",
                "",
            ]
        )
    for message in record.history_messages:
        message_id = message["id"]
        if message_id not in selected_ids:
            icon, status = "❌", "dropped"
        else:
            icon, status = "✅", "selected"
        transcript.extend(
            _message(
                f"{icon} \\#{message_id}",
                message.get("role", "unknown"),
                status,
                message.get("content", ""),
                show_preview=message_id in previewed_transcript_ids,
            )
        )

    stored_by_id = {
        message["id"]: message for message in record.history_messages
    }
    model_input = []
    for position, message in enumerate(record.messages, start=1):
        message_id = message.get("id")
        stored = stored_by_id.get(message_id)
        unchanged = (
            stored is not None
            and stored.get("role") == message.get("role")
            and stored.get("content") == message.get("content")
        )
        if unchanged:
            model_input.extend(
                _input_message(
                    position,
                    message.get("role", "unknown"),
                    f"same as \\#{message_id}",
                )
            )
        else:
            stored_content = None if stored is None else stored.get("content")
            content = message.get("content", "")
            extends_system = (
                message.get("role") == "system"
                and stored is not None
                and stored.get("role") == "system"
                and isinstance(stored_content, str)
                and isinstance(content, str)
                and content.startswith(stored_content)
                and content != stored_content
            )
            if extends_system:
                source = f"extends \\#{message_id}"
                content = content[len(stored_content):].lstrip("\r\n")
            elif stored is not None:
                source = f"modified \\#{message_id}"
            else:
                source = "new"
            model_input.extend(
                _input_message(
                    position,
                    message.get("role", "unknown"),
                    source,
                    content,
                )
            )

    metrics = [
        f"**Policy:** {_escape(record.policy)}  ",
        f"**Main call:** {record.prompt_eval_count or 0} prompt tokens · "
        f"{record.eval_count or 0} generated tokens  ",
        f"**Prompt use:** {utilization}  ",
        f"**Prompt + generation use:** {turn_utilization}  ",
        f"**Completion:** {completion}  ",
    ]
    if record.num_predict is not None:
        metrics.insert(1, f"**Output limit:** {record.num_predict} tokens  ")
    maintenance_calls = record.maintenance_calls
    if maintenance_calls.call_count:
        label = (
            "Maintenance call"
            if maintenance_calls.call_count == 1
            else "Maintenance calls"
        )
        prompt_tokens = maintenance_calls.prompt_eval_count
        output_tokens = maintenance_calls.eval_count
        details = (
            f"{prompt_tokens if prompt_tokens is not None else 'unknown'} "
            "prompt tokens · "
            f"{output_tokens if output_tokens is not None else 'unknown'} "
            "generated tokens"
        )
        if maintenance_calls.call_count > 1:
            details = f"{maintenance_calls.call_count} calls · {details}"
        if maintenance_calls.failed_call_count:
            details += (
                f" · {maintenance_calls.failed_call_count} failed"
            )
        metrics.append(
            f"**{label}:** {details}  ",
        )
    if record.maintenance_error is not None:
        metrics.append(
            f"**Maintenance:** ⚠️ failed — "
            f"{_escape(record.maintenance_error)}  "
        )

    summary = "\n".join(
        [
            f"# Turn {record.turn}",
            "",
            *metrics,
            "",
            "---",
            "",
            "## Context for this answer",
        ]
    )
    details_lines = ["### Model input", "", *model_input]
    state_before = record.manager_state_before
    state_after = record.manager_state_after
    if state_before or state_after:
        details_lines.extend([
            "---",
            "",
            "## Policy state used for this answer",
            "",
            *_tree(state_before),
        ])
        if state_after != state_before:
            details_lines.extend([
                "",
                "---",
                "",
                "## Policy state after maintenance",
                "",
                *_tree(state_after),
            ])
    details = "\n".join(details_lines)
    return InspectorContent(summary, "\n".join(transcript).rstrip(), details)


def format_inspector(record: InspectionRecord | None) -> str:
    content = format_inspector_content(record)
    if content.transcript is None:
        return content.summary
    return "\n\n".join(
        [
            content.summary,
            f"### Stored transcript\n\n{content.transcript}",
            content.details,
        ]
    )
