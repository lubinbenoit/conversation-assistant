from assistant.inspector import format_inspector
from assistant.inspection import InspectionRecord
from assistant.maintenance import MaintenanceCall, MaintenanceSummary


def record(**overrides):
    values = {
        "turn": 2,
        "policy": "custom",
        "num_ctx": 4096,
        "prompt_eval_count": 1024,
        "eval_count": 8,
        "done_reason": "stop",
        "history_messages": [
            {"id": 0, "role": "system", "content": "ORIGINAL SYSTEM CONTENT"},
            {"id": 1, "role": "user", "content": "A" * 900},
            {"id": 2, "role": "assistant", "content": "old reply"},
        ],
        "messages": [
            {"id": 0, "role": "system", "content": "ORIGINAL SYSTEM CONTENT"},
            {"id": 1, "role": "user", "content": "summarized request"},
            {"id": None, "role": "system", "content": "custom instruction"},
        ],
        "manager_state_before": {
            "nested": {"items": ["old", "removed"]},
            "obsolete": True,
        },
        "manager_state_after": {
            "nested": {"items": ["new"]},
            "added": {"value": 3},
        },
    }
    values.update(overrides)
    return InspectionRecord(**values)


def test_inspector_uses_bounded_semantic_message_view():
    rendered = format_inspector(record())

    assert "**Main call:** 1024 prompt tokens · 8 generated tokens" in rendered
    assert "**Prompt use:** 25.0%" in rendered
    assert "**Prompt + generation use:** 25.2%" in rendered
    assert "**Completion:** completed" in rendered
    assert "A" * 200 in rendered
    assert "A" * 201 not in rendered
    assert "_Preview truncated._" in rendered
    assert "selected" in rendered
    assert "dropped" in rendered
    assert "modified \\#1" in rendered
    assert "system · new" in rendered
    assert "1. system · same as \\#0" in rendered
    model_input = rendered.split("### Model input", 1)[1].split("---", 1)[0]
    assert "####" not in model_input
    assert "✅" not in model_input
    assert "✨" not in model_input
    assert rendered.count("ORIGINAL SYSTEM CONTENT") == 1
    assert "summarized request" in rendered
    assert "custom instruction" in rendered
    assert "```json" not in rendered
    assert "```diff" not in rendered


def test_inspector_renders_arbitrary_state_without_a_change_section():
    rendered = format_inspector(record())

    assert "Policy decisions" not in rendered
    assert "Policy state used for this answer" in rendered
    assert "Policy state after maintenance" in rendered
    assert "Changes for the next turn" not in rendered
    assert "added" in rendered
    assert "obsolete" in rendered


def test_inspector_omits_empty_policy_state():
    rendered = format_inspector(record(
        manager_state_before={},
        manager_state_after={},
    ))

    assert "Policy state" not in rendered
    assert "empty mapping" not in rendered


def test_inspector_shows_unchanged_policy_state_once():
    state = {"memory": {"preference": "concise"}}
    rendered = format_inspector(record(
        manager_state_before=state,
        manager_state_after=state,
    ))

    assert rendered.count("Policy state") == 1
    assert "Policy state used for this answer" in rendered
    assert "Policy state after maintenance" not in rendered
    assert rendered.count("concise") == 1


def test_inspector_shows_only_safe_system_prefix_addition():
    trace = record(
        messages=[
            {
                "id": 0,
                "role": "system",
                "content": "ORIGINAL SYSTEM CONTENT\n\nInjected memory",
            },
            {"id": 1, "role": "user", "content": "A" * 900},
        ]
    )

    rendered = format_inspector(trace)

    assert "extends \\#0" in rendered
    assert "Injected memory" in rendered
    assert ">\n>\n> Injected memory" not in rendered
    assert rendered.count("ORIGINAL SYSTEM CONTENT") == 1


def test_inspector_does_not_infer_an_id_for_new_system_input():
    trace = record(
        history_messages=[
            {"id": 0, "role": "system", "content": "base"},
            {"id": 3, "role": "system", "content": "base plus"},
        ],
        messages=[
            {"id": None, "role": "system", "content": "base plus memory"}
        ],
    )

    rendered = format_inspector(trace)

    assert "system · new" in rendered
    assert "base plus memory" in rendered
    assert "extends \\#" not in rendered


def test_inspector_uses_id_to_disambiguate_duplicate_content():
    trace = record(
        history_messages=[
            {"id": 0, "role": "system", "content": "system"},
            {"id": 1, "role": "user", "content": "same"},
            {"id": 3, "role": "user", "content": "same"},
        ],
        messages=[
            {"id": 0, "role": "system", "content": "system"},
            {"id": 3, "role": "user", "content": "same"},
        ],
    )

    rendered = format_inspector(trace)

    assert "same as \\#3" in rendered
    assert "same as \\#1" not in rendered
    assert "selected unchanged" not in rendered
    assert "selected, then transformed" not in rendered


def test_inspector_lists_auxiliary_tokens_after_main_call_metrics():
    trace = record(
        maintenance_calls=MaintenanceSummary((
            MaintenanceCall(30, 12, True, "stop"),
        )),
        manager_state_after={
            "aux_prompt_eval_count": 900,
            "aux_eval_count": 800,
        },
    )

    rendered = format_inspector(trace)

    assert "**Completion:** completed  \n**Maintenance call:**" in rendered
    assert rendered.index("**Completion:**") < rendered.index(
        "**Maintenance call:**"
    )
    assert "**Maintenance call:** 30 prompt tokens · 12 generated tokens" in rendered
    assert rendered.index("**Maintenance call:**") < rendered.index(
        "## Context for this answer"
    )


def test_inspector_does_not_infer_maintenance_usage_from_policy_state():
    rendered = format_inspector(record(manager_state_after={
        "aux_prompt_eval_count": 30,
        "aux_eval_count": 12,
    }))

    assert "**Maintenance call:**" not in rendered


def test_inspector_distinguishes_maintenance_failure_from_generation():
    rendered = format_inspector(
        record(maintenance_error="RuntimeError: maintenance unavailable")
    )

    assert "**Completion:** completed" in rendered
    assert (
        "**Maintenance:** ⚠️ failed — RuntimeError: maintenance unavailable"
        in rendered
    )


def test_inspector_omits_only_old_previews_without_hiding_messages():
    history = [
        {"id": 0, "role": "system", "content": "SYSTEM CONTENT"},
        {"id": 1, "role": "user", "content": "OLD ONLY CONTENT"},
    ]
    history.extend(
        {
            "id": message_id,
            "role": "assistant" if message_id % 2 == 0 else "user",
            "content": (
                "LATEST CONTENT" if message_id == 30 else f"content {message_id}"
            ),
        }
        for message_id in range(2, 31)
    )
    trace = record(
        history_messages=history,
        messages=[
            {"id": 0, "role": "system", "content": "SYSTEM CONTENT"},
            {"id": 1, "role": "user", "content": "OLD ONLY CONTENT"},
            {"id": 30, "role": "assistant", "content": "LATEST CONTENT"},
        ],
    )

    rendered = format_inspector(trace)

    assert (
        "Preview of older messages omitted. See trace for full content."
        in rendered
    )
    assert "#### ✅ \\#1 · user · selected" in rendered
    assert rendered.count("OLD ONLY CONTENT") == 0
    assert rendered.count("> _Preview omitted._") == 19
    assert "LATEST CONTENT" in rendered
    assert "user · same as \\#1" in rendered
