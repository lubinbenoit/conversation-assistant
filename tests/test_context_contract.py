import pytest
from assistant.context import StatelessContext
from assistant.conversation import Conversation, Message


def test_stateless_selects_system_and_current_user():
    c = Conversation("system"); c.add_user("first"); c.add_assistant("reply"); c.add_user("second")
    ctx = StatelessContext().build(c)
    assert [message.id for message in ctx.messages] == [0, 3]
    assert [message.role for message in ctx.messages] == ["system", "user"]


def test_build_after_assistant_is_invalid():
    c = Conversation("system"); c.add_user("hello"); c.add_assistant("hi")
    with pytest.raises(ValueError): StatelessContext().build(c)


def test_model_message_omits_transcript_id():
    message = Message(id=None, role="system", content="model-only context")

    assert message.as_model_message() == {
        "role": "system",
        "content": "model-only context",
    }
