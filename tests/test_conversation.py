import pytest

from assistant.conversation import Conversation


def test_ids_and_roles():
    c = Conversation("system")
    c.add_user("one"); c.add_assistant("two"); c.add_user("three")
    assert [m.id for m in c.messages] == [0, 1, 2, 3]
    assert [m.role for m in c.messages] == ["system", "user", "assistant", "user"]


def test_reset():
    c = Conversation("system"); c.add_user("x"); c.add_assistant("y"); c.reset()
    assert [m.id for m in c.messages] == [0]
    assert c.add_user("new").id == 1


def test_discard_last_restores_message_id():
    c = Conversation("system")
    pending = c.add_user("first attempt")

    c.discard_last(pending)

    assert [m.id for m in c.messages] == [0]
    assert c.add_user("retry").id == pending.id


def test_discard_last_rejects_stale_message():
    c = Conversation("system")
    stale = c.add_user("user")
    c.add_assistant("assistant")

    with pytest.raises(RuntimeError, match="conversation has advanced"):
        c.discard_last(stale)
