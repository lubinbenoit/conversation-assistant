import pytest
from assistant.context import FullHistoryContext, RecentWindowContext
from assistant.conversation import Conversation


@pytest.mark.skip(reason="Enable after implementing the full-history policy")
def test_full_history():
    c = Conversation("system"); c.add_user("a"); c.add_assistant("b"); c.add_user("c")
    assert [m.id for m in FullHistoryContext().build(c).messages] == [0, 1, 2, 3]


@pytest.mark.skip(reason="Enable after implementing the recent-turn policy")
def test_recent_window():
    c = Conversation("system")
    for i in range(5): c.add_user(f"q{i}"); c.add_assistant(f"a{i}")
    c.add_user("final")
    assert [m.id for m in RecentWindowContext(recent_turns=2).build(c).messages] == [0, 7, 8, 9, 10, 11]
