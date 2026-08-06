from langgraph.checkpoint.memory import InMemorySaver
from framework.memory import checkpoint


def test_checkpoint_memory_returns_in_memory_saver():
    cp = checkpoint("memory")
    assert isinstance(cp, InMemorySaver)


def test_checkpoint_unknown_kind_raises():
    import pytest
    with pytest.raises(NotImplementedError):
        checkpoint("sqlite", "state.db")


import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_memory_persists_across_turns():
    from framework import Agent
    from framework.memory import checkpoint

    agent = Agent("groq:llama-3.3-70b-versatile", memory=checkpoint("memory"),
                  system_prompt="Answer briefly.")
    agent.run("My favorite number is 42. Remember it.", thread_id="s1")
    out = agent.run("What is my favorite number?", thread_id="s1")
    assert "42" in out
