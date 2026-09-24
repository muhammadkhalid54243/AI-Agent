from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from framework.context.compaction import compact, compaction


def _long_convo(n=12):
    msgs = [SystemMessage(content="You are Nova.")]
    for i in range(n):
        msgs.append(HumanMessage(content=f"question {i}"))
        msgs.append(AIMessage(content=f"answer {i}"))
    return msgs


def test_compact_replaces_older_keeps_system_and_recent():
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="COMPACTED: goal was X")]))
    msgs = _long_convo(12)
    out = compact(msgs, fake, keep_recent=4)

    assert out[0].content == "You are Nova."                 # system preserved
    assert "COMPACTED: goal was X" in out[1].content          # summary injected
    assert [m.content for m in out[-4:]] == [m.content for m in msgs[-4:]]  # recent verbatim
    assert len(out) < len(msgs)                               # net reduction


def test_compact_noop_when_short():
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="unused")]))
    msgs = [SystemMessage(content="s"), HumanMessage(content="hi"), AIMessage(content="hello")]
    assert compact(msgs, fake, keep_recent=6) is msgs


def test_compaction_factory_builds_summarization_middleware():
    from langchain.agents.middleware import SummarizationMiddleware
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="s")]))
    mw = compaction(fake, trigger_tokens=2000, keep_messages=8)
    assert any(isinstance(m, SummarizationMiddleware) for m in mw)
