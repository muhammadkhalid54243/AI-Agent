from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from framework import Agent
from framework.tools import calculate


class OneShotModel(GenericFakeChatModel):
    """Returns a plain answer with no tool calls — exercises the compose+invoke path.

    create_agent binds tools to the model; this fake ignores them and returns its
    scripted message, so bind_tools is a no-op that returns self.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_runs_and_returns_text_with_injected_model():
    fake = OneShotModel(messages=iter([AIMessage(content="The answer is 144.")]))
    agent = Agent(fake, tools=[calculate])
    out = agent.run("What is 12 times 12?")
    assert "144" in out


def test_add_tools_appends():
    fake = OneShotModel(messages=iter([AIMessage(content="ok")]))
    agent = Agent(fake, tools=[])
    agent.add_tools([calculate])
    assert calculate in agent.tools
