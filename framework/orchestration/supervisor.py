"""Orchestration — delegate to specialist sub-agents.

A sub-agent is just an Agent wrapped as a tool: the supervisor calls it with a
question and gets back its answer. The supervisor is itself an Agent whose tools
are those sub-agents, so it plans, delegates, and synthesizes with the same loop.
Use multiple agents only when the task needs distinct expertise — a single
well-tooled agent is often better.
"""

from langchain_core.tools import StructuredTool

from framework.core.agent import Agent

_DEFAULT_SUPERVISOR_PROMPT = (
    "You are an orchestrator. Break the task down and delegate each part to the "
    "appropriate specialist tool. Then synthesize one clear final answer."
)


def subagent(agent, name: str, description: str):
    """Wrap an Agent as a delegable tool. The tool takes a single `question` string."""
    def _call(question: str) -> str:
        return agent.run(question)

    return StructuredTool.from_function(func=_call, name=name, description=description)


def supervisor(model, subagents: list, *, system_prompt: str | None = None, **kwargs) -> Agent:
    """Build an orchestrator Agent whose tools are the given sub-agent tools."""
    return Agent(
        model,
        tools=subagents,
        system_prompt=system_prompt or _DEFAULT_SUPERVISOR_PROMPT,
        **kwargs,
    )
