"""Agent — a thin, modular composer over LangGraph's create_agent prebuilt.

Only `model` is required. Tools, a system prompt, and fallbacks are optional;
memory, safety middleware, RAG, and MCP are added in later phases. `model` may
be a 'provider:model' spec string or an already-built chat model (handy for tests).
"""

from langchain.agents import create_agent

from framework.core.models import resolve_model


class Agent:
    def __init__(self, model, *, tools=None, system_prompt=None, fallbacks=None):
        self._model = model if not isinstance(model, str) else resolve_model(model, fallbacks)
        self.tools = list(tools) if tools else []
        self._system_prompt = system_prompt
        self._graph = self._build()

    def _build(self):
        kwargs = {"model": self._model, "tools": self.tools}
        if self._system_prompt:
            kwargs["system_prompt"] = self._system_prompt
        return create_agent(**kwargs)

    def add_tools(self, tools: list) -> None:
        self.tools.extend(tools)
        self._graph = self._build()

    def run(self, message: str, thread_id: str | None = None) -> str:
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        result = self._graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        return result["messages"][-1].content
