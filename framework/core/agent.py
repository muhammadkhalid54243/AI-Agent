"""Agent — a thin, modular composer over LangGraph's create_agent prebuilt.

Only `model` is required. Tools, a system prompt, and fallbacks are optional;
memory, safety middleware, RAG, and MCP are added in later phases. `model` may
be a 'provider:model' spec string or an already-built chat model (handy for tests).
"""

from langchain.agents import create_agent
from langgraph.types import Command

from framework.core.models import resolve_model
from framework.middleware.guardrails import ApprovalRequired


class Agent:
    def __init__(self, model, *, tools=None, system_prompt=None, fallbacks=None,
                 memory=None, middleware=None):
        self._model = model if not isinstance(model, str) else resolve_model(model, fallbacks)
        self.tools = list(tools) if tools else []
        self._system_prompt = system_prompt
        self._memory = memory
        self._middleware = list(middleware) if middleware else []
        self._graph = self._build()

    def _build(self):
        kwargs = {"model": self._model, "tools": self.tools}
        if self._system_prompt:
            kwargs["system_prompt"] = self._system_prompt
        if self._memory is not None:
            kwargs["checkpointer"] = self._memory
        if self._middleware:
            kwargs["middleware"] = self._middleware
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
        return self._extract(result, thread_id)

    def resume(self, thread_id: str, approve: bool) -> str:
        """Resume a run paused for approval. approve=True executes; False blocks."""
        config = {"configurable": {"thread_id": thread_id}}
        decision = "approve" if approve else "reject"
        result = self._graph.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            config=config,
        )
        return self._extract(result, thread_id)

    @staticmethod
    def _extract(result: dict, thread_id) -> str:
        """Return the final answer, or raise ApprovalRequired if the run is paused."""
        if "__interrupt__" in result:
            requests = []
            for it in result["__interrupt__"]:
                for ar in it.value.get("action_requests", []):
                    requests.append({"name": ar["name"], "args": ar["args"]})
            raise ApprovalRequired(thread_id, requests)
        return result["messages"][-1].content
