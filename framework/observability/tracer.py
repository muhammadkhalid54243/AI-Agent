"""TrajectoryTracer — a callback handler that records what the agent did.

Attached via config={"callbacks": [tracer]}, it times each model call and tool
call by matching start/end on run_id, then summarizes the trajectory. This is the
observability layer: see model_calls, tool_calls, and total latency after a run.
"""

import time

from langchain_core.callbacks import BaseCallbackHandler


class TrajectoryTracer(BaseCallbackHandler):
    def __init__(self):
        self.events = []
        self._starts = {}  # run_id -> (kind, start_perf, meta)

    # chat models call on_chat_model_start; completion models call on_llm_start
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._starts[run_id] = ("model_call", time.perf_counter(), {})

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._starts[run_id] = ("model_call", time.perf_counter(), {})

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        self._finish(run_id)

    def on_tool_start(self, serialized, input_str, *, run_id=None, name=None, **kwargs):
        tool_name = name or (serialized or {}).get("name", "tool")
        self._starts[run_id] = ("tool_call", time.perf_counter(), {"tool": tool_name})

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        self._finish(run_id)

    def _finish(self, run_id):
        entry = self._starts.pop(run_id, None)
        if not entry:
            return
        kind, start, meta = entry
        self.events.append({
            "kind": kind,
            "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            **meta,
        })

    def summary(self) -> dict:
        model_calls = sum(1 for e in self.events if e["kind"] == "model_call")
        tool_calls = sum(1 for e in self.events if e["kind"] == "tool_call")
        total_ms = round(sum(e.get("duration_ms", 0) for e in self.events), 1)
        return {"model_calls": model_calls, "tool_calls": tool_calls, "total_ms": total_ms}
