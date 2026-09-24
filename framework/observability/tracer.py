"""TrajectoryTracer — a callback handler that records what the agent did.

Attached via config={"callbacks": [tracer]}, it times each model call and tool
call by matching start/end on run_id, then summarizes the trajectory. This is the
observability layer: see model_calls, tool_calls, and total latency after a run.
"""

import time

from langchain_core.callbacks import BaseCallbackHandler

from framework.context.budget import measure_messages


class TrajectoryTracer(BaseCallbackHandler):
    def __init__(self):
        self.events = []
        self._starts = {}  # run_id -> (kind, start_perf, meta)
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0

    # chat models call on_chat_model_start; completion models call on_llm_start
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        meta = {"context": self._measure(messages)}
        self._starts[run_id] = ("model_call", time.perf_counter(), meta)

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._starts[run_id] = ("model_call", time.perf_counter(), {})

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        usage = self._extract_usage(response)
        if usage:
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            details = usage.get("input_token_details") or {}
            self.cache_read_tokens += details.get("cache_read", 0) or 0
            self.cache_write_tokens += details.get("cache_creation", 0) or 0
            entry = self._starts.get(run_id)
            if entry:
                entry[2]["usage"] = usage
        self._finish(run_id)

    @staticmethod
    def _measure(messages) -> dict:
        """messages arrives as List[List[BaseMessage]]; measure the first batch."""
        try:
            flat = messages[0] if messages and isinstance(messages[0], list) else messages
            return measure_messages(flat)
        except Exception:
            return {"system": 0, "history": 0, "tool_results": 0, "total": 0}

    @staticmethod
    def _extract_usage(response) -> dict | None:
        """Pull provider-reported token usage from an LLMResult, if present."""
        try:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                return dict(usage)
        except Exception:
            pass
        try:
            tu = (response.llm_output or {}).get("token_usage") or {}
            if tu:
                return {"input_tokens": tu.get("prompt_tokens", 0),
                        "output_tokens": tu.get("completion_tokens", 0),
                        "total_tokens": tu.get("total_tokens", 0)}
        except Exception:
            pass
        return None

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

    def token_report(self) -> dict:
        """Per-run token accounting: provider usage plus per-section peak context.

        - input/output_tokens: authoritative provider counts (summed over calls).
        - peak_context_tokens: largest single message list sent (tiktoken estimate).
        - context_by_section: section totals at that peak turn — shows where the
          budget concentrates (system / history / tool_results).
        """
        contexts = [e["context"] for e in self.events
                    if e["kind"] == "model_call" and "context" in e]
        peak = max(contexts, key=lambda c: c.get("total", 0), default=None)
        cache_hit_rate = (round(self.cache_read_tokens / self.input_tokens, 3)
                          if self.input_tokens else 0.0)
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_hit_rate": cache_hit_rate,
            "peak_context_tokens": peak["total"] if peak else 0,
            "context_by_section": {k: v for k, v in (peak or {}).items() if k != "total"},
        }
