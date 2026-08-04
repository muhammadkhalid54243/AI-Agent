"""Structured trajectory logging.

Every step the agent takes becomes a structured event with a timestamp and
duration. In production you'd ship these as JSON lines to a log aggregator;
here we collect them in memory and can dump them as JSON. Observability = being
able to answer "what did the agent actually do, and where did the time/money go?"
after the fact, without re-running it.
"""

import json
import time
from contextlib import contextmanager


class TrajectoryLogger:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.events = []

    def log(self, kind: str, **fields):
        self.events.append({"ts": round(time.time(), 3), "kind": kind, **fields})

    @contextmanager
    def timed(self, kind: str, **fields):
        """Time a block and log its duration (e.g., a model call or tool run)."""
        start = time.perf_counter()
        error = None
        try:
            yield
        except Exception as e:
            error = repr(e)
            raise
        finally:
            ms = round((time.perf_counter() - start) * 1000, 1)
            self.log(kind, duration_ms=ms, error=error, **fields)

    def summary(self) -> dict:
        model_calls = [e for e in self.events if e["kind"] == "model_call"]
        tool_calls = [e for e in self.events if e["kind"] == "tool_call"]
        blocked = [e for e in self.events if e["kind"] == "tool_blocked"]
        total_ms = sum(e.get("duration_ms", 0) for e in self.events)
        return {
            "request_id": self.request_id,
            "model_calls": len(model_calls),
            "tool_calls": len(tool_calls),
            "blocked_calls": len(blocked),
            "total_ms": round(total_ms, 1),
        }

    def dump(self) -> str:
        return json.dumps({"summary": self.summary(), "events": self.events}, indent=2)
