"""Context evaluation — the baseline the whole context layer is measured against.

score_context() runs an agent over a set of tasks and reports the numbers that
matter for context management: tokens per task, peak context size, turns to
completion, and (optionally) cost. Phase 0 records these; every later phase must
show its delta against this baseline instead of claiming improvement by vibes.
"""


def _cost(input_tokens, output_tokens, price_per_mtok) -> float | None:
    if not price_per_mtok:
        return None
    return round(input_tokens / 1e6 * price_per_mtok.get("input", 0)
                 + output_tokens / 1e6 * price_per_mtok.get("output", 0), 6)


def score_context(agent, tasks, *, price_per_mtok=None) -> dict:
    """Run `agent` over `tasks` (list of input strings) and aggregate token/cost metrics.

    price_per_mtok: optional {"input": usd, "output": usd} per 1M tokens.
    Returns {"rows": [...per task...], "totals": {...}}.
    """
    rows = []
    for task in tasks:
        result = agent.run_traced(task)
        tok = result["tokens"]
        summ = result["summary"]
        row = {
            "task": task[:48],
            "input_tokens": tok["input_tokens"],
            "output_tokens": tok["output_tokens"],
            "total_tokens": tok["total_tokens"],
            "peak_context": tok["peak_context_tokens"],
            "model_calls": summ["model_calls"],
            "tool_calls": summ["tool_calls"],
            "context_by_section": tok.get("context_by_section", {}),
        }
        cost = _cost(row["input_tokens"], row["output_tokens"], price_per_mtok)
        if cost is not None:
            row["cost_usd"] = cost
        rows.append(row)

    totals = {
        "tasks": len(rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "peak_context": max((r["peak_context"] for r in rows), default=0),
        "model_calls": sum(r["model_calls"] for r in rows),
    }
    cost = _cost(totals["input_tokens"], totals["output_tokens"], price_per_mtok)
    if cost is not None:
        totals["cost_usd"] = cost
    return {"rows": rows, "totals": totals}
