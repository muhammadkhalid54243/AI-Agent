"""Evaluation runner — scores a composed Agent on outcome AND trajectory.

Runs the real Agent.run_traced and reads the tool trajectory from its events, so
evaluation exercises the production code path. Open-ended cases go to an LLM judge.
"""

from framework.eval.dataset import EVAL_SET
from framework.eval.judge import judge_answer

JUDGE_PASS = 4


def _trajectory(events) -> list[str]:
    return [e["tool"] for e in events if e.get("kind") == "tool_call"]


def score_case(agent, judge_model, case) -> dict:
    result = agent.run_traced(case["input"])
    answer = result["answer"]
    traj = _trajectory(result["events"])

    if case["type"] == "judge":
        verdict = judge_answer(judge_model, case["input"], answer, case["rubric"])
        passed = verdict["score"] >= JUDGE_PASS
        detail = f"judge {verdict['score']}/5 — {verdict['reason']}"
    else:
        outcome_ok = all(s.lower() in answer.lower() for s in case.get("expect_contains", []))
        traj_ok = all(t in traj for t in case.get("expect_tools", []))
        passed = outcome_ok and traj_ok
        detail = f"outcome={'OK' if outcome_ok else 'FAIL'}, trajectory={'OK' if traj_ok else 'FAIL'}"

    return {"id": case["id"], "type": case["type"], "passed": passed,
            "detail": detail, "answer": answer, "trajectory": traj}


def run_suite(agent, judge_model=None) -> list[dict]:
    judge_model = judge_model or "groq:llama-3.3-70b-versatile"
    return [score_case(agent, judge_model, c) for c in EVAL_SET]
