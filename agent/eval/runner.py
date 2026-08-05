"""Evaluation runner — scores the canonical agent on the eval set.

This does NOT reimplement the agent loop. It runs the real ProductionAgent and
reads its trajectory out of the structured event log, so evaluation tests the
same code path that runs in production. Two axes are scored:
  - outcome:    does the final answer contain the expected content?
  - trajectory: did the agent actually call the expected tools?
Open-ended cases are graded by an LLM judge.

Run the whole suite:
    python -m agent.eval.runner
"""

import os

from dotenv import load_dotenv

from agent.eval.dataset import EVAL_SET
from agent.eval.judge import judge_answer
from agent.llms.factory import get_llm
from agent.production.agent import ProductionAgent
from agent.production.resilient_llm import ResilientLLM
from agent.safety.guardrails import Guardrails

JUDGE_PASS_THRESHOLD = 4  # open-ended cases need score >= 4/5


def build_eval_agent(llm) -> ProductionAgent:
    """A ProductionAgent configured for evaluation runs."""
    resilient = ResilientLLM(providers=[("eval", llm)], max_retries=1)
    guards = Guardrails(max_calls=8)
    return ProductionAgent(resilient, guards)


def _trajectory(events) -> list[str]:
    """Extract the ordered list of executed tool names from the event log."""
    return [e["tool"] for e in events if e.get("kind") == "tool_call"]


def score_case(agent, judge_llm, case) -> dict:
    result = agent.run(case["input"])
    answer = result["answer"]
    trajectory = _trajectory(result["events"])

    if case["type"] == "judge":
        verdict = judge_answer(judge_llm, case["input"], answer, case["rubric"])
        passed = verdict["score"] >= JUDGE_PASS_THRESHOLD
        detail = f"judge {verdict['score']}/5 — {verdict['reason']}"
    else:
        outcome_ok = all(s.lower() in answer.lower() for s in case.get("expect_contains", []))
        trajectory_ok = all(t in trajectory for t in case.get("expect_tools", []))
        passed = outcome_ok and trajectory_ok
        detail = f"outcome={'OK' if outcome_ok else 'FAIL'}, trajectory={'OK' if trajectory_ok else 'FAIL'}"

    return {"id": case["id"], "type": case["type"], "passed": passed,
            "detail": detail, "trajectory": trajectory, "answer": answer}


def run_suite(llm, judge_llm=None) -> list[dict]:
    """Run every case and return per-case results."""
    agent = build_eval_agent(llm)
    judge_llm = judge_llm or llm
    return [score_case(agent, judge_llm, case) for case in EVAL_SET]


def main():
    load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "groq")
    if provider == "groq" and not os.environ.get("GROQ_MODEL"):
        os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
    llm = get_llm(provider)

    print(f"Running eval set ({len(EVAL_SET)} cases) on {provider}\n" + "-" * 70)
    results = run_suite(llm)

    passed = 0
    for r in results:
        passed += r["passed"]
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']:<20} ({r['type']})")
        print(f"       {r['detail']}")
        if r["type"] != "judge":
            print(f"       tools: {r['trajectory']}")
        print(f"       answer: {r['answer'][:80].strip()}...")
        print("-" * 70)

    total = len(results)
    print(f"\nSCORE: {passed}/{total} passed ({100 * passed // total}%)")


if __name__ == "__main__":
    main()
