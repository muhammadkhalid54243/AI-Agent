"""Milestone 9 demo: an evaluation harness.

Runs the agent against a fixed eval set and scores each case on BOTH
outcome (did the answer contain the right content) and trajectory (did it
use the right tools). Open-ended cases are scored by an LLM judge.

Prints a scorecard you can compare across prompt/model changes to catch
regressions.

Run:
    python eval_demo.py
"""

import os

from dotenv import load_dotenv

from agent.eval.dataset import EVAL_SET
from agent.eval.judge import judge_answer
from agent.eval.runner import run_agent
from agent.llms.factory import get_llm

load_dotenv()

JUDGE_PASS_THRESHOLD = 4  # open-ended cases need score >= 4/5 to pass


def score_case(llm, case):
    answer, trajectory = run_agent(llm, case["input"])

    if case["type"] == "judge":
        verdict = judge_answer(llm, case["input"], answer, case["rubric"])
        passed = verdict["score"] >= JUDGE_PASS_THRESHOLD
        detail = f"judge {verdict['score']}/5 — {verdict['reason']}"
        return passed, answer, trajectory, detail

    # outcome check
    outcome_ok = all(s.lower() in answer.lower() for s in case.get("expect_contains", []))

    # trajectory check: every expected tool must appear in the actual path
    expected_tools = case.get("expect_tools", [])
    trajectory_ok = all(t in trajectory for t in expected_tools)

    passed = outcome_ok and trajectory_ok
    detail = f"outcome={'OK' if outcome_ok else 'FAIL'}, trajectory={'OK' if trajectory_ok else 'FAIL'}"
    return passed, answer, trajectory, detail


def main():
    provider = os.environ.get("LLM_PROVIDER", "groq")
    if provider == "groq" and not os.environ.get("GROQ_MODEL"):
        os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
    llm = get_llm(provider)

    model = os.environ.get("GROQ_MODEL", "default")
    print(f"Running eval set ({len(EVAL_SET)} cases) on {provider}/{model}\n")
    print("-" * 70)

    passed_count = 0
    for case in EVAL_SET:
        passed, answer, trajectory, detail = score_case(llm, case)
        passed_count += passed
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {case['id']:<20} ({case['type']})")
        print(f"       {detail}")
        if case["type"] != "judge":
            print(f"       tools called: {trajectory}")
        print(f"       answer: {answer[:80].strip()}...")
        print("-" * 70)

    total = len(EVAL_SET)
    print(f"\nSCORE: {passed_count}/{total} passed ({100 * passed_count // total}%)")


if __name__ == "__main__":
    main()
