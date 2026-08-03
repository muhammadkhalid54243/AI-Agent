"""LLM-as-judge for open-ended answers.

No exact string to match, so a second LLM scores the answer against a rubric
and returns a 1-5 score with a reason. We force structured JSON so the runner
can act on the number programmatically (the M2 skill, applied to grading).
"""

import json

JUDGE_PROMPT = (
    "You are a strict evaluation judge. Score the answer against the rubric on a 1-5 scale "
    "(5 = excellent, 1 = poor). Be critical; do not give a high score to fluffy or inaccurate "
    "answers. Return ONLY valid JSON: "
    '{"score": <1-5>, "reason": "<one sentence>"}'
)


def judge_answer(llm, question, answer, rubric):
    user = (
        f"Question:\n{question}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Answer to evaluate:\n{answer}"
    )
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user},
    ]
    raw = llm.send(messages)
    try:
        return _parse(raw)
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0, "reason": "judge returned unparseable output"}


def _parse(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
    return {"score": int(data["score"]), "reason": data.get("reason", "")}
