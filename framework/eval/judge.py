"""LLM-as-judge — score an open-ended answer against a rubric, validated."""

from pydantic import BaseModel

from framework.structured import extract


class _Score(BaseModel):
    score: int
    reason: str


def judge_answer(model, question: str, answer: str, rubric: str) -> dict:
    prompt = f"Question:\n{question}\n\nRubric:\n{rubric}\n\nAnswer to score:\n{answer}"
    result = extract(
        model, prompt, _Score,
        instruction="Score the answer from 1-5 against the rubric. Be strict.",
    )
    return {"score": int(result.score), "reason": result.reason}
