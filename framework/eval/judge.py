"""LLM-as-judge — score an open-ended answer against a rubric, and VERIFY that each
citation in an answer is actually grounded in its source passage.

The verifier is the product's last line of defence on citation trust: the retrieval gate
cannot reject a topical near-miss ("death sentences in India" retrieves Pakistan passages
at high similarity), and a confident-but-wrong model can still attribute a claim to a
passage that does not support it (the Asia-Bibi-as-Ahmadi class). Grounding is a TWO-part
gate — a deterministic check that the quoted segment really occurs in the cited passage
(we never trust the model to self-report presence), AND a model judgment that the claim is
entailed by that segment. A claim failing either is ungrounded and is dropped by serve.
"""

import re

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


class _Claim(BaseModel):
    claim: str                  # one factual statement from the answer
    cited_source: str           # the source it cited, e.g. "SHR 2024 — Key issues, p18"
    supporting_quote: str       # the verbatim sentence from the passages that supports it
    entailed: bool              # does that quote actually support the claim?
    relevant: bool = True       # does the claim actually answer the QUESTION? (off-topic -> drop)


class _Verdict(BaseModel):
    claims: list[_Claim]


def _normalize(s: str) -> str:
    """Collapse whitespace and case so a quote matches its passage despite reflow."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def quote_grounded(quote: str, passages, *, min_len: int = 12) -> bool:
    """Deterministic: does `quote` actually occur (verbatim, whitespace/case-insensitive) in
    any passage? A too-short quote can't be meaningfully verified, so it fails closed."""
    q = _normalize(quote)
    if len(q) < min_len:
        return False
    blob = _normalize("\n".join(passages))
    return q in blob


VERIFY_INSTRUCTION = (
    "You are auditing an answer's citations against the SOURCE PASSAGES. For EACH distinct "
    "factual statement in the answer, return: the claim, the source it cited, the EXACT "
    "sentence from the passages that supports it (copied verbatim — not paraphrased), whether "
    "that sentence genuinely entails the claim, and whether the claim actually answers the "
    "QUESTION (relevant). If no passage supports a claim, set entailed=false and give the closest "
    "quote (or empty). If the claim is off-topic for the question, set relevant=false.")


def verify_answer(model, question: str, answer: str, passages: list) -> list:
    """Decompose `answer` into claims and judge each. Returns a list of dicts with a final
    `grounded` = (model says entailed) AND (the quote really is in the passages).

    The verifier MUST be deterministic (a wobbling entailment judgment false-drops good
    citations — measured), so a bare model spec is resolved at temperature=0. A caller that
    already built a temp=0 model (serve does) passes it straight through."""
    if isinstance(model, str):
        from framework.core.models import resolve_model
        model = resolve_model(model, temperature=0)
    text = (f"QUESTION:\n{question}\n\nSOURCE PASSAGES:\n" + "\n\n".join(passages)
            + f"\n\nANSWER TO AUDIT:\n{answer}")
    verdict = extract(model, text, _Verdict, instruction=VERIFY_INSTRUCTION)
    out = []
    for c in verdict.claims:
        grounded = bool(c.entailed) and quote_grounded(c.supporting_quote, passages)
        out.append({"claim": c.claim, "cited_source": c.cited_source,
                    "supporting_quote": c.supporting_quote,
                    "entailed": bool(c.entailed), "relevant": bool(c.relevant),
                    "grounded": grounded})
    return out
