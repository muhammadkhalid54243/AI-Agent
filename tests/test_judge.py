"""Citation grounding — the deterministic half of the verifier (offline, no model)."""

from framework.eval.judge import quote_grounded


PASSAGES = [
    "SHR 2024 — Key issues, p18:\n6,161 prisoners were on death row at the end of 2024.",
    "SHR 2023 — Key issues, p40:\nAbout 3,800 prisoners were on death row.",
]


def test_verbatim_quote_is_grounded():
    assert quote_grounded("6,161 prisoners were on death row", PASSAGES)


def test_whitespace_and_case_are_normalized():
    assert quote_grounded("  6,161   PRISONERS were on   death row ", PASSAGES)


def test_quote_absent_from_passages_is_not_grounded():
    # The India hallucination class: plausible sentence, not actually in any passage.
    assert not quote_grounded("4,200 prisoners were on death row in India", PASSAGES)


def test_too_short_quote_fails_closed():
    assert not quote_grounded("2024", PASSAGES)        # unverifiably short -> not grounded
    assert not quote_grounded("", PASSAGES)
