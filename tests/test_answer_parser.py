"""Cited-answer output parser — deterministic bullet formatting (offline)."""

from framework.structured.answer_parser import CitedAnswerParser, parse_units, render_units

ANSWER = (
    '6,161 prisoners were on death row in 2024. [SHR 2024 — Key issues, p11]\n'
    '"Data showed that 6,161 prisoners were on death row in 2024."\n\n'
    '158 prisoners were on death row in Sindh. [SHR 2023 — Cultural rights, p69]\n'
    '"158 prisoners were on death row, almost all for murder."'
)


def test_parse_units_extracts_fact_citation_quote():
    u = parse_units(ANSWER)
    assert len(u) == 2
    assert u[0]["fact"] == "6,161 prisoners were on death row in 2024."
    assert u[0]["citation"] == "SHR 2024 — Key issues, p11"
    assert u[0]["quote"].startswith("Data showed that 6,161")
    assert u[1]["citation"] == "SHR 2023 — Cultural rights, p69"


def test_parse_handles_missing_quote():
    u = parse_units("No quote here. [SHR 2024 — Punjab, p20]")
    assert len(u) == 1 and u[0]["quote"] == ""


def test_render_is_bulleted_paragraph_separated_quote_on_own_line():
    out = CitedAnswerParser().parse(ANSWER)
    assert out.startswith("- ")                                  # bullet
    assert out.count("\n- ") == 1                                # two bullets total
    assert "\n\n- " in out                                       # blank line between cited answers
    assert '\n  "Data showed that 6,161' in out                 # quote on its own indented line


def test_render_merges_identical_facts_with_combined_sources():
    out = render_units([
        {"fact": "Everyone is born free and equal.", "citation": "SHR 2023 — UDHR, p353", "quote": "q"},
        {"fact": "Everyone is born free and equal.", "citation": "SHR 2024 — UDHR, p350", "quote": "q"},
    ])
    assert out.count("Everyone is born free and equal.") == 1   # stated once
    assert "SHR 2023 — UDHR, p353; SHR 2024 — UDHR, p350" in out  # sources combined into one bracket
    assert out.startswith("- ")


def test_text_without_citation_passes_through_unchanged():
    msg = "The reports don't appear to cover that."
    assert CitedAnswerParser().parse(msg) == msg                 # e.g. the NOT_COVERED abstention
    assert parse_units(msg) == []


def test_preamble_before_first_citation_becomes_the_fact():
    out = CitedAnswerParser().parse("- In 2024, 6,161 were on death row. [SHR 2024 — Key issues, p11]")
    assert out == '- In 2024, 6,161 were on death row. [SHR 2024 — Key issues, p11]'  # leading bullet not doubled
