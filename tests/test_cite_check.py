"""Deterministic citation faithfulness (offline, no model)."""

from framework.eval.cite_check import correct_citation, parse_citation

# Authoritative source lines the retrieved passages actually carry (what serve builds from
# _loc/_page/doc_label): the Balochistan passage on printed p18 really is chapter "Balochistan".
AUTH = {
    ("SHR 2024", "p18"): "Balochistan",
    ("SHR 2024", "p40"): "Key issues",
    ("SHR 2023", "p5"): "Universal Declaration of Human Rights",
}


def test_parse_single():
    assert parse_citation("SHR 2024 — Balochistan, p18") == (["SHR 2024"], "Balochistan", "p18")


def test_parse_merged():
    assert parse_citation("SHR 2023; SHR 2024 — Key issues, p40") == (
        ["SHR 2023", "SHR 2024"], "Key issues", "p40")


def test_parse_pdf_page():
    assert parse_citation("SHR 2024 — Key issues, PDF p7") == (["SHR 2024"], "Key issues", "PDF p7")


def test_wrong_chapter_is_corrected():
    # The measured bug: a Balochistan passage cited under Khyber Pakhtunkhwa -> corrected.
    assert correct_citation("SHR 2024 — Khyber Pakhtunkhwa, p18", AUTH) == "SHR 2024 — Balochistan, p18"


def test_correct_chapter_is_unchanged():
    assert correct_citation("SHR 2024 — Balochistan, p18", AUTH) == "SHR 2024 — Balochistan, p18"


def test_unmappable_page_strips_chapter():
    # No retrieved passage backs (SHR 2024, p99) -> degrade to report + page, never assert a chapter.
    assert correct_citation("SHR 2024 — Sindh, p99", AUTH) == "SHR 2024, p99"


def test_merged_citation_corrected_when_a_report_agrees():
    out = correct_citation("SHR 2023; SHR 2024 — Wrong, p5", AUTH)
    assert out == "SHR 2023; SHR 2024 — Universal Declaration of Human Rights, p5"


def test_no_page_left_as_is():
    # Can't anchor without a page -> leave untouched rather than guess.
    assert correct_citation("SHR 2024 — Some Section", AUTH) == "SHR 2024 — Some Section"


def test_already_degraded_roundtrips():
    assert correct_citation("SHR 2024, p18", AUTH) == "SHR 2024 — Balochistan, p18"
