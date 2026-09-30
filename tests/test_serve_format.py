"""Serve response-generation format — offline (fake retriever, no model).

The multi-doc answer contract: passages are numbered and headed with their report +
page, and the prompt tells the model to put each fact on its own line, cite each, and
list BOTH years when they differ rather than merging. These check the wiring that makes
that possible, without a network call.
"""

from framework.ingest.ir import GlossaryEntry
from framework.serve.app import RAG_SYS, _context, _glossary_note, _prompt


class _FakeRetriever:
    glossary = {"HRCP": GlossaryEntry(expansion="Human Rights Commission of Pakistan")}

    def _search(self, query, k):
        return [
            {"doc_id": "2024_SHR_English", "section_path": ["Key issues"], "page": 42,
             "text": "6,161 prisoners were on death row."},
            {"doc_id": "2023_SHR_English", "section_path": ["Key issues"], "page": 40,
             "text": "About 3,800 prisoners were on death row."},
        ]

    def doc_label(self, doc_id):
        return {"2024_SHR_English": "SHR 2024", "2023_SHR_English": "SHR 2023"}.get(doc_id, doc_id)


def test_context_labels_each_passage_with_report_and_page():
    ctx = _context("how many on death row?", {"retriever": _FakeRetriever()})
    assert "SHR 2024 — Key issues, p42:" in ctx
    assert "SHR 2023 — Key issues, p40:" in ctx
    assert "6,161 prisoners were on death row." in ctx
    assert "About 3,800 prisoners were on death row." in ctx


def test_glossary_note_expands_acronyms_in_question():
    note = _glossary_note("what is HRCP", _FakeRetriever().glossary)
    assert "HRCP = Human Rights Commission of Pakistan" in note
    assert _glossary_note("how many on death row", {}) == ""


def test_prompt_instructs_per_line_citation_and_no_merge():
    ctx = _context("how many on death row?", {"retriever": _FakeRetriever()})
    p = _prompt("how many on death row?", ctx)
    assert "own line" in p.lower()
    assert "[SHR 2024, p18]" in p           # the citation-format example
    assert "bare passage number" in p.lower()  # forbids citing [3] instead of the report
    assert "never merge" in p.lower()
    assert "how many on death row?" in p    # the question is carried through
