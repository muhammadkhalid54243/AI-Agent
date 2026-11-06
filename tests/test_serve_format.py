"""Serve response-generation format — offline (fake retriever, no model).

The multi-doc answer contract: passages are numbered and headed with their report +
page, and the prompt tells the model to put each fact on its own line, cite each, and
list BOTH years when they differ rather than merging. These check the wiring that makes
that possible, without a network call.
"""

import importlib

from framework.ingest.ir import GlossaryEntry
from framework.serve.app import RAG_SYS, _context, _glossary_note, _prompt

# framework.serve.__init__ rebinds the name `app` to the Starlette instance, shadowing the
# submodule, so `import framework.serve.app` yields the wrong object — fetch the module itself.
serve_app = importlib.import_module("framework.serve.app")


class _FakeRetriever:
    glossary = {"HRCP": GlossaryEntry(expansion="Human Rights Commission of Pakistan")}

    def retrieve(self, query, k):
        return [
            {"doc_id": "2024_SHR_English", "section_path": ["Key issues"], "page": 42,
             "printed_page": 48, "text": "6,161 prisoners were on death row."},
            {"doc_id": "2023_SHR_English", "section_path": ["Key issues"], "page": 40,
             "printed_page": None, "text": "About 3,800 prisoners were on death row."},
        ]

    def doc_label(self, doc_id):
        return {"2024_SHR_English": "SHR 2024", "2023_SHR_English": "SHR 2023"}.get(doc_id, doc_id)


def test_context_labels_each_passage_with_printed_page_and_fallback():
    ctx = _context("how many on death row?", {"retriever": _FakeRetriever()})
    assert "SHR 2024 — Key issues, p48:" in ctx           # PRINTED page cited (not PDF index 42)
    assert "SHR 2023 — Key issues, PDF p40:" in ctx        # explicit fallback (no printed page)
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
    assert "[SHR 2024 — Key issues, p18]" in p   # citation example now carries the SECTION
    assert "word for word" in p.lower()          # verbatim supporting quote (the segment) required
    assert "bare passage number" in p.lower()    # forbids citing [3] instead of the report
    assert "never merge" in p.lower()
    assert "how many on death row?" in p         # the question is carried through


# --- deterministic abstention gate ---------------------------------------

def test_system_prompt_has_no_model_abstain_escape_hatch():
    # The flip source: the OLD prompt let the model decide to say "don't appear to cover it".
    # That decision now lives in the gate, so it must be gone from the prompt.
    assert "don't appear to cover" not in RAG_SYS.lower()
    assert "already been made" in RAG_SYS.lower()   # the decision is the gate's, not the model's


class _SpyModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        class _R:
            content = "a grounded answer"
        return _R()


def _state(confident: bool, model):
    r = _FakeRetriever()
    r.confident = lambda hits, **kw: confident      # stub the gate decision
    return {"mode": "rag", "retriever": r, "model": model}


def test_gate_abstains_without_calling_the_model():
    spy = _SpyModel()
    serve_app._state = _state(confident=False, model=spy)
    try:
        assert serve_app.rag_answer("something obscure") == serve_app.NOT_COVERED
        assert spy.calls == 0                        # abstained BEFORE spending a model call
    finally:
        serve_app._state = None


def test_gate_answers_when_confident():
    spy = _SpyModel()
    serve_app._state = _state(confident=True, model=spy)
    try:
        assert serve_app.rag_answer("how many on death row?") == "a grounded answer"
        assert spy.calls == 1
    finally:
        serve_app._state = None


# --- multi-turn (G1) ------------------------------------------------------

class _EchoModel:
    """Records the last prompt it saw and echoes a fixed answer (for condense/history)."""
    def __init__(self, reply="6,161 [SHR 2024 — Key issues, p18]"):
        self.reply, self.last = reply, None

    def invoke(self, messages):
        self.last = messages[0].content
        class _R:
            content = self.reply
        return _R()


def test_condense_is_a_noop_without_history():
    m = _EchoModel()
    assert serve_app._condense(m, "", "what about 2023?") == "what about 2023?"
    assert m.last is None                         # no model call on the first turn


def test_condense_rewrites_followup_with_history():
    m = _EchoModel(reply="How many death sentences in 2023?")
    out = serve_app._condense(m, "User: How many in 2024?\nAssistant: 6,161", "what about 2023?")
    assert out == "How many death sentences in 2023?"
    assert "FOLLOW-UP: what about 2023?" in m.last   # history + follow-up were sent


def test_multi_turn_feeds_prior_turns_into_the_answer_prompt():
    m = _EchoModel()
    serve_app._state = _state(confident=True, model=m)
    serve_app.HISTORY.clear("tt")
    try:
        serve_app.rag_answer("How many on death row in 2024?", thread_id="tt")
        assert len(serve_app.HISTORY.get("tt")) == 1          # turn recorded
        serve_app.rag_answer("what about 2023?", thread_id="tt")
        assert "CONVERSATION SO FAR:" in m.last               # 2nd turn saw history
        assert "How many on death row in 2024?" in m.last
        assert len(serve_app.HISTORY.get("tt")) == 2
    finally:
        serve_app.HISTORY.clear("tt")
        serve_app._state = None


# --- citation verifier (G3) ----------------------------------------------

def test_render_claims_shows_fact_citation_and_segment_quote():
    out = serve_app._render_claims([
        {"claim": "6,161 were on death row", "cited_source": "SHR 2024 — Key issues, p18",
         "supporting_quote": "6,161 prisoners were on death row."}])
    assert "- 6,161 were on death row [SHR 2024 — Key issues, p18]" in out   # bulleted
    assert '\n  "6,161 prisoners were on death row."' in out                 # quote on its OWN line


def test_render_merges_identical_statement_across_reports():
    # Same text reproduced in two reports -> stated ONCE, citations combined (not duplicated).
    out = serve_app._render_claims([
        {"claim": "Everyone is born free and equal in dignity and rights.",
         "cited_source": "SHR 2023 — UDHR, p353", "supporting_quote": "All human beings are born free"},
        {"claim": "Everyone is born free and equal in dignity and rights.",
         "cited_source": "SHR 2024 — UDHR, p350", "supporting_quote": "All human beings are born free"},
    ])
    assert out.count("Everyone is born free and equal in dignity and rights.") == 1   # stated once
    assert "SHR 2023 — UDHR, p353; SHR 2024 — UDHR, p350" in out                      # merged sources


def test_render_separates_cited_answers_with_a_blank_line():
    out = serve_app._render_claims([
        {"claim": "A", "cited_source": "SHR 2024 — s, p1", "supporting_quote": "qa"},
        {"claim": "B", "cited_source": "SHR 2023 — s, p2", "supporting_quote": "qb"},
    ])
    assert "\n\n" in out                                   # paragraph break between answers
    assert out.index("A [") < out.index("\n\n") < out.index("B [")


def test_prompt_instructs_dedup_and_paragraph_separation():
    p = _prompt("q", _context("q", {"retriever": _FakeRetriever()}))
    assert "combine the sources" in p.lower()              # dedup rule present
    assert "blank line" in p.lower()                       # one-paragraph-per-answer rule present


def test_verify_drops_ungrounded_claims(monkeypatch):
    claims = [
        {"claim": "A", "cited_source": "SHR 2024 — s, p1", "supporting_quote": "q1", "grounded": True},
        {"claim": "B", "cited_source": "SHR 2024 — s, p2", "supporting_quote": "q2", "grounded": False},
    ]
    monkeypatch.setattr("framework.eval.judge.verify_answer", lambda *a, **k: claims)
    # A hit backs (SHR 2024, p1) as chapter "s", so the deterministic correction leaves it intact.
    hits = [{"doc_id": "2024_SHR_English", "printed_page": 1, "chapter": "s"}]
    out = serve_app._verify({"model": None}, "q", "A. B.", ["passage"], _FakeRetriever(), hits)
    assert "A [SHR 2024 — s, p1]" in out
    assert "B [" not in out                        # ungrounded claim dropped


def test_used_chunk_ids_matches_only_cited_hits():
    r = _FakeRetriever()                                 # doc_label: 2024_SHR_English -> "SHR 2024"
    hits = [
        {"chunk_id": "c1", "doc_id": "2024_SHR_English", "printed_page": 22},
        {"chunk_id": "c2", "doc_id": "2023_SHR_English", "printed_page": 20},
        {"chunk_id": "c3", "doc_id": "2024_SHR_English", "printed_page": 220},  # retrieved, not cited
    ]
    answer = ("- Fact A [SHR 2024 — Key issues, p22]\n"
              "- Fact B [SHR 2023 — Key issues, p20]")
    used = serve_app._used_chunk_ids(answer, r, hits)
    assert used == ["c1", "c2"]                          # c3 excluded; p22 must not match p220


def test_verify_abstains_when_nothing_is_grounded(monkeypatch):
    monkeypatch.setattr("framework.eval.judge.verify_answer",
                        lambda *a, **k: [{"claim": "X", "cited_source": "s", "supporting_quote": "q",
                                          "grounded": False}])
    assert serve_app._verify({"model": None}, "q", "X.", ["p"], _FakeRetriever(), []) == serve_app.NOT_COVERED


# --- Part 3: deterministic citation verifier (the DEFAULT, 0 model calls) ---

def test_verify_citations_corrects_chapter_and_drops_unverified_quote():
    r = _FakeRetriever()                                 # doc_label: 2024_SHR_English -> "SHR 2024"
    hits = [{"doc_id": "2024_SHR_English", "printed_page": 18, "chapter": "Balochistan",
             "text": "Enforced disappearances continued in the province."}]
    answer = ('- A fact [SHR 2024 — Khyber Pakhtunkhwa, p18]\n'
              '  "Enforced disappearances continued in the province."\n\n'
              '- Another fact [SHR 2024 — Balochistan, p18]\n'
              '  "This sentence appears in no retrieved passage."')
    out = serve_app._verify_citations(answer, r, hits)
    assert "Balochistan, p18]" in out and "Khyber Pakhtunkhwa" not in out   # wrong chapter corrected
    assert "Enforced disappearances continued in the province." in out      # grounded quote kept
    assert "no retrieved passage" not in out                                # ungrounded quote dropped
    assert "A fact" in out and "Another fact" in out                        # NO fact ever dropped


def test_verify_citations_passes_not_covered_through():
    assert serve_app._verify_citations(serve_app.NOT_COVERED, _FakeRetriever(), []) == serve_app.NOT_COVERED


def test_verify_drops_irrelevant_claims(monkeypatch):
    claims = [
        {"claim": "On topic", "cited_source": "SHR 2024 — s, p1", "supporting_quote": "q",
         "grounded": True, "relevant": True},
        {"claim": "Off topic", "cited_source": "SHR 2024 — s, p1", "supporting_quote": "q",
         "grounded": True, "relevant": False},
    ]
    monkeypatch.setattr("framework.eval.judge.verify_answer", lambda *a, **k: claims)
    hits = [{"doc_id": "2024_SHR_English", "printed_page": 1, "chapter": "s"}]
    out = serve_app._verify({"model": None}, "q", "...", ["p"], _FakeRetriever(), hits)
    assert "On topic" in out and "Off topic" not in out
