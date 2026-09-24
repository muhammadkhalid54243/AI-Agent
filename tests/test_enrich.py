"""Phase 3 enrichment — deterministic, no API key."""

from framework.rag.chunker import Chunk
from framework.ingest.ir import GlossaryEntry
from framework.rag.enrich import enrich_chunks, contextual_prefixes


def _chunk(text, section=("A", "B"), kind="prose", cid="c0"):
    return Chunk(id=cid, text=text, embed_text=text, doc_id="d", page_start=0, page_end=0,
                 section_path=section, kind=kind)


GLOSS = {"PECA": GlossaryEntry(expansion="Prevention of Electronic Crimes Act"),
         "HRCP": GlossaryEntry(expansion="Human Rights Commission of Pakistan")}


def test_breadcrumb_and_glossary_enrich_embed_text_only():
    c = _chunk("Charges were filed under PECA in 2024.")
    enrich_chunks([c], glossary=GLOSS)
    assert "A > B" in c.embed_text                                   # breadcrumb
    assert "PECA (Prevention of Electronic Crimes Act)" in c.embed_text  # glossary
    assert "HRCP" not in c.embed_text                                # not mentioned -> not added
    assert c.text == "Charges were filed under PECA in 2024."        # text UNTOUCHED


def test_tables_skip_glossary():
    c = _chunk("|PECA|count|\n|---|---|\n|x|3|", kind="table")
    enrich_chunks([c], glossary=GLOSS)
    assert "Prevention of Electronic Crimes Act" not in c.embed_text


def test_enrichment_is_idempotent():
    c = _chunk("Under PECA.")
    enrich_chunks([c], glossary=GLOSS)
    first = c.embed_text
    enrich_chunks([c], glossary=GLOSS)      # rebuilt from text, not appended twice
    assert c.embed_text == first


def test_ubiquitous_acronym_is_not_expanded():
    # HRCP appears in many chunks -> non-discriminative -> must NOT be expanded;
    # PECA is rare -> expanded. (The measured HRCP regression, fixed.)
    chunks = [_chunk(f"HRCP acted here.", cid=f"h{i}") for i in range(10)]
    chunks.append(_chunk("Charged under PECA.", cid="peca"))
    enrich_chunks(chunks, glossary=GLOSS, max_glossary_freq=8)
    assert all("Human Rights Commission of Pakistan)" not in c.embed_text for c in chunks)
    peca = next(c for c in chunks if c.id == "peca")
    assert "PECA (Prevention of Electronic Crimes Act)" in peca.embed_text


def test_contextual_prefix_caps_and_caches():
    chunks = [_chunk("t", cid=f"c{i}") for i in range(5)]
    calls = {"n": 0}

    def fake(prompt):
        calls["n"] += 1
        return "This chunk covers X."

    cache = {}
    out = contextual_prefixes(chunks, "DOC", summarizer=fake, max_calls=3, cache=cache)
    assert calls["n"] == 3 and len(out) == 3           # hard cap honored
    contextual_prefixes(chunks[:3], "DOC", summarizer=fake, max_calls=3, cache=cache)
    assert calls["n"] == 3                              # cache hit -> no new calls
