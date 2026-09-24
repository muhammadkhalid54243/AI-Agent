"""Phase 6 routing + persistence. Router/store are offline; no-op re-ingest is file-gated."""

import os
import time
import pytest

from framework.ingest.ir import Block, Coverage, GlossaryEntry, ParsedDoc
from framework.ingest.coverage import write_reports
from framework.ingest.store import is_ingested, save_manifest, load_parsed, file_checksum
from framework.rag.router import route, doc_tokens

FFM = "docs/FFM/2026-Deaths-in-police-custody-in-Lahore.pdf"


def _doc(n_blocks=1, words_each=3):
    blocks = [Block(id=f"d:p0:b{i}", kind="paragraph", text=" ".join(["w"] * words_each),
                    page=0, section_path=("A",)) for i in range(n_blocks)]
    return ParsedDoc(doc_id="d", source_path="x.pdf", checksum="abc", profile="report",
                     page_count=1, blocks=blocks,
                     glossary={"PECA": GlossaryEntry(expansion="Prevention of Electronic Crimes Act")},
                     coverage=Coverage(blocks_extracted=n_blocks))


def test_route_stuffs_small_retrieves_large():
    doc = _doc(n_blocks=3)
    assert route(doc, threshold=1000) == "stuff"        # fits -> whole doc in context
    assert route(doc, threshold=1) == "retrieve"        # too big -> chunk + search
    assert doc_tokens(doc) > 0


def test_manifest_matches_only_same_checksum(tmp_path):
    doc = _doc()
    save_manifest(doc, str(tmp_path))
    write_reports(doc, str(tmp_path))                   # provides parsed.json
    assert is_ingested(str(tmp_path), "abc")            # same checksum
    assert not is_ingested(str(tmp_path), "different")  # changed file


def test_parsed_doc_round_trips(tmp_path):
    doc = _doc()
    write_reports(doc, str(tmp_path))
    loaded = load_parsed(str(tmp_path))
    assert loaded.doc_id == "d" and loaded.blocks[0].section_path == ("A",)
    assert loaded.glossary["PECA"].expansion == "Prevention of Electronic Crimes Act"
    assert loaded.coverage.blocks_extracted == 1


@pytest.mark.skipif(not os.path.exists(FFM), reason="FFM corpus not present")
def test_reingest_unchanged_is_a_fast_noop(tmp_path):
    from framework.ingest import ingest
    t0 = time.perf_counter(); d1 = ingest(FFM, out_dir=str(tmp_path)); parse_s = time.perf_counter() - t0
    t0 = time.perf_counter(); d2 = ingest(FFM, out_dir=str(tmp_path)); cache_s = time.perf_counter() - t0
    assert d2.doc_id == d1.doc_id and len(d2.blocks) == len(d1.blocks)
    assert cache_s < parse_s and cache_s < 1.0          # no-op load, sub-second
    # force bypasses the cache and re-parses to the same result
    d3 = ingest(FFM, out_dir=str(tmp_path), force=True)
    assert len(d3.blocks) == len(d1.blocks)
