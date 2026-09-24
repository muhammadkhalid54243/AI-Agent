"""Phase 1 IR + coverage ledger — deterministic, no API key, no PDF."""

import pytest

from framework.ingest import Block, Coverage, ParsedDoc
from framework.ingest.ir import OrphanBlocksError


def _doc():
    blocks = [
        Block(id="d:p0:b0", kind="heading", text="Key issues", page=0, order=0),
        Block(id="d:p0:b1", kind="paragraph", text="379 new cases.", page=0,
              section_path=("Key issues",), order=1),
    ]
    return ParsedDoc(doc_id="d", source_path="x.pdf", checksum="abc", profile="report",
                     blocks=blocks, coverage=Coverage(blocks_extracted=2))


def test_block_ids_and_parsed_markdown():
    doc = _doc()
    assert doc.block_ids() == {"d:p0:b0", "d:p0:b1"}
    md = doc.to_parsed_markdown()
    assert "d:p0:b1" in md and "379 new cases." in md  # block id + author's words both present


def test_coverage_clean_passes_the_gate():
    doc = _doc()
    doc.coverage.assert_complete()          # no orphans → no raise
    assert not doc.coverage.has_orphans


def test_coverage_orphan_is_a_hard_gate():
    doc = _doc()
    # A block extracted but consumed by no chunk = orphan = silent loss.
    doc.coverage.orphan_blocks = ["d:p0:b1"]
    assert doc.coverage.has_orphans
    with pytest.raises(OrphanBlocksError) as e:
        doc.coverage.assert_complete()
    assert "d:p0:b1" in str(e.value)


def test_coverage_markdown_flags_orphans():
    cov = Coverage(pages_total=10, pages_with_text=9, blocks_extracted=50,
                   orphan_blocks=["d:p3:b2"])
    md = cov.to_markdown()
    assert "ORPHAN BLOCKS: 1" in md and "MUST be 0" in md
