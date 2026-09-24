"""Phase 2 structure-aware chunking — deterministic, no API key, no PDF."""

import os
import pytest

from framework.ingest.ir import Block, Coverage, ParsedDoc
from framework.rag.chunker import structure_aware_chunks, leaf_chunks, Chunk
from framework.context.budget import count_tokens

FFM = "docs/FFM/2026-Deaths-in-police-custody-in-Lahore.pdf"


def _doc(blocks):
    return ParsedDoc(doc_id="d", source_path="x.pdf", checksum="c", profile="report",
                     blocks=blocks, coverage=Coverage(blocks_extracted=len(blocks)))


def _blk(bid, text, kind="paragraph", section=(), page=0):
    return Block(id=bid, kind=kind, text=text, page=page, section_path=section)


def _words(n):
    return " ".join(["tok"] * n)


def test_children_respect_max_and_stay_in_section():
    blocks = [
        _blk("d:p0:b0", _words(120), section=("A",)),
        _blk("d:p0:b1", _words(120), section=("A",)),
        _blk("d:p0:b2", _words(120), section=("A",)),
        _blk("d:p1:b3", _words(40), section=("B",)),
    ]
    chunks = structure_aware_chunks(_doc(blocks), child_target=200, child_max=320)
    leaves = leaf_chunks(chunks)
    # every indexed leaf stays under the hard max and inside ONE section
    for c in leaves:
        assert count_tokens(c.text) <= 320
        assert c.section_path in {("A",), ("B",)}
    # no leaf mixes sections A and B
    assert all("B" not in c.section_path for c in leaves if c.section_path == ("A",))


def test_no_orphans_gate_passes():
    blocks = [_blk(f"d:p0:b{i}", _words(80), section=("A",)) for i in range(4)]
    doc = _doc(blocks)
    structure_aware_chunks(doc)
    assert doc.coverage.orphan_blocks == []
    doc.coverage.assert_complete()          # every block landed in a chunk


def test_table_is_atomic_not_merged():
    blocks = [
        _blk("d:p0:b0", "Intro prose.", section=("A",)),
        _blk("d:p0:b1", "|H1|H2|\n|---|---|\n|1|2|", kind="table", section=("A",)),
        _blk("d:p0:b2", "More prose.", section=("A",)),
    ]
    chunks = structure_aware_chunks(_doc(blocks))
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    assert tables[0].text.startswith("|H1|H2|") and tables[0].block_ids == ["d:p0:b1"]
    # the table text is never folded into a prose chunk
    assert all("|H1|H2|" not in c.text for c in chunks if c.kind == "prose")


def test_oversized_table_splits_by_rows_and_counts():
    rows = "\n".join(f"|{i}|{_words(20)}|" for i in range(200))
    md = "|id|val|\n|---|---|\n" + rows
    blocks = [_blk("d:p0:b0", md, kind="table", section=("A",))]
    doc = _doc(blocks)
    chunks = structure_aware_chunks(doc, child_max=320)
    table_chunks = [c for c in chunks if c.kind == "table"]
    assert len(table_chunks) > 1                       # was split
    assert doc.coverage.tables_split == 1
    assert all(c.text.startswith("|id|val|") for c in table_chunks)  # header repeated
    assert doc.coverage.orphan_blocks == []            # the one block still covered


def test_children_link_to_a_real_parent():
    blocks = [_blk(f"d:p0:b{i}", _words(150), section=("A",)) for i in range(3)]
    chunks = structure_aware_chunks(_doc(blocks), child_target=200, child_max=320)
    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]
    assert children and parents
    parent_ids = {c.id for c in parents}
    assert all(c.parent_id in parent_ids for c in children)
    assert parents[0] not in leaf_chunks(chunks)       # the big parent is not indexed


@pytest.mark.skipif(not os.path.exists(FFM), reason="FFM corpus not present")
def test_structure_aware_chunks_on_real_doc():
    from framework.ingest import ingest
    doc = ingest(FFM, write=False)
    chunks = structure_aware_chunks(doc)
    assert chunks and doc.coverage.orphan_blocks == []   # Block->Chunk join is complete
    doc.coverage.assert_complete()
    leaves = leaf_chunks(chunks)
    assert any(c.section_path for c in leaves)            # structure carried through
    assert all(count_tokens(c.text) <= 320 for c in leaves if c.kind == "prose")
