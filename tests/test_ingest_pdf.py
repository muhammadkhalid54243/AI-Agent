"""Phase 1 PDF extraction. Reading-order logic is unit-tested with synthetic boxes
(no PDF); end-to-end extraction is file-gated on a real report if present."""

import os
import pytest

from framework.ingest.pdf import _reading_order, _is_two_column, extract_blocks, _norm

FFM = "docs/FFM/2026-Deaths-in-police-custody-in-Lahore.pdf"


def test_single_column_orders_top_to_bottom():
    boxes = [(50, 300, 280, 320), (50, 100, 280, 120), (50, 200, 280, 220)]
    assert _reading_order(boxes, page_width=600) == [1, 2, 0]  # by y


def test_two_column_reads_left_then_right_not_interleaved():
    # left col at x~50-280, right col at x~320-550; two rows each.
    boxes = [
        (50, 100, 280, 120),   # 0 left-top
        (320, 100, 550, 120),  # 1 right-top
        (50, 130, 280, 150),   # 2 left-bottom
        (320, 130, 550, 150),  # 3 right-bottom
    ]
    assert _is_two_column(boxes, 600)
    order = _reading_order(boxes, 600)
    # correct: whole left column, then whole right column
    assert order == [0, 2, 1, 3]
    # the naive y-then-x sort would INTERLEAVE (the bug we fix):
    naive = sorted(range(4), key=lambda i: (boxes[i][1], boxes[i][0]))
    assert naive == [0, 1, 2, 3] and naive != order


def test_single_column_not_flagged_two_column():
    boxes = [(50, 100, 280, 120), (50, 130, 280, 150)]
    assert not _is_two_column(boxes, 600)


def test_norm_collapses_page_numbers_for_furniture_detection():
    # A running header differs only by page number across pages -> same normalised key.
    assert _norm("State of Human Rights in 2024  25") == _norm("State of Human Rights in 2024  65")
    assert _norm("Chapter One") != _norm("Chapter Two")


@pytest.mark.skipif(not os.path.exists(FFM), reason="FFM corpus not present")
def test_extract_blocks_on_real_report():
    blocks, stats = extract_blocks("ffm", FFM)
    assert stats["pages_total"] == 25
    assert blocks and all(b.id.startswith("ffm:p") for b in blocks)
    joined = " ".join(b.text for b in blocks)
    assert "Amina and Anmol" in joined and "Township Police Station" in joined
    assert any(b.kind == "heading" for b in blocks)  # font heuristic found some headings
