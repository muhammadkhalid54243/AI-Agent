"""Heading-prefix block splitting — the breadcrumb-nesting fix (offline, synthetic blocks)."""

from framework.ingest.pdf import split_heading_prefix


def _line(text, size, bbox):
    return {"spans": [{"text": text, "size": size}], "bbox": bbox}


def _blk(lines):
    xs = [b for ln in lines for b in [ln["bbox"]]]
    bbox = (min(b[0] for b in xs), min(b[1] for b in xs),
            max(b[2] for b in xs), max(b[3] for b in xs))
    return {"lines": lines, "bbox": bbox}


BODY = 9.0   # heading threshold = 9.0 * 1.15 = 10.35


def test_splits_title_merged_with_smaller_remainder():
    # The UDHR bug shape: big title line stacked above a smaller label line in ONE block.
    b = _blk([_line("Universal Declaration of Human Rights", 12.0, (0, 0, 100, 12)),
              _line("Article 1", 10.0, (0, 13, 40, 23))])
    groups = split_heading_prefix(b, BODY)
    assert len(groups) == 2
    assert groups[0][0] == "Universal Declaration of Human Rights" and groups[0][1] == 12.0
    assert groups[1][0] == "Article 1" and groups[1][1] == 10.0


def test_all_heading_lines_stay_one_group():
    # A genuine two-line heading (both lines heading-sized) must NOT split.
    b = _blk([_line("A Very Long Section", 16.0, (0, 0, 100, 16)),
              _line("Title That Wraps", 16.0, (0, 17, 90, 33))])
    groups = split_heading_prefix(b, BODY)
    assert len(groups) == 1
    assert groups[0][1] == 16.0


def test_ordinary_paragraph_stays_one_group():
    b = _blk([_line("first body line", 9.0, (0, 0, 100, 9)),
              _line("second body line", 9.0, (0, 10, 100, 19))])
    assert len(split_heading_prefix(b, BODY)) == 1


def test_single_line_block_is_never_split():
    b = _blk([_line("Lonely Heading", 16.0, (0, 0, 100, 16))])
    assert len(split_heading_prefix(b, BODY)) == 1


def test_no_body_size_falls_back_to_single_group():
    b = _blk([_line("title", 12.0, (0, 0, 50, 12)), _line("body", 9.0, (0, 13, 50, 22))])
    assert len(split_heading_prefix(b, 0.0)) == 1


def test_union_bbox_covers_the_group():
    b = _blk([_line("Title", 12.0, (5, 0, 100, 12)), _line("x", 9.0, (0, 13, 40, 23))])
    heading_bbox = split_heading_prefix(b, BODY)[0][2]
    assert heading_bbox == (5, 0, 100, 12)        # just the title line's box
