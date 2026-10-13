"""Phase C — printed-page offset fit (offline, no PDF)."""

from framework.ingest.pdf import _fit_printed_offset, _ints


def test_ints_extracts_digit_runs():
    assert _ints("307State of Human Rights in 2024") == [307, 2024]
    assert _ints("no digits here") == []


def test_offset_picks_incrementing_page_over_constant_year():
    # Each header carries the page number (increments with the page) AND the year
    # (constant). Realistic: printed 307 sits near PDF index 312 (a few front-matter pages).
    header_nums = {312: [307, 2024], 313: [308, 2024], 314: [309, 2024], 315: [310, 2024]}
    printed = _fit_printed_offset(header_nums)
    assert printed[312] == 307 and printed[315] == 310   # page number (offset -5) wins
    assert printed[314] == 309                            # the year is rejected by magnitude


def test_front_matter_before_body_start_is_unmapped():
    # Front-matter pages 0-1 show only the year; body numbering starts at PDF page 6.
    header_nums = {0: [2024], 1: [2024], 6: [1, 2024], 7: [2, 2024], 8: [3, 2024]}
    printed = _fit_printed_offset(header_nums)
    assert 0 not in printed and 1 not in printed          # front matter -> citation fallback
    assert printed[6] == 1 and printed[8] == 3            # body numbered from PDF page 6


def test_empty_returns_empty():
    assert _fit_printed_offset({}) == {}
