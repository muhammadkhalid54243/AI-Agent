"""SHR structural analysis — TOC parsing, chapter matching, chapter assignment (offline)."""

from framework.ingest.shr import (
    _toc_ranges, assign_chapters, chapter_aliases, chapter_name, extract_year,
    match_chapter, parse_toc,
)


def test_extract_year():
    assert extract_year("State of Human Rights in 2019") == 2019
    assert extract_year("Human Rights Commission of Pakistan  State of Human Rights  in 2024") == 2024
    assert extract_year("no year here") is None


TOC_LINES = [
    (62, "Note on sources"), (62, "Abbreviations"), (62, "Glossary"),
    (62, "Introduction"), (324, "… 1"),
    (62, "Key issues"), (324, "… 7"),
    (62, "Punjab"), (324, "… 15"),
    (62, "Balochistan"), (324, "… 161"),
    (62, "Islamabad Capital Territory"), (324, "… 203"),
    (62, "Azad Jammu and Kashmir"), (324, "… 241"),
    (62, "Gilgit-Baltistan"), (324, "… 227"),          # corrupt in the text layer (should be 251)
    (62, "Appendix 6"), (151, "HRCP"), (151, "stands"), (324, "… 315"),
]


def test_parse_toc_front_matter_body_and_appendix():
    e = parse_toc(TOC_LINES)
    by = {x["title"]: x for x in e}
    assert by["Note on sources"]["page"] is None       # front matter: no page
    assert by["Introduction"]["page"] == 1
    assert by["Punjab"]["page"] == 15
    assert by["Balochistan"]["page"] == 161
    assert by["Appendix 6"]["subtitle"] == "HRCP stands"   # indented description joined
    assert chapter_name(by["Appendix 6"]) == "HRCP stands"  # appendix → description is the name


def test_chapter_aliases_and_matching():
    amap = chapter_aliases(parse_toc(TOC_LINES))
    assert match_chapter("BALOCHISTAN", amap) == "Balochistan"
    assert match_chapter("PUNJAB", amap) == "Punjab"
    assert match_chapter("Gilgit-Baltistan", amap) == "Gilgit-Baltistan"
    assert match_chapter("totally unrelated", amap) is None


def test_toc_ranges_drop_non_monotonic():
    ranges = _toc_ranges(parse_toc(TOC_LINES))
    names = [r[0] for r in ranges]
    assert "Gilgit-Baltistan" not in names             # 227 < 241 (AJK) → non-monotonic, dropped
    # Balochistan range starts at 161 and runs until the next monotonic entry (ICT, 203).
    bal = next(r for r in ranges if r[0] == "Balochistan")
    assert bal[1] == 161 and bal[2] == 202


def test_assign_chapters_margin_then_range_then_carry():
    toc = parse_toc(TOC_LINES)
    amap = chapter_aliases(toc)
    vmargin = {10: "PUNJAB", 11: "BALOCHISTAN"}
    printed = {10: 20, 11: 165, 12: 166}               # page 12 has no margin, only a printed no.
    ch = assign_chapters(vmargin, toc, printed, n_pages=14, alias_map=amap)
    assert ch[10] == "Punjab"                          # margin (primary)
    assert ch[11] == "Balochistan"                     # margin
    assert ch[12] == "Balochistan"                     # TOC range (printed 166 ∈ 161..314)
    assert ch[13] == "Balochistan"                     # carry-forward (no margin, no printed)
    assert 0 not in ch                                 # nothing before the first known chapter
