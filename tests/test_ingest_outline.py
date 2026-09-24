"""Phase 1 outline + section-path assignment, and profile detection. No network."""

import os
import pytest

from framework.ingest.ir import Block
from framework.ingest.outline import build_outline_and_sections
from framework.ingest.profiles import detect_profile


def _h(text, size, order, page=0):
    return Block(id=f"d:p{page}:b{order}", kind="heading", text=text, page=page,
                 order=order, meta={"font_size": size})


def _p(text, order, page=0):
    return Block(id=f"d:p{page}:b{order}", kind="paragraph", text=text, page=page,
                 order=order, meta={"font_size": 10})


def test_outline_levels_and_nesting():
    blocks = [
        _h("Chapter 1", 20, 0), _h("1.1 Sub", 14, 1), _p("body", 2),
        _h("Chapter 2", 20, 3), _p("more", 4),
    ]
    outline, out = build_outline_and_sections(blocks)
    levels = [(o.level, o.title) for o in outline]
    assert levels == [(1, "Chapter 1"), (2, "1.1 Sub"), (1, "Chapter 2")]


def test_section_path_stamped_and_resets_across_chapters():
    blocks = [
        _h("Chapter 1", 20, 0), _h("1.1 Sub", 14, 1), _p("body under 1.1", 2),
        _h("Chapter 2", 20, 3), _p("body under 2", 4),
    ]
    _, out = build_outline_and_sections(blocks)
    by_text = {b.text: b.section_path for b in out}
    assert by_text["body under 1.1"] == ("Chapter 1", "1.1 Sub")
    assert by_text["body under 2"] == ("Chapter 2",)          # 1.1 correctly popped
    assert by_text["1.1 Sub"] == ("Chapter 1",)               # heading gets ancestors only


def test_detect_profile_by_extension_and_size(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi", encoding="utf-8")
    assert detect_profile(str(f)) == "plaintext"


SHR = "docs/SHR/2024_SHR_English.pdf"
FFM = "docs/FFM/2026-Deaths-in-police-custody-in-Lahore.pdf"


@pytest.mark.skipif(not os.path.exists(FFM), reason="FFM corpus not present")
def test_detect_profile_report_vs_structured():
    assert detect_profile(FFM) == "report"          # 25 pages
    if os.path.exists(SHR):
        assert detect_profile(SHR) == "structured_pdf"   # 370 pages
