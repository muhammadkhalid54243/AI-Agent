"""Phase 1 vision route + garble detection. Offline via a fake transcriber and a
tiny generated PDF; the real Urdu→scanned routing is file-gated."""

import os
import pytest

from framework.ingest.pdf import _pua_ratio


def _fake_transcriber(png_bytes):
    return "# Section Title\n\nFirst paragraph of body text.\n\nSecond paragraph."


def _tiny_pdf(tmp_path, text="x"):
    import pymupdf
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), text)
    f = str(tmp_path / "tiny.pdf")
    d.save(f)
    return f


def test_pua_ratio_flags_glyph_garbage_not_real_text():
    assert _pua_ratio("normal readable text") == 0.0
    assert _pua_ratio(" ") > 0.9      # private-use glyphs


def test_vision_route_builds_blocks_from_transcription(tmp_path):
    from framework.ingest.vision import transcribe_pdf
    blocks, stats = transcribe_pdf("t", _tiny_pdf(tmp_path), transcriber=_fake_transcriber,
                                   max_pages=1)
    assert stats["pages_via_vision"] == 1
    kinds = {b.kind for b in blocks}
    assert "heading" in kinds and "paragraph" in kinds
    assert any("First paragraph" in b.text for b in blocks)
    assert all(b.meta.get("via") == "vision" for b in blocks)


def test_transcription_error_degrades_not_aborts(tmp_path):
    def boom(png):
        raise RuntimeError("vision service down")
    from framework.ingest.vision import transcribe_pdf
    blocks, stats = transcribe_pdf("t", _tiny_pdf(tmp_path), transcriber=boom, max_pages=1)
    assert blocks == [] and stats["pages_failed"] == [0]


def test_ingest_routes_scanned_profile_to_vision(tmp_path):
    # A near-empty text layer detects as "scanned" → the vision route (fake transcriber).
    from framework.ingest import ingest
    doc = ingest(_tiny_pdf(tmp_path), write=False, transcriber=_fake_transcriber)
    assert doc.profile == "scanned"
    assert doc.coverage.pages_via_vision == 1
    assert any(b.text for b in doc.blocks)


UR = "docs/FFM-UR/2025-Unequal-claims-UR.pdf"
EN = "docs/FFM/2026-Deaths-in-police-custody-in-Lahore.pdf"


@pytest.mark.skipif(not os.path.exists(UR), reason="Urdu corpus not present")
def test_urdu_detects_scanned_english_stays_report():
    from framework.ingest.profiles import detect_profile
    assert detect_profile(UR) == "scanned"      # unmapped-glyph Urdu → vision route
    if os.path.exists(EN):
        assert detect_profile(EN) == "report"   # solid English text path unaffected
