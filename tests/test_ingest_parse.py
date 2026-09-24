"""Phase 1 end-to-end ingest — file-gated on the real FFM report (no API key)."""

import os
import pytest

from framework.ingest import ingest

FFM = "docs/FFM/2026-Deaths-in-police-custody-in-Lahore.pdf"


@pytest.mark.skipif(not os.path.exists(FFM), reason="FFM corpus not present")
def test_ingest_ffm_end_to_end(tmp_path):
    doc = ingest(FFM, out_dir=str(tmp_path))

    # ParsedDoc is populated and self-consistent.
    assert doc.profile == "report" and doc.page_count == 25
    assert len(doc.checksum) == 64                       # sha256 hex
    assert doc.blocks and doc.outline
    joined = " ".join(b.text for b in doc.blocks)
    assert "Amina and Anmol" in joined

    # Coverage ledger is populated; no orphans at parse time (chunking is Phase 2).
    assert doc.coverage.blocks_extracted == len(doc.blocks)
    assert doc.coverage.pages_total == 25
    assert not doc.coverage.has_orphans
    doc.coverage.assert_complete()                       # passes the gate

    # Reports written for a human to eyeball.
    for name in ("parsed.md", "coverage.md", "parsed.json"):
        p = tmp_path / name
        assert p.exists() and p.stat().st_size > 0

    # Some block carries a real section path (outline stamping worked).
    assert any(b.section_path for b in doc.blocks)
