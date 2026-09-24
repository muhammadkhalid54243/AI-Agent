"""Phase 1 glossary harvest. Unit tests are synthetic; real check is file-gated."""

import os
import pytest

from framework.ingest.ir import Block
from framework.ingest.glossary import harvest_glossary, _plausible


def _block(text):
    return Block(id="d:p0:b0", kind="paragraph", text=text, page=0)


def test_captures_inline_definition():
    g = harvest_glossary([_block("...under the Prevention of Electronic Crimes Act (PECA) 2016.")])
    assert "PECA" in g
    assert g["PECA"].expansion == "Prevention of Electronic Crimes Act"


def test_initialism_check_rejects_false_positive():
    # "(XYZ)" is not an initialism of the preceding words -> not a glossary entry.
    assert not _plausible("Global Report Aggregation", "XYZ")
    assert _plausible("Human Rights Commission of Pakistan", "HRCP")
    g = harvest_glossary([_block("The Global Report Aggregation (XYZ) was noted.")])
    assert "XYZ" not in g


def test_first_definition_wins_and_records_page():
    blocks = [Block(id="d:p3:b0", kind="paragraph", page=3,
                    text="the Federal Investigation Agency (FIA) said")]
    g = harvest_glossary(blocks)
    assert g["FIA"].expansion == "Federal Investigation Agency" and g["FIA"].source_page == 3


FFM = "docs/FFM/2026-Deaths-in-police-custody-in-Lahore.pdf"


@pytest.mark.skipif(not os.path.exists(FFM), reason="FFM corpus not present")
def test_glossary_on_real_report():
    from framework.ingest import ingest
    doc = ingest(FFM, write=False)
    assert "HRCP" in doc.glossary
    assert "Human Rights Commission of Pakistan" in doc.glossary["HRCP"].expansion
