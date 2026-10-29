"""ingest() — the orchestrator: source file → ParsedDoc (+ optional on-disk reports).

detect profile → extract ordered blocks → build outline & stamp section paths →
assemble the coverage ledger → (optionally) write parsed.md / coverage.md / parsed.json.

Coverage's `orphan_blocks` stays empty here by design: an orphan is a block no CHUNK
consumed, and chunking is Phase 2. This populates the extraction-side ledger; the
orphan gate runs once chunks exist and join back on block ids.
"""

import hashlib
import os
import re

from framework.ingest.glossary import harvest_glossary
from framework.ingest.ir import Coverage, ParsedDoc
from framework.ingest.outline import build_outline_and_sections
from framework.ingest.pdf import extract_blocks
from framework.ingest.profiles import detect_profile

LOW_CONFIDENCE = 0.5
_TEXT_PDF_PROFILES = {"structured_pdf", "report", "publication"}  # solid text-layer route


def _doc_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def _checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(path: str, *, out_dir: str | None = None, write: bool = True, transcriber=None,
           cache: bool = True, force: bool = False, cache_dir: str = ".rag_cache") -> ParsedDoc:
    doc_id = _doc_id(path)
    doc_dir = out_dir or os.path.join(cache_dir, doc_id)

    # No-op fast path: an unchanged file (same checksum) loads from cache, no re-parse.
    if cache and not force:
        from framework.ingest.store import is_ingested, load_parsed
        checksum = _checksum(path)
        if is_ingested(doc_dir, checksum):
            cached = load_parsed(doc_dir)
            if cached is not None:
                return cached

    profile = detect_profile(path)

    if profile == "scanned":
        # SEPARATE ROUTE — garbled/scanned/Urdu docs: the text layer is unusable, so
        # transcribe the rendered pages with a vision model. English never comes here.
        from framework.ingest.vision import transcribe_pdf
        blocks, stats = transcribe_pdf(doc_id, path, transcriber=transcriber)
    elif profile in _TEXT_PDF_PROFILES:
        blocks, stats = extract_blocks(doc_id, path)
    else:
        raise NotImplementedError(
            f"profile {profile!r} not handled yet (office/plaintext land in a later step)")

    # SHR structural overlay: when this is an SHR annual, the CHAPTER (TOC + vertical margin)
    # becomes the authoritative top of section_path. Other docs fall back to the generic font path.
    from framework.ingest import shr
    chapter_by_page = None
    shr_year = None
    if profile in _TEXT_PDF_PROFILES:
        try:
            fm = shr.front_matter(path)
            if shr.is_shr_annual(fm, stats.get("vmargin_by_page", {})):
                shr_year = fm.get("year")
                chapter_by_page = shr.assign_chapters(
                    stats.get("vmargin_by_page", {}), fm["toc_entries"],
                    stats.get("printed_by_page", {}), stats["pages_total"],
                    shr.chapter_aliases(fm["toc_entries"]))
        except Exception:
            chapter_by_page = None            # never let structural analysis break ingestion

    outline, blocks = build_outline_and_sections(blocks, chapter_by_page=chapter_by_page)
    glossary = harvest_glossary(blocks)

    coverage = Coverage(
        pages_total=stats["pages_total"],
        pages_with_text=stats["pages_with_text"],
        pages_failed=stats["pages_failed"],
        blocks_extracted=len(blocks),
        blocks_low_confidence=[b.id for b in blocks if b.confidence < LOW_CONFIDENCE],
        tables_found=stats.get("tables_found", 0),
        pages_low_quality=stats.get("pages_low_quality", []),
        pages_via_vision=stats.get("pages_via_vision", 0),
        pages_printed_recovered=stats.get("pages_printed_recovered", 0),
        pages_printed_fallback=stats.get("pages_printed_fallback", []),
        glossary_terms=len(glossary),
    )
    if stats["pages_failed"]:
        coverage.warnings.append(f"{len(stats['pages_failed'])} page(s) failed extraction")
    if coverage.pages_low_quality:
        coverage.warnings.append(
            f"{len(coverage.pages_low_quality)} page(s) are garbled (unmapped-glyph fonts) — "
            "text extraction unreliable; route to vision/OCR")

    title = next((b.text for b in blocks if b.kind == "heading"), doc_id)
    doc = ParsedDoc(
        doc_id=doc_id, source_path=path, checksum=_checksum(path), profile=profile,
        title=title, page_count=stats["pages_total"], outline=outline, blocks=blocks,
        glossary=glossary, coverage=coverage,
    )

    if write:
        from framework.ingest.coverage import write_reports
        from framework.ingest.store import save_manifest
        write_reports(doc, doc_dir)
        save_manifest(doc, doc_dir)          # checksum manifest → future no-op re-ingest
    return doc
