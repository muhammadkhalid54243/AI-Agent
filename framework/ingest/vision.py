"""Vision/OCR route — for pages whose text layer is garbage (unmapped-glyph fonts,
scans, Urdu Nastaliq). The RENDERED page is perfect even when extraction returns
glyph IDs, so we rasterize and ask a vision model to transcribe it.

This is the SEPARATE route from the solid English text path: English PDFs never
touch a model here; only pages the ledger flags as garbled/scanned do. Costs money
per page, so it is hard-capped and the transcriber is injectable (tests pass a fake,
no API key). Language is preserved — Gemini transcribes Urdu natively.
"""

import os

from framework.ingest.ir import Block

VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.6-flash")
MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "20"))     # cost cap (CLAUDE.md)
TRANSCRIBE_PROMPT = (
    "Transcribe this page to Markdown exactly. Preserve headings (use #), lists, and "
    "tables. Keep the ORIGINAL language and script (e.g. Urdu) — do not translate. Do "
    "not summarize or add commentary. Output only the transcription."
)


def gemini_transcriber(png_bytes: bytes, *, max_retries: int = 5) -> str:
    from google import genai
    from google.genai import types
    import time

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=VISION_MODEL,
                contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                          TRANSCRIBE_PROMPT],
            )
            return (resp.text or "").strip()
        except Exception as exc:
            s = str(exc).lower()
            if attempt < max_retries and ("429" in s or "resource_exhausted" in s or "rate" in s):
                time.sleep(min(2 ** attempt, 30))         # backoff on transient rate limit
                continue
            raise


def transcribe_pdf(doc_id: str, path: str, *, transcriber=None, max_pages: int = MAX_PAGES,
                   dpi: int = 150, page_indices=None) -> tuple:
    """Rasterize pages and transcribe them into Blocks. Returns (blocks, stats).

    A page that raises during transcription is recorded in pages_failed and skipped
    — a vision error degrades that page, never aborts the ingest.
    """
    import pymupdf

    transcriber = transcriber or gemini_transcriber
    doc = pymupdf.open(path)
    pages = list(page_indices) if page_indices is not None else list(range(doc.page_count))
    pages = [p for p in pages if 0 <= p < doc.page_count][:max_pages]

    blocks, order, via_vision = [], 0, 0
    pages_failed = []
    for pno in pages:
        png = doc[pno].get_pixmap(dpi=dpi).tobytes("png")
        try:
            md = transcriber(png)
        except Exception:
            pages_failed.append(pno)
            continue
        via_vision += 1
        for seg in (s.strip() for s in md.split("\n\n")):
            if not seg:
                continue
            is_heading = seg.lstrip().startswith("#")
            text = seg.lstrip("# ").strip() if is_heading else seg
            blocks.append(Block(
                id=f"{doc_id}:p{pno}:v{order}", kind="heading" if is_heading else "paragraph",
                text=text, page=pno, order=order, confidence=0.85, meta={"via": "vision"},
            ))
            order += 1

    stats = {"pages_total": doc.page_count, "pages_with_text": via_vision,
             "pages_failed": pages_failed, "pages_low_quality": [], "pages_via_vision": via_vision,
             "tables_found": 0, "furniture_removed": 0, "body_size": 0.0,
             "capped": len(pages) < (len(page_indices) if page_indices is not None else doc.page_count)}
    return blocks, stats
