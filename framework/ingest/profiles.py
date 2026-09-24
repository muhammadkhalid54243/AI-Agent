"""Document-type detection — a cheap probe that selects a parsing strategy.

The corpus is heterogeneous (large structured PDFs, short reports, plaintext, scans,
Urdu). A single pipeline tuned for one shape quietly mangles the others, so we detect
first and log the decision (a misdetected profile is a silent quality failure). This
is deliberately minimal for now — the six-profile table in the plan is the target;
add profiles as entries, not by editing the parser.
"""

import os

# Below this many characters per page (with large image area) → likely a scan → vision.
SCAN_CHAR_THRESHOLD = 50
# PDFs longer than this are treated as full structured documents rather than reports.
STRUCTURED_MIN_PAGES = 30


def detect_profile(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("txt", "md"):
        return "plaintext"
    if ext in ("docx", "xlsx"):
        return "office"
    if ext != "pdf":
        return "plaintext"

    import pymupdf
    from framework.ingest.pdf import _pua_ratio

    doc = pymupdf.open(path)
    n = doc.page_count
    sample = min(n, 5) or 1
    sample_text = "".join(doc[i].get_text() for i in range(sample))
    if len(sample_text) / sample < SCAN_CHAR_THRESHOLD:
        return "scanned"                 # near-empty text layer → a real scan → vision
    if _pua_ratio(sample_text) > 0.5:
        return "scanned"                 # text present but unmapped glyphs (Urdu/broken font) → vision
    return "structured_pdf" if n > STRUCTURED_MIN_PAGES else "report"
