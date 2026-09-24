"""PyMuPDF text extraction into ordered `Block`s.

The load-bearing part is READING ORDER. A page's text blocks come out of the parser
in an arbitrary order; a naive top-to-bottom (y) sort INTERLEAVES a two-column layout
— it reads the first line of the left column, then the first line of the right, then
back — producing fluent-looking nonsense that no downstream stage can recover. So we
detect columns by x-position and sort WITHIN a column before moving right. This failure
is invisible in the text itself, which is exactly why `parsed.md` exists to eyeball.
"""

import re

from framework.ingest.ir import Block
from framework.ingest.tables import extract_page_tables, find_caption


def _center(bbox: tuple) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _inside_any(center: tuple, boxes: list) -> bool:
    x, y = center
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)

# A block is a heading if its largest font is this much bigger than the body size.
HEADING_RATIO = 1.15
# Text (digits stripped) repeating on at least this many pages is page furniture
# (running header/footer) — dropped so it pollutes neither the outline nor chunks.
FURNITURE_MIN_PAGES = 5


def _norm(text: str) -> str:
    """Normalise for repetition detection: lowercase, drop digits (page numbers vary)."""
    return re.sub(r"\d+", "", text).lower().strip()


# Fraction of a block's non-space chars that fall in the Unicode Private Use Area
# (U+E000–U+F8FF). High = the font has no ToUnicode map and we got glyph IDs, not
# text (broken-font / Urdu Nastaliq PDFs). This is the garbage the ledger must catch.
GARBLE_THRESHOLD = 0.5


def _pua_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if 0xE000 <= ord(c) <= 0xF8FF) / len(chars)


def _block_text(bdict: dict) -> str:
    """Reconstruct a block's text: spans joined within a line, lines joined by space."""
    lines = []
    for line in bdict.get("lines", []):
        lines.append("".join(s.get("text", "") for s in line.get("spans", [])))
    return " ".join(t.strip() for t in lines if t.strip()).strip()


def _block_size(bdict: dict) -> float:
    sizes = [s.get("size", 0) for line in bdict.get("lines", []) for s in line.get("spans", [])]
    return max(sizes) if sizes else 0.0


def _is_two_column(boxes: list, page_width: float) -> bool:
    """True if blocks clearly populate both halves of the page (a two-column layout)."""
    mid = page_width / 2
    left = sum(1 for b in boxes if b[2] < mid)    # entirely left of centre (x1 < mid)
    right = sum(1 for b in boxes if b[0] > mid)   # entirely right of centre (x0 > mid)
    return left >= 2 and right >= 2


def _reading_order(boxes: list, page_width: float) -> list:
    """Return indices of `boxes` in human reading order. boxes = [(x0,y0,x1,y1), ...]."""
    idx = list(range(len(boxes)))
    if _is_two_column(boxes, page_width):
        mid = page_width / 2
        # column 0 (left) fully before column 1 (right); top-to-bottom within each.
        return sorted(idx, key=lambda i: (0 if (boxes[i][0] + boxes[i][2]) / 2 < mid else 1,
                                          boxes[i][1]))
    return sorted(idx, key=lambda i: (boxes[i][1], boxes[i][0]))  # single column: y then x


def extract_blocks(doc_id: str, path: str) -> tuple:
    """Parse `path` into ordered Blocks. Returns (blocks, page_stats).

    section_path is left empty here — outline.py assigns it. A page that raises is
    recorded in page_stats['pages_failed'] and skipped, never aborting the parse.
    """
    import pymupdf

    doc = pymupdf.open(path)
    # Doc-level body font size = the most common span size (robust heading threshold).
    from collections import Counter
    sizes = Counter()
    per_page = []
    for pno in range(doc.page_count):
        try:
            d = doc[pno].get_text("dict")
        except Exception:
            per_page.append(None)
            continue
        per_page.append(d)
        for b in d.get("blocks", []):
            for line in b.get("lines", []):
                for s in line.get("spans", []):
                    sizes[round(s.get("size", 0), 1)] += len(s.get("text", ""))
    body_size = sizes.most_common(1)[0][0] if sizes else 0.0

    # Pass 1: per page, detect tables (atomic) + prose blocks NOT inside a table,
    # interleave in reading order, and count text repetition for furniture detection.
    candidates = []           # dicts: {page, idx, kind, text, size, bbox, caption?}
    pages_failed = []
    pages_low_quality = []
    pages_with_text = 0
    norm_pages = {}
    tables_found = 0
    for pno, d in enumerate(per_page):
        if d is None:
            pages_failed.append(pno)
            continue
        page = doc[pno]
        tables = extract_page_tables(page)
        tboxes = [t["bbox"] for t in tables]
        raw = [b for b in d.get("blocks", []) if b.get("type") == 0 and _block_text(b)]

        page_cands = []
        for j, b in enumerate(raw):
            if _inside_any(_center(b["bbox"]), tboxes):     # part of a table → dedup out
                continue
            page_cands.append({"idx": str(j), "kind": None, "text": _block_text(b),
                               "size": _block_size(b), "bbox": b["bbox"]})
        text_pairs = [(_block_text(b), b["bbox"]) for b in raw]
        for k, t in enumerate(tables):
            page_cands.append({"idx": f"t{k}", "kind": "table", "text": t["markdown"],
                               "size": body_size, "bbox": t["bbox"],
                               "caption": find_caption(text_pairs, t["bbox"])})
        if not page_cands:
            continue
        pages_with_text += 1
        tables_found += len(tables)
        width = d.get("width") or page.rect.width
        boxes = [c["bbox"] for c in page_cands]
        page_pua = page_chars = 0
        for oi in _reading_order(boxes, width):
            c = dict(page_cands[oi], page=pno)
            candidates.append(c)
            if c["kind"] is None:
                norm_pages.setdefault(_norm(c["text"]), set()).add(pno)
                ns = [ch for ch in c["text"] if not ch.isspace()]
                page_chars += len(ns)
                page_pua += sum(1 for ch in ns if 0xE000 <= ord(ch) <= 0xF8FF)
        if page_chars and page_pua / page_chars > GARBLE_THRESHOLD:
            pages_low_quality.append(pno)

    furniture = {t for t, pages in norm_pages.items() if len(pages) >= FURNITURE_MIN_PAGES}

    # Pass 2: drop furniture, classify prose, keep tables atomic, assign reading order.
    blocks = []
    order = 0
    for c in candidates:
        if c["kind"] is None and _norm(c["text"]) in furniture:
            continue
        meta = {"font_size": round(c["size"], 1), "bbox": c["bbox"]}
        if c["kind"] == "table":
            kind = "table"
            if c.get("caption"):
                meta["caption"] = c["caption"]
        else:
            kind = "heading" if body_size and c["size"] >= body_size * HEADING_RATIO else "paragraph"
        conf = round(1.0 - _pua_ratio(c["text"]), 2)
        blocks.append(Block(id=f"{doc_id}:p{c['page']}:b{c['idx']}", kind=kind,
                            text=c["text"], page=c["page"], order=order,
                            confidence=conf, meta=meta))
        order += 1

    page_stats = {"pages_total": doc.page_count, "pages_with_text": pages_with_text,
                  "pages_failed": pages_failed, "pages_low_quality": pages_low_quality,
                  "body_size": body_size, "furniture_removed": len(furniture),
                  "tables_found": tables_found}
    return blocks, page_stats
