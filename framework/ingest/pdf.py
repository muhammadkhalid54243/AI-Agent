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


def _ints(text: str) -> list:
    """Integer digit-runs in `text` — page-number / year candidates from furniture."""
    return [int(m) for m in re.findall(r"\d+", text)]


def _fit_printed_offset(header_nums: dict) -> dict:
    """PDF page index -> PRINTED page number, recovered from furniture (headers/footers).

    Headers carry BOTH a page number (near the pdf index) and a constant year (~2000
    away). We first find the modal offset `printed - pdf_index` to tell them apart, then
    assign EACH page the header number closest to that offset — per-page, so a mid-document
    offset DRIFT (blank/full-page images inserted in the PDF but not in the printed
    numbering) still yields each page's own true number instead of one contiguous band.
    Pages whose header held no page number (year-only, blanks, roman front matter) are
    absent from the result, so their citation falls back to the PDF index.
    """
    from collections import Counter
    # Candidate offsets per page, with the year rejected by magnitude.
    cand = {pg: [n - pg for n in nums if abs(n - pg) <= 100] for pg, nums in header_nums.items()}
    if not any(cand.values()):
        return {}
    pages = sorted(cand)
    window = 6                               # pages each side for the LOCAL offset vote
    printed = {}
    for pg in pages:
        local = Counter(o for q in pages if abs(q - pg) <= window for o in cand[q])
        if not local:
            continue
        best_off, support = local.most_common(1)[0]
        if support < 2:                      # need agreement from neighbours, not one stray number
            continue
        for n in header_nums[pg]:            # assign this page's OWN number matching the local offset
            if n - pg == best_off:
                printed[pg] = n
                break
    return printed


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


def _line_text(line: dict) -> str:
    return "".join(s.get("text", "") for s in line.get("spans", [])).strip()


def _line_size(line: dict) -> float:
    return max((s.get("size", 0) for s in line.get("spans", [])), default=0.0)


def _union_bbox(boxes: list) -> tuple:
    boxes = [b for b in boxes if b]
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _is_vertical(bdict: dict) -> bool:
    """A block whose text is written vertically (rotated 90°) — in the SHR reports this is the
    chapter name printed down the side margin. PyMuPDF marks each line's writing direction in
    `dir`: horizontal is (1,0), vertical ≈ (0,±1)."""
    for ln in bdict.get("lines", []):
        if ln.get("spans"):
            dx, _ = ln.get("dir", (1, 0))
            return abs(dx) < 0.5
    return False


def split_heading_prefix(bdict: dict, body_size: float, ratio: float = HEADING_RATIO) -> list:
    """Split a text block at the boundary between a LEADING run of heading-sized lines and
    a smaller remainder. Returns [(text, size, bbox), ...].

    A PyMuPDF text block sometimes groups a heading-sized title line with the smaller lines
    that follow it (e.g. a section title stacked above its first label/paragraph). Left whole,
    `_block_text` concatenates both into one wrong heading and `_block_size` (the max) marks the
    whole thing a heading — which is exactly how the UDHR appendix became a section misnamed
    "Universal Declaration of Human Rights Article 1". Splitting off the heading-sized prefix
    gives the title its own heading and keeps the remainder as body — no text dropped.

    Only the bug shape splits: a block whose first line(s) are heading-sized AND which has
    smaller lines after. All-heading blocks (genuine multi-line headings) and blocks with no
    leading heading line (ordinary paragraphs) return a single group — today's behaviour.
    """
    lines = [ln for ln in bdict.get("lines", []) if _line_text(ln)]
    if len(lines) <= 1 or not body_size:
        return [(_block_text(bdict), _block_size(bdict), bdict.get("bbox"))]
    thresh = body_size * ratio
    i = 0
    while i < len(lines) and _line_size(lines[i]) >= thresh:
        i += 1
    groups = [lines[:i], lines[i:]] if 0 < i < len(lines) else [lines]
    out = []
    for g in groups:
        text = " ".join(_line_text(ln) for ln in g).strip()
        size = max(_line_size(ln) for ln in g)
        out.append((text, size, _union_bbox([ln.get("bbox") for ln in g])))
    return out


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
    vmargin_by_page = {}          # pno -> chapter name printed vertically in the side margin
    for pno, d in enumerate(per_page):
        if d is None:
            pages_failed.append(pno)
            continue
        page = doc[pno]
        tables = extract_page_tables(page)
        tboxes = [t["bbox"] for t in tables]
        raw = [b for b in d.get("blocks", []) if b.get("type") == 0 and _block_text(b)]

        # Harvest the vertical side-margin chapter label (read-only) — do NOT remove these blocks
        # from the pipeline: they stay furniture/heading exactly as before, so section_path and
        # embed_text are unchanged and retrieval is unaffected. The chapter rides as metadata only.
        vblocks = [b for b in raw if _is_vertical(b)]
        if vblocks:
            vtext = max((_block_text(b) for b in vblocks), key=len)
            vmargin_by_page[pno] = re.sub(r"\s+", " ", vtext).strip()

        page_cands = []
        for j, b in enumerate(raw):
            if _inside_any(_center(b["bbox"]), tboxes):     # part of a table → dedup out
                continue
            # Split a title-merged block (heading prefix + smaller remainder) into separate
            # candidates; most blocks yield exactly one. Split pieces get unique ids (b{j}, b{j}s1).
            for k, (text, size, bbox) in enumerate(split_heading_prefix(b, body_size)):
                page_cands.append({"idx": str(j) if k == 0 else f"{j}s{k}", "kind": None,
                                   "text": text, "size": size, "bbox": bbox})
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

    # Recover the PRINTED page number from the furniture we're about to drop: collect each
    # page's header/footer digit-runs, then fit a constant offset (page number vs year).
    header_nums = {}
    for c in candidates:
        if c["kind"] is None and _norm(c["text"]) in furniture:
            header_nums.setdefault(c["page"], []).extend(_ints(c["text"]))
    printed_by_page = _fit_printed_offset(header_nums)

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
                            text=c["text"], page=c["page"],
                            printed_page=printed_by_page.get(c["page"]),
                            order=order, confidence=conf, meta=meta))
        order += 1

    block_pages = {b.page for b in blocks}
    printed_fallback = sorted(p for p in block_pages if p not in printed_by_page)
    page_stats = {"pages_total": doc.page_count, "pages_with_text": pages_with_text,
                  "pages_failed": pages_failed, "pages_low_quality": pages_low_quality,
                  "body_size": body_size, "furniture_removed": len(furniture),
                  "tables_found": tables_found,
                  "pages_printed_recovered": len(block_pages) - len(printed_fallback),
                  "pages_printed_fallback": printed_fallback,
                  "vmargin_by_page": vmargin_by_page,
                  "printed_by_page": printed_by_page}
    return blocks, page_stats
