"""Font-heuristic outline + section-path assignment.

SHR's embedded TOC is garbage (12 bookmarks all named with a sha256), so we can't
trust `get_toc()`. Instead we reconstruct the hierarchy from heading font sizes:
the largest distinct heading size is level 1, next is level 2, and so on (capped).
Then we walk the document in reading order, maintaining a heading stack, and stamp
every block with its enclosing `section_path` — which is what makes a citation real
and what keeps structure-aware chunking from crossing a section boundary later.
"""

import dataclasses

from framework.ingest.ir import Block, OutlineNode

MAX_LEVELS = 4


def _level_map(headings: list) -> dict:
    """Map each distinct heading font size to a level (largest size = level 1)."""
    sizes = sorted({round(b.meta.get("font_size", 0), 1) for b in headings}, reverse=True)
    return {s: min(i + 1, MAX_LEVELS) for i, s in enumerate(sizes)}


def build_outline_and_sections(blocks: list, chapter_by_page: dict | None = None) -> tuple:
    """Return (outline, blocks_with_section_path).

    Headings get their ancestor path (excluding themselves); other blocks get the full enclosing
    path. Blocks are frozen, so we return new ones via replace().

    When `chapter_by_page` is given (SHR structural path), the page's CHAPTER (from the TOC +
    vertical margin, see shr.py) is stamped on each block as `meta["chapter"]` — a SEPARATE level
    used for CITATIONS and query filtering. `section_path` is computed EXACTLY as without it, so
    `embed_text` (which `section_path` feeds) is byte-for-byte unchanged and retrieval is unaffected
    — chapter correctness is decoupled from the embedding (measured: prepending or reshaping
    `section_path` regressed SHR 2024 recall; stamping chapter as metadata does not).
    """
    headings = [b for b in blocks if b.kind == "heading"]
    lmap = _level_map(headings)
    outline = []
    stack = []          # [(level, title)]
    out_blocks = []

    def _finish(b, ch, sect):
        meta = {**b.meta, "chapter": ch} if ch else b.meta
        return dataclasses.replace(b, section_path=sect, meta=meta)

    for b in blocks:
        ch = chapter_by_page.get(b.page) if chapter_by_page else None
        if b.kind == "heading":
            lvl = lmap.get(round(b.meta.get("font_size", 0), 1), MAX_LEVELS)
            while stack and stack[-1][0] >= lvl:   # pop siblings/deeper
                stack.pop()
            ancestors = tuple(t for _, t in stack)
            outline.append(OutlineNode(level=lvl, title=b.text, page=b.page))
            stack.append((lvl, b.text))
            out_blocks.append(_finish(b, ch, ancestors))
        else:
            out_blocks.append(_finish(b, ch, tuple(t for _, t in stack)))

    return outline, out_blocks
