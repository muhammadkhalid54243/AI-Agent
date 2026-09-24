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


def build_outline_and_sections(blocks: list) -> tuple:
    """Return (outline, blocks_with_section_path).

    Headings get their ancestor path (excluding themselves); other blocks get the
    full enclosing path. Blocks are frozen, so we return new ones via replace().
    """
    headings = [b for b in blocks if b.kind == "heading"]
    lmap = _level_map(headings)
    outline = []
    stack = []          # [(level, title)]
    out_blocks = []

    for b in blocks:
        if b.kind == "heading":
            lvl = lmap.get(round(b.meta.get("font_size", 0), 1), MAX_LEVELS)
            while stack and stack[-1][0] >= lvl:   # pop siblings/deeper
                stack.pop()
            ancestors = tuple(t for _, t in stack)
            outline.append(OutlineNode(level=lvl, title=b.text, page=b.page))
            stack.append((lvl, b.text))
            out_blocks.append(dataclasses.replace(b, section_path=ancestors))
        else:
            out_blocks.append(dataclasses.replace(b, section_path=tuple(t for _, t in stack)))

    return outline, out_blocks
