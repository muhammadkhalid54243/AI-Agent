"""Table detection — a table is NOT prose. Chunking a table as running text destroys
it (a financial row becomes '2024 2025 Revenue 41 52'), and no embedding or reranker
recovers that. So each detected table becomes ONE atomic Block(kind='table') holding
Markdown, and the text inside its bounding box is excluded from the prose blocks so it
is not double-counted. Caption binding keeps '[Table 4.2: ...]' with its numbers.
"""

import re


def extract_page_tables(page) -> list:
    """Return [{'bbox': (x0,y0,x1,y1), 'markdown': str}] for tables on `page`.

    Degrades to [] on any parser error — a failed table never aborts the ingest.
    """
    try:
        finder = page.find_tables()
    except Exception:
        return []
    out = []
    for t in getattr(finder, "tables", []):
        try:
            md = (t.to_markdown() or "").strip()
        except Exception:
            continue
        if md:
            out.append({"bbox": tuple(t.bbox), "markdown": md})
    return out


def find_caption(text_blocks: list, table_bbox: tuple) -> str:
    """A caption is a nearby line starting with 'Table'. text_blocks = [(text, bbox)]."""
    top, bottom = table_bbox[1], table_bbox[3]
    for text, bbox in text_blocks:
        if re.match(r"(?i)^table\b", text.strip()):
            if abs(bbox[3] - top) < 80 or abs(bbox[1] - bottom) < 80:  # just above/below
                return text.strip()[:120]
    return ""
