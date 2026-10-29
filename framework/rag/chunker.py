"""Chunking — turn a document into retrieval units.

Two paths:
  - `chunk_text` (legacy): a 300-word sliding window over raw text. Structure-blind —
    it slices tables mid-row and straddles section boundaries. Kept for existing callers.
  - `structure_aware_chunks` (Phase 2): chunks a parsed `ParsedDoc` using its structure.

The Phase-2 design in one line: decouple WHAT MATCHES from WHAT ANSWERS. Small child
chunks embed precisely (less semantic noise dilutes the vector); a big parent chunk
carries the context needed to answer. Tables are atomic (a flattened table is
unusable), and no chunk crosses a section boundary (the author's own semantics).
"""

from dataclasses import dataclass, field

from framework.context.budget import count_tokens


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += max(1, chunk_size - overlap)
    return chunks


@dataclass
class Chunk:
    id: str
    text: str                    # author's words — what is RETURNED to the model
    embed_text: str              # what is EMBEDDED — = text in Phase 2; Phase 3 enriches it
    doc_id: str
    page_start: int
    page_end: int
    printed_page_start: int | None = None  # PRINTED page of page_start (citation); None = fallback
    section_path: tuple = ()
    kind: str = "prose"          # "prose" | "table"
    chapter: str | None = None   # SHR structural chapter (TOC/margin) — for citation + filtering
    parent_id: str | None = None  # set on children; None on parents and lone leaves
    block_ids: list = field(default_factory=list)  # coverage join key (Block ids consumed)


def leaf_chunks(chunks: list) -> list:
    """The units to INDEX: children, lone leaves, and tables — not the big parents.
    A chunk is a parent iff some other chunk names it as parent_id."""
    referenced = {c.parent_id for c in chunks if c.parent_id}
    return [c for c in chunks if c.id not in referenced]


def _split_table_rows(md: str, max_tokens: int) -> list:
    """Split an oversized Markdown table by rows, repeating the header in each piece."""
    lines = md.splitlines()
    header = lines[:2] if len(lines) >= 2 else lines      # header row + separator
    rows = lines[2:] if len(lines) >= 2 else []
    pieces, cur = [], []
    for row in rows:
        cur.append(row)
        if count_tokens("\n".join(header + cur)) > max_tokens and len(cur) > 1:
            cur.pop()
            pieces.append("\n".join(header + cur))
            cur = [row]
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces or [md]


def structure_aware_chunks(doc, *, child_target: int = 200, child_max: int = 320) -> list:
    """`ParsedDoc` -> list[Chunk]. Small children + a big parent per leaf section;
    atomic tables; never cross a section boundary. Sets `doc.coverage.orphan_blocks`
    (every block must land in >=1 chunk) so `assert_complete()` is meaningful."""
    chunks = []
    consumed = set()
    printed = {b.page: b.printed_page for b in doc.blocks}   # PDF page -> printed page number
    chapter = {b.page: b.meta.get("chapter") for b in doc.blocks}  # PDF page -> SHR chapter
    n = 0

    def mk(text, kind, page_start, page_end, section_path, parent_id, block_ids):
        nonlocal n
        c = Chunk(id=f"{doc.doc_id}:ck{n}", text=text, embed_text=text, doc_id=doc.doc_id,
                  page_start=page_start, page_end=page_end,
                  printed_page_start=printed.get(page_start), section_path=tuple(section_path),
                  kind=kind, chapter=chapter.get(page_start), parent_id=parent_id,
                  block_ids=list(block_ids))
        n += 1
        consumed.update(block_ids)
        chunks.append(c)
        return c

    for section_path, group in _group_by_section(doc.blocks):
        parent_id = f"{doc.doc_id}:ck{n}"          # reserve id for this section's parent
        prose = [b for b in group if b.kind != "table"]
        tables = [b for b in group if b.kind == "table"]

        # Big parent = the whole section's prose (context unit). Emit only if there is
        # more than one child, so a short section is a single chunk, not a duplicated pair.
        children = list(_pack_children(prose, child_target, child_max))
        has_parent = len(children) > 1
        if has_parent:
            ptext = "\n\n".join(b.text for b in prose)
            pages = [b.page for b in prose]
            mk(ptext, "prose", min(pages), max(pages), section_path, None,
               [b.id for b in prose])   # this IS ck{parent_id} because n was reserved

        # Children (or the lone leaf when has_parent is False).
        for cblocks in children:
            pages = [b.page for b in cblocks]
            mk("\n\n".join(b.text for b in cblocks), "prose", min(pages), max(pages),
               section_path, parent_id if has_parent else None, [b.id for b in cblocks])

        # Tables — atomic, row-split only if oversized.
        for t in tables:
            if count_tokens(t.text) > child_max:
                for piece in _split_table_rows(t.text, child_max):
                    mk(piece, "table", t.page, t.page, section_path,
                       parent_id if has_parent else None, [t.id])
                doc.coverage.tables_split += 1
            else:
                mk(t.text, "table", t.page, t.page, section_path,
                   parent_id if has_parent else None, [t.id])

    doc.coverage.orphan_blocks = sorted(doc.block_ids() - consumed)
    return chunks


def _group_by_section(blocks: list):
    """Yield (section_path, [blocks]) for consecutive runs sharing a section_path."""
    group, current = [], None
    for b in blocks:
        if current is None or b.section_path == current:
            group.append(b)
            current = b.section_path
        else:
            yield current, group
            group, current = [b], b.section_path
    if group:
        yield current, group


def _pack_children(prose_blocks: list, target: int, hard_max: int):
    """Pack prose blocks into children of ~target tokens (hard break at hard_max),
    never splitting a block. A single over-max block becomes its own child."""
    cur, cur_tok = [], 0
    for b in prose_blocks:
        bt = count_tokens(b.text)
        if cur and cur_tok + bt > hard_max:
            yield cur
            cur, cur_tok = [], 0
        cur.append(b)
        cur_tok += bt
        if cur_tok >= target:
            yield cur
            cur, cur_tok = [], 0
    if cur:
        yield cur
