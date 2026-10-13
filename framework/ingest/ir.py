"""The intermediate representation — get these shapes right first; everything
downstream (chunking, coverage accounting, citations) joins on them.

A `Block` is the unit of coverage accounting: it carries a stable id and the
`section_path`/`page` that make a citation real. `Coverage` is the loss ledger.
`ParsedDoc` is the whole parsed document plus its ledger.
"""

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal[
    "heading", "paragraph", "table", "list", "caption", "figure", "code", "reference",
]


@dataclass(frozen=True)
class Block:
    id: str                          # f"{doc_id}:p{page}:b{n}" — unit of coverage accounting
    kind: BlockKind
    text: str                        # Markdown
    page: int                        # PDF page INDEX (0-based) — the RETRIEVAL key
    printed_page: int | None = None  # the number PRINTED on the page (for CITATIONS); None = fallback
    section_path: tuple = ()         # ("4. Transmission", "4.2 Losses")
    order: int = 0                   # document reading order
    confidence: float = 1.0          # extraction confidence; low → flagged in coverage
    meta: dict = field(default_factory=dict)  # heading level, table_id, caption_of, printed_page...


@dataclass
class OutlineNode:
    level: int
    title: str
    page: int                        # PDF page index where the section starts


@dataclass
class GlossaryEntry:
    expansion: str = ""              # "Power Purchase Agreement" for "PPA"
    definition: str = ""             # a fuller sentence definition, when present
    source_page: int | None = None


@dataclass
class TemporalAnchor:
    """Document-level anchors for resolving relative dates ('the previous year')."""
    document_date: str | None = None
    fiscal_year_start: str = "January"   # explicit, never inferred (calendar ≠ fiscal)
    resolved: dict = field(default_factory=dict)  # phrase -> explicit range


@dataclass
class Coverage:
    """The loss ledger. Converts SILENT loss into REPORTED loss.

    `orphan_blocks` MUST be empty — a block extracted but never indexed is content
    that silently vanished. Ingestion fails loudly on orphans (see `assert_complete`).
    """
    pages_total: int = 0
    pages_with_text: int = 0
    pages_via_vision: int = 0            # needed OCR/VLM fallback
    pages_failed: list = field(default_factory=list)      # page indices that failed outright
    blocks_extracted: int = 0
    blocks_low_confidence: list = field(default_factory=list)  # block ids below threshold
    tables_found: int = 0
    tables_split: int = 0                # exceeded max_tokens and were row-split
    orphan_blocks: list = field(default_factory=list)     # extracted but never indexed. MUST be empty.
    pages_low_quality: list = field(default_factory=list)  # text extracted but mostly unmapped glyphs
    pages_printed_recovered: int = 0     # pages whose PRINTED page number was recovered from headers
    pages_printed_fallback: list = field(default_factory=list)  # pages citing PDF index (no printed #)
    glossary_terms: int = 0
    temporal_anchors: int = 0
    warnings: list = field(default_factory=list)

    @property
    def has_orphans(self) -> bool:
        return bool(self.orphan_blocks)

    def assert_complete(self) -> None:
        """Hard gate: raise if any block never reached a chunk."""
        if self.orphan_blocks:
            raise OrphanBlocksError(self.orphan_blocks)

    def to_markdown(self) -> str:
        lines = [
            "# Coverage report",
            f"- pages: {self.pages_with_text}/{self.pages_total} with text"
            f" ({self.pages_via_vision} via vision, {len(self.pages_failed)} failed)",
            f"- blocks extracted: {self.blocks_extracted}"
            f" ({len(self.blocks_low_confidence)} low-confidence)",
            f"- tables: {self.tables_found} found, {self.tables_split} row-split",
            f"- printed page numbers: {self.pages_printed_recovered} recovered, "
            f"{len(self.pages_printed_fallback)} fallback (cite PDF index)",
            f"- glossary terms: {self.glossary_terms} | temporal anchors: {self.temporal_anchors}",
            f"- ORPHAN BLOCKS: {len(self.orphan_blocks)}"
            + ("  ← MUST be 0" if self.orphan_blocks else "  ✓"),
            f"- LOW-QUALITY (garbled/unmapped-glyph) pages: {len(self.pages_low_quality)}"
            + ("  ← needs vision/OCR" if self.pages_low_quality else "  ✓"),
        ]
        if self.pages_low_quality:
            lines.append(f"- low-quality pages: {self.pages_low_quality[:30]}")
        if self.pages_failed:
            lines.append(f"- failed pages: {self.pages_failed}")
        if self.warnings:
            lines.append("\n## Warnings")
            lines += [f"- {w}" for w in self.warnings]
        return "\n".join(lines)


class OrphanBlocksError(RuntimeError):
    """Raised by Coverage.assert_complete when blocks were extracted but never indexed."""

    def __init__(self, orphan_ids: list):
        self.orphan_ids = orphan_ids
        preview = ", ".join(orphan_ids[:10]) + (" ..." if len(orphan_ids) > 10 else "")
        super().__init__(f"{len(orphan_ids)} orphan block(s) never indexed: {preview}")


@dataclass
class ParsedDoc:
    doc_id: str
    source_path: str
    checksum: str                    # sha256 — drives incremental re-ingest
    profile: str                     # which DocumentProfile handled it
    title: str = ""
    page_count: int = 0
    outline: list = field(default_factory=list)     # list[OutlineNode]
    blocks: list = field(default_factory=list)      # list[Block]
    glossary: dict = field(default_factory=dict)    # term -> GlossaryEntry
    temporal: TemporalAnchor = field(default_factory=TemporalAnchor)
    coverage: Coverage = field(default_factory=Coverage)

    def block_ids(self) -> set:
        return {b.id for b in self.blocks}

    def to_parsed_markdown(self) -> str:
        """The full extracted text, so a human can SEE what the parser produced."""
        out = []
        for b in self.blocks:
            crumb = " > ".join(b.section_path)
            head = f"<!-- {b.id} | {b.kind} | p{b.page}" + (f" | {crumb}" if crumb else "") + " -->"
            out.append(head + "\n" + b.text)
        return "\n\n".join(out)
