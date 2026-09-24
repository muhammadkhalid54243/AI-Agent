"""Document ingestion — source files → a structured intermediate representation.

Parsing is document-domain logic with no notion of vectors, so it lives apart from
`framework/rag` and is testable without an API key. The heart of this package is the
COVERAGE LEDGER (`ir.Coverage`): every block gets an id, every chunk later records the
block ids it consumed, and any block that reaches no chunk is an ORPHAN — content that
survived parsing then silently vanished before it could ever be retrieved. Silent loss
is the enemy, not loss itself, so orphans are a hard gate, not a warning.
"""

from framework.ingest.ir import (
    Block, Coverage, GlossaryEntry, OutlineNode, ParsedDoc, TemporalAnchor,
)
from framework.ingest.parse import ingest

__all__ = [
    "Block", "Coverage", "GlossaryEntry", "OutlineNode", "ParsedDoc", "TemporalAnchor",
    "ingest",
]
