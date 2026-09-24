"""Persistence — make ingestion incremental. Parsing (PyMuPDF, table detection) is
the expensive step; embeddings are already cached by chunk hash. So we persist the
`ParsedDoc` plus a manifest keyed by the source file's checksum: re-ingesting an
unchanged file is a no-op load (<1s), and a changed file (new checksum) re-parses.

Writing to disk is a side effect, so `ingest` gates overwrite behind cache-skip +
an explicit `force` flag rather than silently clobbering.
"""

import hashlib
import json
import os

from framework.ingest.ir import (
    Block, Coverage, GlossaryEntry, OutlineNode, ParsedDoc, TemporalAnchor,
)


def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(doc_dir: str) -> str:
    return os.path.join(doc_dir, "manifest.json")


def _parsed(doc_dir: str) -> str:
    return os.path.join(doc_dir, "parsed.json")


def is_ingested(doc_dir: str, checksum: str) -> bool:
    """True iff `doc_dir` already holds a parsed doc for exactly this file checksum."""
    if not os.path.exists(_manifest(doc_dir)) or not os.path.exists(_parsed(doc_dir)):
        return False
    try:
        with open(_manifest(doc_dir), encoding="utf-8") as f:
            return json.load(f).get("checksum") == checksum
    except Exception:
        return False


def save_manifest(doc: ParsedDoc, doc_dir: str, *, embedding_model: str | None = None,
                  embedding_dims: int | None = None) -> None:
    os.makedirs(doc_dir, exist_ok=True)
    manifest = {"doc_id": doc.doc_id, "checksum": doc.checksum, "profile": doc.profile,
                "page_count": doc.page_count, "blocks": len(doc.blocks),
                "embedding_model": embedding_model, "embedding_dims": embedding_dims}
    with open(_manifest(doc_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_parsed(doc_dir: str) -> ParsedDoc | None:
    if not os.path.exists(_parsed(doc_dir)):
        return None
    try:
        with open(_parsed(doc_dir), encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    return _from_dict(d)


def _from_dict(d: dict) -> ParsedDoc:
    blocks = [Block(id=b["id"], kind=b["kind"], text=b["text"], page=b["page"],
                    section_path=tuple(b.get("section_path") or ()), order=b.get("order", 0),
                    confidence=b.get("confidence", 1.0), meta=b.get("meta") or {})
              for b in d.get("blocks", [])]
    outline = [OutlineNode(**o) for o in d.get("outline", [])]
    glossary = {k: GlossaryEntry(**v) for k, v in (d.get("glossary") or {}).items()}
    temporal = TemporalAnchor(**(d.get("temporal") or {}))
    coverage = Coverage(**(d.get("coverage") or {}))
    return ParsedDoc(doc_id=d["doc_id"], source_path=d["source_path"], checksum=d["checksum"],
                     profile=d["profile"], title=d.get("title", ""),
                     page_count=d.get("page_count", 0), outline=outline, blocks=blocks,
                     glossary=glossary, temporal=temporal, coverage=coverage)
