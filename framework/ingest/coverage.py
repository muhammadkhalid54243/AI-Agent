"""Coverage report writers — put what the parser produced on disk so a human can SEE
it, rather than infer three phases later from a bad answer.

Writes per document:
  - parsed.md    the full extracted text (eyeball structure: nesting, no interleaving)
  - coverage.md  the loss ledger, human-readable (orphans, failed pages, low-confidence)
  - parsed.json  the machine-readable ParsedDoc (drives incremental re-ingest)
"""

import dataclasses
import json
import os

from framework.ingest.ir import ParsedDoc


def write_reports(doc: ParsedDoc, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    paths["parsed_md"] = os.path.join(out_dir, "parsed.md")
    with open(paths["parsed_md"], "w", encoding="utf-8") as f:
        f.write(doc.to_parsed_markdown())

    paths["coverage_md"] = os.path.join(out_dir, "coverage.md")
    with open(paths["coverage_md"], "w", encoding="utf-8") as f:
        f.write(doc.coverage.to_markdown())

    paths["parsed_json"] = os.path.join(out_dir, "parsed.json")
    with open(paths["parsed_json"], "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(doc), f, indent=2, ensure_ascii=False, default=str)

    return paths
