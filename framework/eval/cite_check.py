"""Deterministic citation faithfulness — the free half of the output-side verifier.

The model writes each citation by copying a passage's source line (`Report — Chapter, pN`,
per RAG_SYS). When it deviates — the measured bug is a Balochistan passage cited
`Khyber Pakhtunkhwa` — we don't need a model to catch it: every retrieved passage already
carries its AUTHORITATIVE chapter (Part 1's TOC metadata, via serve `_loc`/`_page`). So we
rebuild the set of valid source lines as a map `{(report_label, page_token): chapter}` and
hold each citation to it:

  - cited chapter disagrees with the passage at that (report, page)  -> CORRECT to the
    authoritative chapter (we KNOW the right one);
  - (report, page) matches no retrieved passage                      -> STRIP the chapter
    (degrade to `Report, pN`), never assert a breadcrumb we can't back.

This is degrade/correct-only: it never drops a fact, so it can't regress coverage, and it
costs zero model calls. Claim relevance/entailment (does the quote support the claim, does
the claim answer the question) is a judgment and stays in the model path (`judge.py`).
"""

# Source line shape: "Report(s) — Chapter/Section, pN". Reports may be merged across years
# for reproduced text ("SHR 2023; SHR 2024 — ..."); the page is the LAST ", p" field so a
# section path containing " > " is never mistaken for the page.
_DASH = " — "


def parse_citation(citation: str):
    """Split a citation's bracket content into (reports, loc, page).

    "SHR 2024 — Balochistan, p18" -> (["SHR 2024"], "Balochistan", "p18")
    "SHR 2023; SHR 2024 — UDHR, p5" -> (["SHR 2023", "SHR 2024"], "UDHR", "p5")
    A citation with no " — " (already degraded, or malformed) -> (reports, "", page?).
    """
    s = (citation or "").strip()
    if not s:
        return [], "", ""
    if _DASH in s:
        left, right = s.split(_DASH, 1)
    else:
        left, right = s, ""           # no chapter present
    reports = [r.strip() for r in left.split(";") if r.strip()]
    loc, page = right, ""
    if right:
        head, _, tail = right.rpartition(", ")
        if tail.startswith("p") or tail.startswith("PDF p"):
            loc, page = head.strip(), tail.strip()
        else:                         # left/right had no trailing page; whole right is loc
            loc = right.strip()
    else:
        # degraded form "Report, p18": the page rode on the left half
        head, _, tail = left.rpartition(", ")
        if tail.startswith("p") or tail.startswith("PDF p"):
            reports = [r.strip() for r in head.split(";") if r.strip()]
            loc, page = "", tail.strip()
    return reports, loc, page


def _fmt(reports, loc, page):
    rep = "; ".join(reports)
    if loc and page:
        return f"{rep}{_DASH}{loc}, {page}"
    if page:
        return f"{rep}, {page}"       # degraded: report + page, no chapter
    return rep


def correct_citation(citation: str, authmap: dict) -> str:
    """Return the citation with its chapter corrected to, or degraded against, `authmap`
    ({(report_label, page_token): authoritative_loc}). Pure; serve builds `authmap` from
    its hits via `_loc`/`_page`/`doc_label`."""
    reports, loc, page = parse_citation(citation)
    if not reports or not page:
        return citation                       # nothing to anchor on; leave as written
    auth = next((authmap[(r, page)] for r in reports if (r, page) in authmap), None)
    if auth is None:
        return _fmt(reports, "", page) if loc else citation   # unmappable -> strip chapter
    if auth and auth != loc:
        return _fmt(reports, auth, page)                      # disagree -> correct
    return citation                                           # agrees (or no better loc)
