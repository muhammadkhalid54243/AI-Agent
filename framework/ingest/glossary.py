"""Glossary / abbreviation harvesting.

An acronym as a bare token carries almost no semantic signal, so a query about "PECA"
lands nowhere near a chunk spelling out "Prevention of Electronic Crimes Act" — the
silent vocabulary-mismatch miss we measured in Phase 0 (shr24-003 was None even @20).
Harvesting the expansions here lets Phase-3 enrichment append them to `embed_text`.

The robust signal across document types is the INLINE first-use definition —
`Expansion (ACRONYM)`. To avoid capturing the whole sentence before the parenthesis,
we walk BACKWARD from `(ACRONYM)` and keep only the minimal trailing span of words
whose initials spell the acronym (skipping small connector words like of/and/the).
That both extracts a clean expansion AND validates it's a real initialism.
"""

import re

from framework.ingest.ir import GlossaryEntry

_ACRONYM = re.compile(r"\(([A-Z]{2,8})\)")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&.'\-]*")
_SMALL = {"of", "the", "and", "for", "in", "to", "a", "an", "on", "&", "de", "da"}


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


def _plausible(expansion: str, acronym: str) -> bool:
    """The acronym's letters form an initialism of the expansion's words."""
    words = _WORD.findall(expansion)
    if len(words) < 2:
        return False
    return _is_subsequence(acronym.upper(), "".join(w[0] for w in words).upper())


def _trailing_expansion(pre_text: str, acronym: str) -> str | None:
    """Return the minimal trailing word-span of `pre_text` whose initials spell
    `acronym`, or None if the words before the parenthesis are not that initialism."""
    words = _WORD.findall(pre_text)
    target = list(acronym.upper())
    i, j, start = len(words) - 1, len(target) - 1, None
    while i >= 0 and j >= 0:
        w = words[i]
        if w.lower() in _SMALL:                 # connector: keep scanning, don't consume
            i -= 1
            continue
        if w[0].upper() == target[j]:
            start, j, i = i, j - 1, i - 1        # matched this acronym letter
        else:
            break
    if j < 0 and start is not None:
        return " ".join(words[start:])           # includes intermediate small words
    return None


def harvest_glossary(blocks: list) -> dict:
    """Return {acronym: GlossaryEntry(expansion, source_page)} from inline definitions."""
    gloss = {}
    for b in blocks:
        for m in _ACRONYM.finditer(b.text):
            acronym = m.group(1)
            if acronym in gloss:
                continue
            expansion = _trailing_expansion(b.text[:m.start()], acronym)
            if expansion:
                gloss[acronym] = GlossaryEntry(expansion=expansion, source_page=b.page)
    return gloss
