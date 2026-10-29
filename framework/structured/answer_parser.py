"""Cited-answer output parser — turns a model's cited answer into a clean BULLET list.

The served answer must always read the same way: each cited answer as its own bullet, in its own
paragraph (blank line before the next), with the verbatim segment quote on its own line beneath the
fact. Relying on the model to format that is unreliable, so this OUTPUT PARSER re-emits it
deterministically. It is a LangChain `BaseOutputParser`, so it composes as `prompt | model | parser`
and can also be called directly (`CitedAnswerParser().parse(text)`).

Parsing is anchored on the citation brackets the answer already carries ([Report — Chapter, pN]):
the text since the previous citation is the fact, the bracket is the source, and an immediately
following "..." line is the segment quote. Identical facts are merged with their sources combined
(the cross-doc dedup). If the text carries no citation at all (e.g. the NOT_COVERED abstention), it
is passed through unchanged — never bulletised, never dropped.
"""

import re

from langchain_core.output_parsers import BaseOutputParser

_CITE = re.compile(r"\[([^\[\]]+)\]")
_BULLET = re.compile(r"^\s*[-*•]\s*")


def _key(fact: str) -> str:
    """Normalized key for merging identical facts (case/space/trailing-punct-insensitive)."""
    return re.sub(r"\s+", " ", (fact or "")).strip().strip(".;:,").lower()


def parse_units(text: str) -> list:
    """Parse a cited answer into ordered units `{fact, citation, quote}`, one per citation.

    The fact is the text since the previous citation (leading bullet markers stripped, whitespace
    collapsed); the citation is the bracket content; the quote is a `"..."` immediately after it.
    Returns [] when the text has no citation (caller passes such text through unchanged).
    """
    units, last = [], 0
    for m in _CITE.finditer(text):
        fact = _BULLET.sub("", text[last:m.start()].strip())
        fact = re.sub(r"\s+", " ", fact).strip(' "')
        rest = text[m.end():]
        qm = re.match(r'\s*"([^"]*)"', rest)
        quote = qm.group(1).strip() if qm else ""
        last = m.end() + (qm.end() if qm else 0)
        if fact or m.group(1).strip():
            units.append({"fact": fact, "citation": m.group(1).strip(), "quote": quote})
    return units


def render_units(units: list) -> str:
    """Render units as a bullet list: `- <fact> [<sources>]`, the quote on its own indented line,
    a BLANK LINE between bullets. Identical facts are merged, their sources combined (de-duped)."""
    order, groups = [], {}
    for u in units:
        k = _key(u.get("fact", ""))
        g = groups.get(k)
        if g is None:
            g = groups[k] = {"fact": u.get("fact", ""), "sources": [], "quote": u.get("quote") or ""}
            order.append(k)
        src = (u.get("citation") or "").strip()
        if src and src not in g["sources"]:
            g["sources"].append(src)
        if not g["quote"] and u.get("quote"):
            g["quote"] = u["quote"]
    blocks = []
    for k in order:
        g = groups[k]
        head = f"- {g['fact']} [{'; '.join(g['sources'])}]" if g["sources"] else f"- {g['fact']}"
        if g["quote"]:
            head += f'\n  "{g["quote"]}"'
        blocks.append(head)
    return "\n\n".join(blocks)


class CitedAnswerParser(BaseOutputParser):
    """Re-emit a cited answer as a deterministic bullet list (see module docstring)."""

    def parse(self, text: str) -> str:
        units = parse_units(text or "")
        return render_units(units) if units else (text or "")

    @property
    def _type(self) -> str:
        return "cited_answer"
