"""SHR-aware structural analysis — the document → CHAPTER → … hierarchy for HRCP annual reports.

Font-size heading detection cannot name the chapter: province titles are rendered in a raster
BANNER (so PUNJAB/SINDH/BALOCHISTAN often aren't even in the text layer) and a huge 36pt province
name on a divider page gets mis-crowned as the top level. The reliable signals, measured on the
real PDFs, are:
  - the **vertical side-margin** chapter name printed down every body page (PUNJAB, BALOCHISTAN …)
    — in the text layer, rotated 90°, harvested by `pdf.extract_blocks` as `vmargin_by_page`;
  - the **Table of Contents** (chapter title | printed start page) — gives the chapter vocabulary,
    order, and page ranges (a fallback for pages without a margin label: Introduction, appendices).

`assign_chapters` fuses them: the margin is primary, the TOC range fills the gaps, carry-forward
covers the rest. TOC page numbers are NOT fully trusted (one 2019 row reads 227 where the printed
book shows 251), so ranges are built only from the monotonically-increasing entries.

Everything here is text-based and deterministic — no vision, no model call.
"""

import re

import pymupdf

# Alias → canonical province chapter (helps the margin match and query entity extraction).
_PROVINCE_ALIASES = {
    "kp": "Khyber Pakhtunkhwa", "kpk": "Khyber Pakhtunkhwa",
    "ict": "Islamabad Capital Territory", "islamabad": "Islamabad Capital Territory",
    "ajk": "Azad Jammu and Kashmir", "gb": "Gilgit-Baltistan",
}


def normalize(s: str) -> str:
    """Lowercase, strip non-alphanumerics to spaces, collapse spaces — for matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


def extract_year(text: str):
    """The report year from a title page ('State of Human Rights in 2019' → 2019)."""
    m = re.search(r"state of human rights.{0,30}?((?:19|20)\d{2})", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b((?:19|20)\d{2})\b", text)
    return int(m.group(1)) if m else None


def _is_page_ref(text: str) -> bool:
    """A TOC right-column entry: optional dots/ellipsis then a page number."""
    return bool(re.fullmatch(r"[\s.…]*\d{1,4}\s*", text or ""))


def parse_toc(lines: list) -> list:
    """Parse a TOC page's `(x0, text)` lines (reading order) into ordered entries
    `{title, subtitle, page}`. A left-column line starts an entry; indented lines are an appendix
    description; a page-ref line closes the current entry. Entries with no page ref (front matter)
    keep page=None."""
    lines = [(x, t) for x, t in lines if t and t.strip()]
    if not lines:
        return []
    left = min(x for x, _ in lines)
    entries, cur = [], None
    for x, t in lines:
        if _is_page_ref(t):
            if cur is not None:
                cur["page"] = int(re.search(r"\d+", t).group())
                entries.append(cur)
                cur = None
            continue
        if x <= left + 8:                       # a new left-column title
            if cur is not None:
                entries.append(cur)             # previous entry had no page (front matter)
            cur = {"title": t.strip(), "subtitle": "", "page": None}
        else:                                   # indented continuation (appendix description)
            if cur is None:
                cur = {"title": t.strip(), "subtitle": "", "page": None}
            else:
                cur["subtitle"] = (cur["subtitle"] + " " + t.strip()).strip()
    if cur is not None:
        entries.append(cur)
    return entries


def chapter_name(entry: dict) -> str:
    """Display/citation name for a TOC entry — the description for appendices, else the title."""
    if re.match(r"(?i)^appendix\b", entry["title"]) and entry.get("subtitle"):
        return entry["subtitle"]
    return entry["title"]


# TOC rows that are the book title / navigation / plates — never content chapters.
_NOISE = ("state of human rights", "contents", "image sources")


def _is_noise(name: str) -> bool:
    n = normalize(name)
    return any(x in n for x in _NOISE)


def chapter_aliases(toc_entries: list) -> dict:
    """{normalized alias → canonical chapter name} from the TOC + province aliases."""
    amap = {}
    for e in toc_entries:
        if _is_noise(e["title"]):
            continue                            # skip book-title / 'Contents' noise rows
        name = chapter_name(e)
        for key in (name, e["title"], e.get("subtitle", "")):
            n = normalize(key)
            if n:
                amap.setdefault(n, name)
    for alias, canon in _PROVINCE_ALIASES.items():
        # only register an alias whose canonical chapter is actually in this TOC
        if any(normalize(chapter_name(e)) == normalize(canon) for e in toc_entries):
            amap.setdefault(alias, canon)
    return amap


def match_chapter(text: str, alias_map: dict):
    """Resolve a margin/title string to a canonical chapter via the alias map (exact, then
    substring both ways). None if nothing matches."""
    n = normalize(text)
    if not n:
        return None
    if n in alias_map:
        return alias_map[n]
    for alias, canon in alias_map.items():
        if len(alias) >= 3 and (alias in n or n in alias):
            return canon
    return None


def _toc_ranges(toc_entries: list) -> list:
    """[(canonical_name, printed_start, printed_end)] from TOC entries that HAVE a page, keeping
    only the monotonically-increasing ones (a stray/corrupt page number is skipped)."""
    numbered = [(chapter_name(e), e["page"]) for e in toc_entries
                if e.get("page") is not None and not _is_noise(e["title"])]
    mono, last = [], -1
    for name, pg in numbered:
        if pg > last:
            mono.append((name, pg))
            last = pg
    ranges = []
    for i, (name, start) in enumerate(mono):
        end = mono[i + 1][1] - 1 if i + 1 < len(mono) else 10 ** 9
        ranges.append((name, start, end))
    return ranges


def _range_lookup(ranges: list, printed: int):
    for name, start, end in ranges:
        if start <= printed <= end:
            return name
    return None


def assign_chapters(vmargin_by_page: dict, toc_entries: list, printed_by_page: dict,
                    n_pages: int, alias_map: dict) -> dict:
    """Per-PDF-page chapter: vertical margin (primary) → TOC printed-range (fallback) →
    carry-forward (fill)."""
    ch = {}
    for pno, txt in vmargin_by_page.items():            # 1. margin — authoritative, text-layer
        c = match_chapter(txt, alias_map)
        if c:
            ch[pno] = c
    ranges = _toc_ranges(toc_entries)                   # 2. TOC printed-range for the rest
    for pno in range(n_pages):
        if pno in ch:
            continue
        pp = printed_by_page.get(pno)
        if pp is not None:
            c = _range_lookup(ranges, pp)
            if c:
                ch[pno] = c
    last = None                                         # 3. carry-forward across gaps
    for pno in range(n_pages):
        if pno in ch:
            last = ch[pno]
        elif last is not None:
            ch[pno] = last
    return ch


def front_matter(path: str, max_pages: int = 14) -> dict:
    """Read the front pages: {year, toc_entries, toc_pno, cover}. Cover = an (almost) text-less
    first page. Opens the PDF directly (cheap — only the first pages)."""
    doc = pymupdf.open(path)
    pages = []
    for pno in range(min(max_pages, doc.page_count)):
        d = doc[pno].get_text("dict")
        lines = []
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            for ln in b.get("lines", []):
                t = "".join(s.get("text", "") for s in ln.get("spans", [])).strip()
                if t:
                    lines.append((round(ln["bbox"][0]), t))
        pages.append(lines)

    cover = bool(pages) and sum(len(t) for _, t in pages[0]) < 10
    year = None
    for lines in pages[:4]:
        blob = " ".join(t for _, t in lines)
        if "state of human rights" in blob.lower():
            year = extract_year(blob)
            if year:
                break

    best_n, toc_pno = -1, None
    for pno, lines in enumerate(pages):
        n = sum(1 for _, t in lines if _is_page_ref(t))
        if n > best_n:
            best_n, toc_pno = n, pno
    toc_entries = parse_toc(pages[toc_pno]) if best_n >= 4 else []
    return {"year": year, "toc_entries": toc_entries, "toc_pno": toc_pno, "cover": cover}


def is_shr_annual(fm: dict, vmargin_by_page: dict) -> bool:
    """An SHR annual report: a 'State of Human Rights' year + a province TOC (so the structural
    path applies). Other docs (FFM/thematic) return False → the generic font path is used."""
    titles = {normalize(e["title"]) for e in fm.get("toc_entries", [])}
    provinces = {"punjab", "sindh", "balochistan", "khyber pakhtunkhwa"}
    return bool(fm.get("year")) and len(provinces & titles) >= 2
