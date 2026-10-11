"""Normalisation helpers shared by the three API clients.

Every client converts its native format into one plain dict shape
(``origin, external_id, doi, title, authors, year, venue, abstract, url``) so
that the Processor never needs to know which API a record came from. A plain
dict is used rather than a class because the record is written straight into
the blackboard's ``sources`` table, whose columns mirror these keys.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

_STOPWORDS = frozenset("""a an and are as at be by for from how in into is it of on or that the
their this to what which who why with within does do can vs versus using use based""".split())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalise_doi(doi: str | None) -> str | None:
    """Lower-case bare DOI ("10.x/y"). DOIs are case-insensitive by
    specification, and APIs disagree on whether to prefix https://doi.org/,
    so normalising is what makes DOI de-duplication reliable."""
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    doi = doi.removeprefix("doi:")
    return doi or None


def normalise_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def strip_markup(text: str | None) -> str | None:
    """Crossref abstracts arrive as JATS XML; arXiv's contain line breaks."""
    if not text:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(abstract|summary)[\s.:]+", "", text, flags=re.I)
    return text or None


def keywords(query: str, limit: int = 6) -> list[str]:
    """Content words of a query, used to build arXiv's boolean search string."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", query.lower())
    seen: list[str] = []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            seen.append(w)
    return seen[:limit]


def reconstruct_inverted_abstract(inv: dict[str, list[int]] | None) -> str | None:
    """OpenAlex ships abstracts as an inverted index ({word: [positions]})
    for licensing reasons; rebuild the plain text for the Processor."""
    if not inv:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions)) or None
