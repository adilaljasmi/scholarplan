"""Clients for the three keyless scholarly APIs named in the Unit 6 design.

Why these three (Unit 6, section 2.1): each returns a stable identifier
(arXiv ID or DOI), so every claim in the final report is traceable to a
source, and none needs an account, which keeps the system reproducible by a
marker. They are also complementary: arXiv gives recent CS preprints with full
abstracts, OpenAlex gives broad coverage of published work (Priem, Piwowar and
Orr, 2022), and Crossref is the authoritative DOI registry.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from scholarplan.sources.http import ScholarlyHTTP
from scholarplan.sources.records import (keywords, normalise_doi, now_iso,
                                         reconstruct_inverted_abstract, strip_markup)

ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivClient:
    name = "arxiv"
    URL = "https://export.arxiv.org/api/query"

    def __init__(self, http: ScholarlyHTTP):
        self.http = http

    @staticmethod
    def build_query(query: str) -> str:
        # arXiv's search is boolean, not ranked free text: a long natural-
        # language string returns nothing. AND-ing the content words keeps
        # precision high; if that over-constrains, the Retriever's "no
        # results" feedback lets the Planner broaden the query.
        terms = keywords(query, limit=5)
        return " AND ".join(f"all:{t}" for t in terms) if terms else f'all:"{query}"'

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        params = {"search_query": self.build_query(query), "start": 0, "max_results": limit,
                  "sortBy": "relevance"}
        return self.parse(self.http.get_text(self.name, self.URL, params))

    @staticmethod
    def parse(xml_text: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_text)
        out = []
        for e in root.findall("a:entry", ATOM):
            arxiv_url = (e.findtext("a:id", default="", namespaces=ATOM) or "").strip()
            title = strip_markup(e.findtext("a:title", default="", namespaces=ATOM))
            if not title or "/abs/" not in arxiv_url:
                continue  # arXiv returns an "Error" pseudo-entry for bad queries
            published = e.findtext("a:published", default="", namespaces=ATOM)
            # Strip the version suffix (v1, v2...): citations should point at
            # the work, and the unversioned URL always resolves to the latest.
            arxiv_id = re.sub(r"v\d+$", "", arxiv_url.rsplit("/abs/", 1)[-1])
            out.append({
                "origin": "arxiv",
                "external_id": arxiv_id,
                "doi": normalise_doi(e.findtext("arxiv:doi", default=None, namespaces=ATOM)),
                "title": title,
                "authors": [a.findtext("a:name", default="", namespaces=ATOM).strip()
                            for a in e.findall("a:author", ATOM)],
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "venue": "arXiv preprint",
                "abstract": strip_markup(e.findtext("a:summary", default="", namespaces=ATOM)),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "fetched_at": now_iso(),
            })
        return out


class OpenAlexClient:
    name = "openalex"
    URL = "https://api.openalex.org/works"
    FIELDS = "id,doi,display_name,publication_year,authorships,primary_location,abstract_inverted_index"

    def __init__(self, http: ScholarlyHTTP):
        self.http = http

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        # has_abstract:true because the Critic can only verify a claim against
        # text it can read; a title-only record is unverifiable by design.
        params = {"search": query, "per-page": limit, "filter": "has_abstract:true", "select": self.FIELDS}
        if self.http.contact_email:
            params["mailto"] = self.http.contact_email
        return self.parse(self.http.get_json(self.name, self.URL, params))

    @staticmethod
    def parse(data: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for w in data.get("results", []):
            if not w.get("display_name"):
                continue
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            out.append({
                "origin": "openalex",
                "external_id": (w.get("id") or "").rsplit("/", 1)[-1],
                "doi": normalise_doi(w.get("doi")),
                "title": strip_markup(w["display_name"]),
                "authors": [(a.get("author") or {}).get("display_name", "")
                            for a in w.get("authorships", [])][:20],
                "year": w.get("publication_year"),
                "venue": src.get("display_name"),
                "abstract": reconstruct_inverted_abstract(w.get("abstract_inverted_index")),
                "url": loc.get("landing_page_url") or w.get("doi") or w.get("id"),
                "fetched_at": now_iso(),
            })
        return out


class CrossrefClient:
    name = "crossref"
    URL = "https://api.crossref.org/works"

    def __init__(self, http: ScholarlyHTTP):
        self.http = http

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        params = {"query.bibliographic": query, "rows": limit, "filter": "has-abstract:true",
                  "select": "DOI,title,author,issued,container-title,abstract,URL"}
        if self.http.contact_email:
            params["mailto"] = self.http.contact_email
        return self.parse(self.http.get_json(self.name, self.URL, params))

    @staticmethod
    def parse(data: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for it in (data.get("message") or {}).get("items", []):
            titles = it.get("title") or []
            if not titles:
                continue
            parts = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
            authors = []
            for a in it.get("author", [])[:20]:
                name = " ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name", "")
                authors.append(name)
            venue = (it.get("container-title") or [None])[0]
            out.append({
                "origin": "crossref",
                "external_id": it.get("DOI"),
                "doi": normalise_doi(it.get("DOI")),
                "title": strip_markup(titles[0]),
                "authors": authors,
                "year": parts[0] if parts and isinstance(parts[0], int) else None,
                "venue": venue,
                "abstract": strip_markup(it.get("abstract")),
                "url": it.get("URL"),
                "fetched_at": now_iso(),
            })
        return out
