"""Scholarly API clients: parsing recorded responses, politeness, caching, retries."""

from __future__ import annotations

import json

import httpx
import pytest

from scholarplan.sources.clients import ArxivClient, CrossrefClient, OpenAlexClient
from scholarplan.sources.http import RateLimiter, ScholarlyHTTP
from scholarplan.sources.records import (keywords, normalise_doi, normalise_title,
                                         reconstruct_inverted_abstract, strip_markup)
from tests.conftest import FIXTURES


def test_arxiv_parse_recorded_response():
    recs = ArxivClient.parse((FIXTURES / "arxiv_react.xml").read_text(encoding="utf-8"))
    assert recs[0]["external_id"] == "2210.03629"  # ReAct (Yao et al., 2023)
    assert recs[0]["title"].startswith("ReAct")
    assert recs[0]["url"] == "https://arxiv.org/abs/2210.03629"  # version suffix removed
    assert recs[0]["year"] == 2022 and recs[0]["authors"][0] == "Shunyu Yao"
    assert all(r["abstract"] and "\n" not in r["abstract"] for r in recs)


def test_arxiv_query_is_boolean_over_content_words():
    assert ArxivClient.build_query("How do agents use the ReAct method") == \
        "all:agents AND all:react AND all:method"


def test_arxiv_error_entry_is_ignored():
    xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#x</id>'
           '<title>Error</title><summary>bad query</summary></entry></feed>')
    assert ArxivClient.parse(xml) == []


def test_openalex_parse_rebuilds_inverted_abstract():
    data = json.loads((FIXTURES / "openalex_blackboard.json").read_text(encoding="utf-8"))
    recs = OpenAlexClient.parse(data)
    assert len(recs) == 3
    assert all(r["doi"] and r["doi"].startswith("10.") for r in recs)
    assert all(len(r["abstract"]) > 100 for r in recs)


def test_crossref_parse_strips_jats_markup():
    data = json.loads((FIXTURES / "crossref_hallucination.json").read_text(encoding="utf-8"))
    recs = CrossrefClient.parse(data)
    ji = recs[0]
    assert ji["doi"] == "10.1145/3571730" and ji["authors"][0] == "Ziwei Ji"  # Ji et al. (2023)
    assert "<jats" not in ji["abstract"] and not ji["abstract"].lower().startswith("abstract")


@pytest.mark.parametrize("raw, expected", [
    ("https://doi.org/10.1145/ABC", "10.1145/abc"),
    ("http://dx.doi.org/10.1/x", "10.1/x"),
    ("doi:10.1/X ", "10.1/x"),
    ("", None), (None, None),
])
def test_normalise_doi(raw, expected):
    assert normalise_doi(raw) == expected


def test_title_normalisation_and_keywords():
    assert normalise_title("ReAct: Synergizing  Reasoning!") == "react synergizing reasoning"
    assert keywords("What is the role of a blackboard in multi-agent systems?") == \
        ["role", "blackboard", "multi-agent", "systems"]


def test_inverted_abstract_and_markup():
    assert reconstruct_inverted_abstract({"world": [1], "hello": [0]}) == "hello world"
    assert reconstruct_inverted_abstract(None) is None
    assert strip_markup("<jats:p>Abstract: Some &amp; text</jats:p>") == "Some & text"


def test_rate_limiter_enforces_spacing():
    t = {"now": 0.0}
    slept: list[float] = []
    lim = RateLimiter(3.0, clock=lambda: t["now"], sleep=lambda s: slept.append(s))
    lim.wait()
    t["now"] = 1.0
    lim.wait()  # only 1 s since last call: must wait 2 s (arXiv's 3-second rule)
    assert slept == [pytest.approx(2.0)]


def _transport(handler):
    return httpx.MockTransport(handler)


def test_http_cache_avoids_second_network_call(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, text="payload")

    http = ScholarlyHTTP(tmp_path, transport=_transport(handler))
    assert http.get_text("x", "https://api.test/a", {"q": 1}) == "payload"
    assert http.get_text("x", "https://api.test/a", {"q": 1}) == "payload"
    assert calls["n"] == 1 and http.cache_hits == 1


def test_http_retries_transient_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("scholarplan.sources.http.time.sleep", lambda s: None)
    responses = iter([httpx.Response(503), httpx.Response(200, text="ok")])
    http = ScholarlyHTTP(None, transport=_transport(lambda r: next(responses)))
    assert http.get_text("x", "https://api.test/b", {}) == "ok"


def test_http_does_not_retry_client_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("scholarplan.sources.http.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    http = ScholarlyHTTP(None, transport=_transport(handler))
    with pytest.raises(RuntimeError):
        http.get_text("x", "https://api.test/c", {})
    assert calls["n"] == 1


def test_user_agent_identifies_client(tmp_path):
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, text="{}")

    ScholarlyHTTP(None, contact_email="me@example.org", transport=_transport(handler)).get_text("x", "https://a.test", {})
    assert "ScholarPlan/" in seen["ua"] and "mailto:me@example.org" in seen["ua"]
