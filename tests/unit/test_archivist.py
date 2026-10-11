"""Archivist: Harvard formatting, BibTeX, vector index round-trip."""

from __future__ import annotations

from scholarplan.agents.archivist import (bibtex_entry, bibtex_key, build_index, disambiguate_years,
                                          harvard_intext, harvard_reference, search_index)


def s(authors, year=2023, **kw):
    base = {"title": "ReAct: Synergizing Reasoning and Acting", "authors": authors, "year": year,
            "origin": "openalex", "doi": "10.1/x", "venue": "ICLR", "url": "https://x", "external_id": "x"}
    return base | kw


def test_harvard_intext_variants():
    assert harvard_intext(s(["Shunyu Yao"])) == "(Yao, 2023)"
    assert harvard_intext(s(["Ian Sommerville", "Jane Doe"])) == "(Sommerville and Doe, 2023)"
    assert harvard_intext(s(["Shunyu Yao", "J Zhao", "D Yu"])) == "(Yao et al., 2023)"
    assert harvard_intext(s([], year=None)).endswith("n.d.)")


def test_same_author_year_gets_letters():
    a, b = s(["L Wang", "X Y", "P Q"], title="Beta paper"), s(["R Wang", "Q Z", "R S"], title="Alpha paper")
    disambiguate_years([a, b])
    assert harvard_intext(b) == "(Wang et al., 2023a)" and harvard_intext(a) == "(Wang et al., 2023b)"


def test_harvard_reference_format():
    ref = harvard_reference(s(["Shunyu Yao", "Jeffrey Zhao"]), "11 October 2026")
    assert ref == ("Yao, S. and Zhao, J. (2023) 'ReAct: Synergizing Reasoning and Acting', ICLR. "
                   "Available at: https://doi.org/10.1/x (Accessed: 11 October 2026).")
    assert "et al." in harvard_reference(s(["A B", "C D", "E F", "G H"]), "x")


def test_bibtex_arxiv_vs_article():
    arx = s(["Shunyu Yao"], origin="arxiv", doi=None, external_id="2210.03629")
    assert bibtex_key(arx) == "yao2023react"
    entry = bibtex_entry(arx, "yao2023react")
    assert entry.startswith("@misc{yao2023react") and "eprint = {2210.03629}" in entry
    art = bibtex_entry(s(["Shunyu Yao"]), "k")
    assert art.startswith("@article{k") and "doi = {10.1/x}" in art and "journal = {ICLR}" in art


def test_vector_index_round_trip(tmp_path, embedder):
    items = [{"kind": "claim", "text": "blackboard systems coordinate agents through shared memory",
              "citation": "(Hayes-Roth, 1985)", "link": "a"},
             {"kind": "claim", "text": "retrieval augmented generation grounds answers in documents",
              "citation": "(Lewis et al., 2020)", "link": "b"}]
    path = tmp_path / "index.sqlite"
    assert build_index(path, items, embedder) == 2
    hits = search_index(path, "shared blackboard memory for agents", embedder, k=2)
    assert hits[0]["citation"] == "(Hayes-Roth, 1985)"
    assert hits[0]["distance"] <= hits[1]["distance"]


def test_citation_rendering_merges_adjacent_and_accepts_spacing():
    """Regression for R20: '[ C27]' was left raw and adjacent markers ran together."""
    from scholarplan.agents.archivist import render_citations
    sources = {1: s(["Lisa Robert", "Ann Sanyal"], title="A"), 2: s(["Bo Fu", "C D", "E F"], title="B")}
    claims = [{"id": 20, "source_id": 1}, {"id": 21, "source_id": 1}, {"id": 25, "source_id": 2}]
    out = render_citations("Better by 23%[C20][C21]. Also [ C25], and [C99].", claims, sources)
    assert "23% (Robert and Sanyal, 2023)<sup>C20, C21</sup>." in out
    assert "Also (Fu et al., 2023)<sup>C25</sup>," in out
    assert "[C99]" in out  # unknown ids stay visible rather than being dropped


def test_critic_counts_spaced_citations():
    from scholarplan.agents.critic import citation_problems
    assert citation_problems("Claim [ C3].", {3}) == []
