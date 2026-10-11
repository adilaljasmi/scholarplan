"""Processor: measurable de-duplication / relevance thresholds and fabricated-citation guard."""

from __future__ import annotations

from scholarplan.agents.processor import ProcessorAgent, select_sources
from scholarplan.llm import ScriptedLLM

LONG = " This sentence pads the abstract so that it passes the minimum usable length check."


def src(i, title, abstract, doi=None):
    return {"id": i, "title": title, "abstract": abstract + LONG, "doi": doi, "authors": [], "year": 2024}


def test_doi_duplicates_removed_case_insensitively(settings, embedder):
    cands = [src(1, "Agents that plan", "language agents plan tasks", doi="10.1/ABC"),
             src(2, "Agents that plan (published)", "language agents plan tasks", doi="https://doi.org/10.1/abc")]
    d = select_sources("language agents plan", cands, [], embedder, settings)
    assert d[2]["duplicate_of"] == 1 and d[1]["selected"]


def test_title_duplicates_removed(settings, embedder):
    cands = [src(1, "ReAct: Reasoning, Acting!", "react agents reasoning acting"),
             src(2, "react   reasoning ACTING", "something else entirely about react agents")]
    cands[0]["title"] = "ReAct Reasoning Acting"
    d = select_sources("react reasoning acting", cands, [], embedder, settings)
    assert d[2]["duplicate_of"] == 1


def test_embedding_near_duplicates_use_threshold(settings, embedder):
    text = "blackboard architecture for multi agent coordination and control"
    cands = [src(1, "Blackboard coordination", text), src(2, "Blackboard coordination study", text)]
    assert select_sources("blackboard coordination", cands, [], embedder, settings)[2]["duplicate_of"] == 1
    strict = settings.with_overrides(dedup_similarity=1.01)  # impossible threshold: nothing is a duplicate
    assert select_sources("blackboard coordination", cands, [], embedder, strict)[2]["duplicate_of"] is None


def test_relevance_floor_filters_noise(settings, embedder):
    cands = [src(1, "Language agent planning", "language agents planning tasks reasoning"),
             src(2, "Soil moisture in vineyards", "irrigation of grapes in dry climates")]
    d = select_sources("language agent planning", cands, [], embedder, settings)
    assert d[1]["selected"] and not d[2]["selected"] and "relevance" in d[2]["reason"]


def test_top_k_and_missing_abstract(settings, embedder):
    # Topics must differ: near-identical abstracts would (correctly) be
    # removed as near-duplicates before top-k is applied (remediation R9).
    topics = ["robot navigation", "theorem proving", "web shopping", "medical triage", "code repair"]
    cands = [{"id": i, "title": f"Agents planning for {t}", "doi": None, "authors": [],
              "abstract": f"Agents use planning for {t}. Planning agents in {t} tasks are evaluated "
                          f"on {t} benchmarks with several baselines and ablations."}
             for i, t in enumerate(topics, start=1)]
    cands.append({"id": 9, "title": "agents planning", "abstract": None, "doi": None, "authors": []})
    d = select_sources("agents planning", cands, [], embedder, settings.with_overrides(top_k_sources=2))
    assert sum(x["selected"] for x in d.values()) == 2
    assert d[9]["reason"] == "no usable abstract"


def test_sources_used_by_earlier_subtasks_are_duplicates(settings, embedder):
    used = [src(7, "Shared paper", "shared content about agents", doi="10.9/shared")]
    cands = [src(1, "Shared paper", "shared content about agents", doi="10.9/shared")]
    assert select_sources("agents", cands, used, embedder, settings)[1]["duplicate_of"] == 7


def _prepared_board(bb):
    sid = bb.add_subtask(1, "language agents planning", status="retrieved")
    sources = [bb.add_source(sid, 0, {"origin": "arxiv", "title": f"Language agents planning {i}",
                                      "authors": ["A B"], "abstract": "Language agents perform planning. "
                                      "Planning helps language agents." + LONG}) for i in range(2)]
    return sid, sources


def test_processor_drops_claims_for_unknown_sources(settings, embedder, bb):
    sid, sources = _prepared_board(bb)
    fabricated = {"summaries": [{"source_id": 999, "summary": "invented summary text", "claims": ["fake"]},
                                {"source_id": sources[0], "summary": "Language agents perform planning.",
                                 "claims": ["Language agents perform planning."]}]}
    ProcessorAgent(settings, ScriptedLLM({"summarise": [fabricated]}), embedder).act(bb)
    claims = bb.claims(subtask_id=sid)
    assert [c["source_id"] for c in claims] == [sources[0]]
    assert any("unknown source_id 999" in t["content"] for t in bb.trace())
    assert bb.get_subtask(sid)["status"] == "processed"


def test_processor_caps_claims_per_source(settings, embedder, bb):
    sid, sources = _prepared_board(bb)
    many = {"summaries": [{"source_id": sources[0], "summary": "a summary here",
                           "claims": ["c1", "c2", "c3", "c4"]}]}
    ProcessorAgent(settings, ScriptedLLM({"summarise": [many]}), embedder).act(bb)
    assert len(bb.claims(subtask_id=sid)) == settings.max_claims_per_source


def test_processor_rejects_when_nothing_relevant(settings, embedder, bb):
    sid = bb.add_subtask(1, "quantum chromodynamics lattice", status="retrieved")
    bb.add_source(sid, 0, {"origin": "arxiv", "title": "Grape irrigation", "authors": [],
                           "abstract": "Vineyards need irrigation in dry summers." + LONG})
    ProcessorAgent(settings, ScriptedLLM(), embedder).act(bb)
    assert bb.get_subtask(sid)["status"] == "rejected"
    assert bb.feedback()[0]["kind"] == "no_relevant"
