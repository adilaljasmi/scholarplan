"""Functional tests: the whole multi-agent system, end to end, offline.

Each scenario forces one behaviour promised in the Unit 6 design or its
feedback, and checks it through the *outputs a user would see* (run status,
report, trace), not through internals.
"""

from __future__ import annotations

import json
import re

import pytest

from scholarplan.agents.archivist import search_index
from scholarplan.controller import Controller
from scholarplan.embeddings import HashingEmbedder
from scholarplan.llm import MockLLM, ScriptedLLM
from tests.conftest import FakeClient

pytestmark = pytest.mark.functional

GOAL = "How do language model agents use react reasoning, evaluation and challenges"


def plan(*queries):
    return {"thought": "decompose", "subtasks": [{"query": q, "rationale": "facet"} for q in queries]}


def run(settings, llm, clients, goal=GOAL, **kw):
    ctl = Controller(settings, llm=llm, clients=clients, embedder=HashingEmbedder(), **kw)
    return ctl, ctl.run(goal)


def kinds(ctl):
    return [(t["agent"], t["kind"]) for t in ctl.bb.trace()]


def test_happy_path_produces_verified_report(settings, fake_clients):
    llm = ScriptedLLM({"plan": [plan("react agents", "agents evaluation", "agents challenges")]})
    ctl, res = run(settings, llm, fake_clients)

    assert res.status == "done"
    out = res.output_dir
    for f in ("report.md", "references.bib", "index.sqlite", "trace.jsonl", "run_summary.json"):
        assert (out / f).exists(), f
    s = res.summary
    assert s["subtasks"] == {"verified": 3}
    assert s["steps_used"] <= settings.step_budget
    assert s["claims_supported"] > 0 and s["claims_unsupported"] == 0

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Answer" in report and "## References" in report
    assert re.search(r"\(Example, 2023[a-z]?\)", report)  # Harvard in-text citation rendered

    # Every agent role took part, and the Planner followed Thought -> Action.
    roles = {a for a, _ in kinds(ctl)}
    assert {"planner", "retriever", "processor", "critic", "archivist"} <= roles
    seq = [k for a, k in kinds(ctl) if a == "planner"]
    assert seq[:2] == ["thought", "action"]

    # Trace on disk matches the blackboard (auditability).
    lines = (out / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(ctl.bb.trace()) - 1  # archivist's final log line is written after export
    assert json.loads(lines[0])["kind"] == "event"


def test_critic_rejection_triggers_reflexion_replanning(settings, fake_clients):
    """Unit 6 Figure 2 dashed path: reject -> Feedback -> Planner re-plans -> verified."""
    def reject_all(p):
        return {"verdicts": [{"claim_id": it["claim_id"], "label": "unsupported",
                              "reason": "overstates the source"} for it in p["items"]]}

    def refine_on_feedback(p):
        if not p["feedback"]:
            return None  # routine step: let the default policy issue the subtask
        return {"thought": "The Critic says the claims overstate the source; narrow to tool use.",
                "action": "refine_subtask", "subtask_id": p["feedback"][0]["subtask_id"],
                "query": "react agents tools"}

    llm = ScriptedLLM({"plan": [plan("react agents")], "verify_claims": [reject_all],
                       "react": [refine_on_feedback]})
    ctl, res = run(settings, llm, fake_clients)

    assert res.status == "done"
    st = ctl.bb.subtasks()[0]
    assert st["status"] == "verified" and st["attempts"] == 1  # succeeded on the re-plan
    fb = ctl.bb.feedback()
    assert fb[0]["author"] == "critic" and fb[0]["consumed"] == 1
    k = kinds(ctl)
    i = k.index(("critic", "feedback"))
    assert ("planner", "thought") in k[i:] and ("planner", "action") in k[i:]
    # Rejected claims never reach the report.
    # Rejected claims never reach the report. Compare by claim id: the re-plan
    # may legitimately re-extract the same sentence, which is then a new,
    # independently verified claim with a new id.
    rejected_ids = [c["id"] for c in ctl.bb.claims(status="unsupported")]
    report = (res.output_dir / "report.md").read_text(encoding="utf-8")
    assert rejected_ids and not any(f"| C{i} |" in report or f"C{i}</sup>" in report for i in rejected_ids)


def test_replan_cap_marks_subtask_unresolvable(settings, fake_clients):
    """Unit 6 feedback #3: at most three re-plans, then 'unresolvable'; the run still completes."""
    def reject_all(p):
        return {"verdicts": [{"claim_id": it["claim_id"], "label": "unsupported", "reason": "no"}
                             for it in p["items"]]}

    refine_queries = iter(["react agents tools", "react agents question answering", "react agents search"])

    def keep_refining(p):
        if not p["feedback"]:
            return None
        fb = p["feedback"][0]
        return {"thought": "try another phrasing", "action": "refine_subtask",
                "subtask_id": fb["subtask_id"], "query": next(refine_queries)}

    llm = ScriptedLLM({"plan": [plan("react agents", "agents evaluation")],
                       "verify_claims": [reject_all] * 4, "react": [keep_refining] * 3})
    ctl, res = run(settings, llm, fake_clients)

    assert res.status == "done"
    first, second = ctl.bb.subtasks()
    assert first["status"] == "unresolvable" and first["attempts"] == settings.max_replans_per_subtask
    assert second["status"] == "verified"
    assert any("re-plan cap" in t["content"] for t in ctl.bb.trace())
    assert "unresolvable" in (res.output_dir / "report.md").read_text(encoding="utf-8")


def test_no_results_feedback_leads_to_alternate_source(settings, fake_clients):
    def alt(p):
        if not p["feedback"]:
            return None
        return {"thought": "Nothing found everywhere; try OpenAlex with a known phrasing.",
                "action": "refine_subtask", "subtask_id": p["feedback"][0]["subtask_id"],
                "query": "react agents"}

    llm = ScriptedLLM({"plan": [plan("zzz nonexistent topic")], "react": [alt]})
    ctl, res = run(settings, llm, fake_clients)
    assert res.status == "done"
    fb = ctl.bb.feedback()[0]
    assert fb["author"] == "retriever" and fb["kind"] == "no_results"
    assert ctl.bb.subtasks()[0]["status"] == "verified"


def test_step_budget_bounds_the_run(settings, fake_clients):
    small = settings.with_overrides(step_budget=3)
    llm = ScriptedLLM({"plan": [plan("react agents", "agents evaluation", "agents challenges")]})
    ctl, res = run(small, llm, fake_clients)
    assert res.status == "done"
    assert res.summary["steps_used"] <= 3
    statuses = [s["status"] for s in ctl.bb.subtasks()]
    assert "abandoned" in statuses  # unfinished work abandoned, not looped on
    assert any("budget" in t["content"] for t in ctl.bb.trace())


def test_injected_hallucination_is_filtered(settings, fake_clients):
    """A fabricated claim from the Processor must be caught by the (independent) Critic."""
    def summarise_with_fabrication(p):
        first = p["sources"][0]
        return {"summaries": [{"source_id": first["source_id"], "summary": "summary of the source",
                               "claims": ["Language model agents interleave reasoning traces and actions.",
                                          "Language model agents achieve 99 percent accuracy on every benchmark."]}]}

    llm = ScriptedLLM({"plan": [plan("react agents")], "summarise": [summarise_with_fabrication]})
    ctl, res = run(settings, llm, fake_clients)
    assert res.status == "done"
    unsupported = [c["text"] for c in ctl.bb.claims(status="unsupported")]
    assert any("99 percent" in u for u in unsupported)
    assert "99 percent" not in (res.output_dir / "report.md").read_text(encoding="utf-8")


def test_synthesis_rejection_then_fallback(settings, fake_clients):
    bad = {"text": "Agents are perfect and need no verification [C999]."}
    llm = ScriptedLLM({"plan": [plan("react agents")], "synthesise": [bad, bad]})
    ctl, res = run(settings, llm, fake_clients)
    assert res.status == "done"
    syn = ctl.bb.syntheses()
    assert [s["status"] for s in syn] == ["rejected", "rejected", "fallback"]
    report = (res.output_dir / "report.md").read_text(encoding="utf-8")
    assert "Verified findings" in report and "C999" not in report


def test_api_failure_is_isolated(settings):
    class Broken(FakeClient):
        def search(self, query, limit):
            raise RuntimeError("HTTP 503")

    clients = {"arxiv": Broken("arxiv"), "openalex": FakeClient("openalex"), "crossref": FakeClient("crossref")}
    llm = ScriptedLLM({"plan": [plan("react agents", "agents challenges")]})
    ctl, res = run(settings, llm, clients)
    assert res.status == "done"
    assert any(t["kind"] == "error" and "503" in t["content"] for t in ctl.bb.trace())


def test_default_mock_backend_runs_end_to_end(settings, fake_clients):
    """The configuration a marker without an API key would use."""
    ctl, res = run(settings, MockLLM(), fake_clients,
                   goal="language model agents react evaluation challenges")
    assert res.status == "done" and res.summary["synthesis_status"] in ("verified", "fallback")


def test_cli_run_in_mock_mode(settings, fake_clients, monkeypatch, capsys):
    from scholarplan import cli
    monkeypatch.setattr("scholarplan.controller.build_clients", lambda s: fake_clients)
    code = cli.main(["run", "react agents evaluation challenges", "--llm", "mock",
                     "--out", str(settings.output_dir), "--embeddings", "hashing"])
    out = capsys.readouterr().out
    assert code == 0
    assert "planner" in out and "critic" in out and "Outputs:" in out


def test_saved_index_supports_later_semantic_search(settings, fake_clients):
    """The offline store is usable after the run: query the vector index directly."""
    _, res = run(settings, ScriptedLLM({"plan": [plan("react agents")]}), fake_clients)
    hits = search_index(res.output_dir / "index.sqlite", "reasoning and acting agents", HashingEmbedder(), k=3)
    assert hits and hits[0]["citation"].startswith("(")
