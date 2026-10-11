"""Planner: allowed-action computation, validation, fallback rules, caps and budget."""

from __future__ import annotations

from scholarplan.agents.planner import PlannerAgent, allowed_actions, coverage, default_action
from scholarplan.llm import ScriptedLLM


def st(i, status, attempts=0, query="q"):
    return {"id": i, "status": status, "attempts": attempts, "query": query, "strategy": "all"}


def test_coverage():
    assert coverage([st(1, "verified"), st(2, "pending")]) == 0.5
    assert coverage([]) == 0.0


def test_allowed_actions_feedback_first():
    acts = allowed_actions([st(1, "rejected"), st(2, "pending")], {1}, 0.75)
    assert acts == ["refine_subtask", "alternate_source", "abandon_subtask"]


def test_allowed_actions_synthesis_gated_by_coverage():
    assert allowed_actions([st(1, "verified"), st(2, "pending")], set(), 0.75) == ["issue_subtask"]
    assert allowed_actions([st(1, "verified"), st(2, "verified"), st(3, "verified"), st(4, "pending")],
                           set(), 0.75) == ["issue_subtask", "synthesise"]
    # all terminal (even if not verified) -> synthesis allowed, nothing else to do
    assert allowed_actions([st(1, "abandoned"), st(2, "unresolvable")], set(), 0.75) == ["synthesise"]


def test_default_policy_cycles_sources_then_abandons():
    fb = [{"subtask_id": 1}]
    first = default_action([st(1, "rejected", attempts=0)], fb, ["refine_subtask", "alternate_source", "abandon_subtask"])
    late = default_action([st(1, "rejected", attempts=2)], fb, ["refine_subtask", "alternate_source", "abandon_subtask"])
    assert (first.action, first.source) == ("alternate_source", "openalex")
    assert late.action == "abandon_subtask"


def test_validate_rejects_illegal_and_repeated_moves(settings):
    p = PlannerAgent(settings, ScriptedLLM())
    p.query_history[1] = ["old query"]
    subs = [st(1, "rejected"), st(2, "pending")]
    allowed = ["refine_subtask", "alternate_source", "abandon_subtask"]
    from scholarplan.llm.schemas import ReActOutput as R
    assert "not in allowed" in p._validate(R(thought="t", action="synthesise"), allowed, subs, {1})
    assert "already tried" in p._validate(R(thought="t", action="refine_subtask", subtask_id=1,
                                            query="Old Query"), allowed, subs, {1})
    assert "rejected subtask" in p._validate(R(thought="t", action="abandon_subtask", subtask_id=2),
                                             allowed, subs, {1})
    assert p._validate(R(thought="t", action="refine_subtask", subtask_id=1, query="new query"),
                       allowed, subs, {1}) == ""


def test_plan_is_bounded_and_deduplicated(settings, bb):
    plan = {"thought": "t", "subtasks": [{"query": q} for q in
                                         ["alpha topic", "Alpha Topic", "beta topic", "gamma", "delta", "eps"]]}
    PlannerAgent(settings.with_overrides(max_subtasks=3), ScriptedLLM({"plan": [plan]})).act(bb)
    assert [s["query"] for s in bb.subtasks()] == ["alpha topic", "beta topic", "gamma"]
    assert bb.get_run()["status"] == "running" and bb.get_run()["steps_used"] == 1


def test_replan_cap_marks_unresolvable_without_llm(settings, bb):
    bb.set_run_status("running")
    sid = bb.add_subtask(1, "q", status="rejected")
    for _ in range(settings.max_replans_per_subtask):
        bb.set_subtask_status(sid, "rejected", bump_attempt=True)
    bb.add_feedback("critic", "unsupported", "still wrong", subtask_id=sid)
    llm = ScriptedLLM()
    PlannerAgent(settings, llm).act(bb)
    assert bb.get_subtask(sid)["status"] == "unresolvable"
    assert llm.calls == []  # a hard rule, not an LLM decision
    assert bb.feedback(unconsumed_only=True) == []


def test_budget_exhaustion_forces_synthesis(settings, bb):
    bb.set_run_status("running")
    bb.add_subtask(1, "q", status="pending")
    for _ in range(20):
        bb.consume_step()
    PlannerAgent(settings, ScriptedLLM()).act(bb)
    assert bb.subtasks()[0]["status"] == "abandoned"
    assert bb.syntheses()[0]["status"] == "fallback"  # no verified claims -> honest empty answer
    assert any("budget" in t["content"] for t in bb.trace())


def test_invalid_llm_decisions_fall_back_to_policy(settings, bb):
    bb.set_run_status("running")
    bb.add_subtask(1, "q", status="pending")
    bad = {"thought": "t", "action": "synthesise"}  # schema-valid but not allowed (coverage 0)
    llm = ScriptedLLM({"react": [bad] * 10})
    PlannerAgent(settings, llm).act(bb)
    assert bb.subtasks()[0]["status"] == "open"  # fallback issued the pending subtask
    assert any("fallback policy" in t["content"] for t in bb.trace())


def test_refine_records_new_query_and_attempt(settings, bb):
    bb.set_run_status("running")
    sid = bb.add_subtask(1, "too vague", status="rejected")
    bb.add_feedback("retriever", "no_results", "nothing found", subtask_id=sid)
    step = {"thought": "Narrow it", "action": "refine_subtask", "subtask_id": sid, "query": "specific terms"}
    PlannerAgent(settings, ScriptedLLM({"react": [step]})).act(bb)
    s = bb.get_subtask(sid)
    assert (s["status"], s["query"], s["attempts"]) == ("open", "specific terms", 1)
    kinds = [t["kind"] for t in bb.trace()]
    assert kinds[-2:] == ["thought", "action"]  # ReAct: Thought precedes Action


def test_planner_waits_while_work_in_flight(settings, bb):
    bb.set_run_status("running")
    bb.add_subtask(1, "q", status="retrieving")
    assert PlannerAgent(settings, ScriptedLLM()).can_act(bb) is False


def test_clean_query_normalises_typographic_hyphens():
    """Regression for R16: gpt-oss emitted U+2011 non-breaking hyphens in queries."""
    from scholarplan.agents.planner import clean_query
    assert clean_query("retrieval‑augmented  generation–based") == "retrieval-augmented generation-based"
