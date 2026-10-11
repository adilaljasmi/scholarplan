"""Critic: acceptance rule, conservative defaults, citation pre-check, feedback."""

from __future__ import annotations

import pytest

from scholarplan.agents.critic import CriticAgent, citation_problems, subtask_passes
from scholarplan.llm import ScriptedLLM


@pytest.mark.parametrize("supported, total, expected", [
    (2, 2, True), (1, 2, True), (1, 3, False), (0, 2, False), (0, 0, False),
])
def test_subtask_acceptance_rule(supported, total, expected):
    assert subtask_passes(supported, total, min_ratio=0.5) is expected


def test_citation_precheck():
    assert citation_problems("A [C1]. B [C2].", {1, 2}) == []
    assert citation_problems("A [C1]. B [C9].", {1, 2}) == ["cites unknown claim C9"]
    assert citation_problems("No citations here.", {1}) == ["contains no [C#] citations"]


def _board_with_claims(bb, texts):
    sid = bb.add_subtask(1, "q", status="processed")
    src = bb.add_source(sid, 0, {"origin": "arxiv", "title": "T", "authors": [],
                                 "abstract": "Agents call tools during reasoning."})
    ids = [bb.add_claim(sid, src, 0, t) for t in texts]
    return sid, ids


def test_missing_verdict_defaults_to_unsupported(settings, bb):
    sid, ids = _board_with_claims(bb, ["Agents call tools.", "Agents call tools during reasoning."])
    only_one = {"verdicts": [{"claim_id": ids[0], "label": "supported", "reason": "stated"}]}
    CriticAgent(settings, ScriptedLLM({"verify_claims": [only_one]})).act(bb)
    statuses = {c["id"]: c["status"] for c in bb.claims()}
    assert statuses == {ids[0]: "supported", ids[1]: "unsupported"}
    assert bb.get_subtask(sid)["status"] == "verified"  # 1/2 meets the 50% rule


def test_rejection_writes_feedback_for_planner(settings, bb):
    sid, ids = _board_with_claims(bb, ["Agents cure cancer.", "Agents are 99% accurate."])
    verdicts = {"verdicts": [{"claim_id": i, "label": "unsupported", "reason": "not in source"} for i in ids]}
    CriticAgent(settings, ScriptedLLM({"verify_claims": [verdicts]})).act(bb)
    assert bb.get_subtask(sid)["status"] == "rejected"
    fb = bb.feedback(unconsumed_only=True)
    assert fb[0]["kind"] == "unsupported" and "0/2" in fb[0]["message"]
    assert len(bb.verdicts()) == 2  # every judgement is recorded for audit


def test_critic_llm_failure_rejects_rather_than_accepts(settings, bb):
    sid, _ = _board_with_claims(bb, ["Agents call tools."])
    CriticAgent(settings, ScriptedLLM({"verify_claims": ["garbage"] * 3})).act(bb)
    assert bb.get_subtask(sid)["status"] == "rejected"


def test_synthesis_with_unknown_citation_rejected_without_llm(settings, bb):
    sid, ids = _board_with_claims(bb, ["Agents call tools."])
    bb.record_claim_verdict(ids[0], "supported", "ok")
    bb.set_subtask_status(sid, "verified")
    bb.add_synthesis(1, f"Agents call tools [C{ids[0]}] and fly [C999].")
    llm = ScriptedLLM()
    CriticAgent(settings, llm).act(bb)
    assert bb.syntheses()[0]["status"] == "rejected"
    assert not any(c["task"] == "verify_synthesis" for c in llm.calls)  # cheap check short-circuited
    assert bb.feedback(unconsumed_only=True)[0]["kind"] == "synthesis_rejected"
