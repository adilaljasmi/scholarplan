"""Critic: independent verification of claims and of the final synthesis.

Trigger: a subtask in state ``processed``, or a synthesis in state ``pending``.

Why a *separate* model: LLMs are poor at checking their own plans and output
(Kambhampati et al., 2024), so the Critic uses a different model family
(Llama-3.3-70B, as chosen in Unit 6) from the one that wrote the claims
(gpt-oss). It judges entailment against the retrieved text only, which is
the grounding principle of RAG (Lewis et al., 2020) and the task measured by
the LLM-AggreFact benchmark used to select the model (Tang, Laban and
Durrett, 2024).

Two layers of checking, cheapest first:
  1. deterministic: a synthesis that cites a claim id which was never
     verified is rejected without spending an LLM call;
  2. LLM entailment judgement for everything else.
Missing verdicts default to "unsupported": for a hallucination guard, a
false negative (dropping a true claim) is far cheaper than a false positive
(publishing a fabricated one).
"""

from __future__ import annotations

import re
from typing import Any

from scholarplan.agents import prompts
from scholarplan.agents.base import Agent
from scholarplan.blackboard import Blackboard
from scholarplan.config import Settings
from scholarplan.llm import LLMClient, LLMError, call_json
from scholarplan.llm.schemas import CriticOutput, SynthesisVerdict

# Tolerates spacing such as "[ C27]": a live synthesis used that form, and the
# strict pattern left the citation unrendered and unchecked (remediation R20).
CITATION = re.compile(r"\[\s*C\s*(\d+)\s*\]")


def subtask_passes(n_supported: int, n_total: int, min_ratio: float) -> bool:
    """Acceptance rule for a subtask (unit-tested in isolation)."""
    return n_total > 0 and n_supported >= 1 and n_supported / n_total >= min_ratio


def citation_problems(text: str, verified_ids: set[int]) -> list[str]:
    """Deterministic pre-check of a synthesis' citations."""
    cited = {int(x) for x in CITATION.findall(text)}
    problems = [f"cites unknown claim C{c}" for c in sorted(cited - verified_ids)]
    if verified_ids and not cited:
        problems.append("contains no [C#] citations")
    return problems


class CriticAgent(Agent):
    role = "critic"
    uses_llm = True

    def __init__(self, settings: Settings, llm: LLMClient):
        super().__init__(settings)
        self.llm = llm

    def can_act(self, bb: Blackboard) -> bool:
        return bool(bb.subtasks("processed")) or any(s["status"] == "pending" for s in bb.syntheses())

    def act(self, bb: Blackboard) -> str:
        if bb.subtasks("processed"):
            return self._verify_subtask(bb)
        return self._verify_synthesis(bb)

    # ------------------------------------------------------------ claims
    def _verify_subtask(self, bb: Blackboard) -> str:
        sb = self.scoped(bb)
        st = sb.claim_subtask("processed", "verifying", owner=self.role)
        if st is None:
            return "no processed subtask"
        claims = sb.claims(subtask_id=st["id"], attempt=st["attempts"])
        sources = {s["id"]: s for s in sb.sources(subtask_id=st["id"], attempt=st["attempts"])}
        items = [{"claim_id": c["id"], "claim": c["text"],
                  "source_text": f"{sources[c['source_id']]['title']}. {sources[c['source_id']]['abstract']}"}
                 for c in claims]
        sb.log("action", f"verify {len(items)} claims of subtask {st['id']} against their sources",
               {"subtask_id": st["id"]})
        try:
            out = call_json(self.llm, CriticOutput, task="verify_claims", model=self.settings.models.critic,
                            system=prompts.VERIFY_CLAIMS, payload={"items": items},
                            temperature=self.settings.worker_temperature,
                            max_retries=self.settings.llm_max_retries)
            verdicts = {v.claim_id: v for v in out.verdicts}
        except LLMError as exc:
            sb.log("error", f"critic failed: {exc}")
            verdicts = {}

        supported, reasons = 0, []
        for c in claims:
            v = verdicts.get(c["id"])
            label = v.label if v else "unsupported"
            reason = (v.reason if v else "no verdict returned; rejected by default") or ""
            sb.record_claim_verdict(c["id"], label, reason)
            sb.log("verdict", f"C{c['id']} {label.upper()}: {reason}",
                   {"claim_id": c["id"], "subtask_id": st["id"], "label": label})
            if label == "supported":
                supported += 1
            else:
                reasons.append(f"\"{c['text'][:90]}\" -> {reason}")

        if subtask_passes(supported, len(claims), self.settings.min_supported_ratio):
            sb.set_subtask_status(st["id"], "verified")
            sb.log("observation", f"subtask {st['id']} VERIFIED ({supported}/{len(claims)} claims supported)",
                   {"subtask_id": st["id"]})
            return f"subtask {st['id']}: verified {supported}/{len(claims)}"

        msg = (f"Only {supported}/{len(claims)} claims for \"{st['query']}\" were supported by their "
               f"sources (need >= {self.settings.min_supported_ratio:.0%}). Problems: " + " | ".join(reasons[:3]))
        sb.add_feedback(self.role, "unsupported", msg, subtask_id=st["id"])
        sb.set_subtask_status(st["id"], "rejected")
        sb.log("feedback", msg, {"subtask_id": st["id"]})
        return f"subtask {st['id']}: rejected {supported}/{len(claims)}"

    # --------------------------------------------------------- synthesis
    def _verify_synthesis(self, bb: Blackboard) -> str:
        sb = self.scoped(bb)
        syn = next(s for s in sb.syntheses() if s["status"] == "pending")
        verified = sb.claims(status="supported")
        verified_ids = {c["id"] for c in verified}
        problems = citation_problems(syn["text"], verified_ids)
        if problems:
            label, reason = "unsupported", "; ".join(problems)
        else:
            payload: dict[str, Any] = {"synthesis": syn["text"],
                                       "claims": [{"claim_id": c["id"], "text": c["text"]} for c in verified]}
            try:
                v = call_json(self.llm, SynthesisVerdict, task="verify_synthesis",
                              model=self.settings.models.critic, system=prompts.VERIFY_SYNTHESIS,
                              payload=payload, temperature=self.settings.worker_temperature,
                              max_retries=self.settings.llm_max_retries)
                label, reason = v.label, v.reason
            except LLMError as exc:
                label, reason = "unsupported", f"critic failed: {exc}"
        sb.record_synthesis_verdict(syn["id"], label, reason)
        sb.log("verdict", f"synthesis #{syn['attempt']} {label.upper()}: {reason}",
               {"synthesis_id": syn["id"], "label": label})
        if label != "supported":
            sb.add_feedback(self.role, "synthesis_rejected", reason)
        return f"synthesis {label}"
