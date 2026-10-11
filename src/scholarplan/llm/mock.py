"""Deterministic stand-in for the LLMs.

Why a mock at all: LLM output is non-deterministic and the free API is rate
limited, so tests that call it would be slow, flaky and unrepeatable. The
Unit 6 plan was to unit-test deterministic parts conventionally and evaluate
the LLM parts separately with a harness. The mock makes the *orchestration*
(blackboard, triggers, re-planning, budgets) fully testable offline, and it
lets a marker run the whole system without an API key.

The mock is intentionally simple and transparent: its "planner" splits the
goal into keyword facets, its "processor" extracts leading sentences
(extractive summarisation cannot hallucinate), and its "critic" uses lexical
overlap. It is not a model of LLM quality, and the README says so.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Callable

from scholarplan.llm.base import Usage
from scholarplan.sources.records import keywords

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _payload(user: str) -> dict[str, Any]:
    start = user.find("INPUT:")
    body = user[start + len("INPUT:"):].lstrip() if start != -1 else user
    obj, _ = json.JSONDecoder().raw_decode(body)
    return obj


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if len(s.strip()) > 25]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))


class MockLLM:
    """Rule-based responses keyed by the ``task`` name."""

    def __init__(self) -> None:
        self.usage = Usage()
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, task: str, model: str, system: str, user: str,
                 temperature: float, json_mode: bool = True) -> str:
        self.calls.append({"task": task, "model": model})
        p = _payload(user)
        handler: Callable[[dict[str, Any]], dict[str, Any]] = getattr(self, f"_{task}")
        out = json.dumps(handler(p))
        # Rough token estimate (4 chars per token) so usage panels still work offline.
        self.usage.add(f"mock:{model}", len(user) // 4, len(out) // 4)
        return out

    # ------------------------------------------------------------- Planner
    def _plan(self, p: dict[str, Any]) -> dict[str, Any]:
        goal = p["goal"]
        kw = keywords(goal, limit=8)
        core = " ".join(kw[:4]) or goal
        facets = [core, f"{core} evaluation", f"{core} challenges limitations"]
        n = max(1, min(int(p.get("max_subtasks", 3)), len(facets)))
        return {"thought": f"Decompose '{goal}' into {n} facets: definition, evaluation, limitations.",
                "subtasks": [{"query": q, "rationale": "facet of the goal"} for q in facets[:n]]}

    def _react(self, p: dict[str, Any]) -> dict[str, Any]:
        allowed = p["allowed_actions"]
        fb = p.get("feedback") or []
        if fb and ("refine_subtask" in allowed or "abandon_subtask" in allowed):
            f = fb[0]
            st = next(s for s in p["subtasks"] if s["id"] == f["subtask_id"])
            if st["attempts"] == 0 and "refine_subtask" in allowed:
                broader = " ".join(st["query"].split()[:-1]) or st["query"]
                return {"thought": f"Feedback says '{f['message'][:80]}'. Broaden the query.",
                        "action": "refine_subtask", "subtask_id": st["id"], "query": broader}
            if st["attempts"] == 1 and "alternate_source" in allowed:
                return {"thought": "Refining did not help; try a different database.",
                        "action": "alternate_source", "subtask_id": st["id"], "source": "openalex"}
            return {"thought": "Repeated failures; this facet is not worth more budget.",
                    "action": "abandon_subtask", "subtask_id": st["id"]}
        pending = [s for s in p["subtasks"] if s["status"] == "pending"]
        if pending and "issue_subtask" in allowed:
            return {"thought": f"Next uncovered facet is '{pending[0]['query']}'.",
                    "action": "issue_subtask", "subtask_id": pending[0]["id"]}
        return {"thought": "Coverage is sufficient; synthesise the verified evidence.",
                "action": "synthesise"}

    def _synthesise(self, p: dict[str, Any]) -> dict[str, Any]:
        claims = p["claims"]
        if not claims:
            return {"text": "No verified evidence was found for this goal within the step budget."}
        parts = [f"{c['text'].rstrip('.')} [C{c['claim_id']}]." for c in claims]
        return {"text": f"Evidence on '{p['goal']}': " + " ".join(parts)}

    # ----------------------------------------------------------- Processor
    def _summarise(self, p: dict[str, Any]) -> dict[str, Any]:
        out = []
        for s in p["sources"]:
            sents = _sentences(s.get("abstract") or "")
            if not sents:
                continue
            out.append({"source_id": s["source_id"], "summary": " ".join(sents[:2]),
                        "claims": sents[: int(p.get("max_claims", 2))]})
        return {"summaries": out}

    # -------------------------------------------------------------- Critic
    def _verify_claims(self, p: dict[str, Any]) -> dict[str, Any]:
        verdicts = []
        for it in p["items"]:
            ct, st = _tokens(it["claim"]), _tokens(it["source_text"])
            overlap = len(ct & st) / max(len(ct), 1)
            ok = overlap >= 0.8
            verdicts.append({"claim_id": it["claim_id"], "label": "supported" if ok else "unsupported",
                             "reason": f"lexical overlap {overlap:.2f} ({'>=' if ok else '<'} 0.80)"})
        return {"verdicts": verdicts}

    def _verify_synthesis(self, p: dict[str, Any]) -> dict[str, Any]:
        known = {c["claim_id"] for c in p["claims"]}
        cited = {int(x) for x in re.findall(r"\[C(\d+)\]", p["synthesis"])}
        bad = cited - known
        if bad or (known and not cited):
            return {"label": "unsupported", "reason": f"citations not backed by verified claims: {sorted(bad)}"}
        return {"label": "supported", "reason": "every cited claim is a verified claim"}


class ScriptedLLM(MockLLM):
    """Mock whose answers can be overridden per task, for testing failure paths.

    ``script[task]`` is a list consumed in order; each item is a raw string, a
    dict (serialised to JSON), or a callable receiving the payload. When a
    task's list is exhausted the default mock behaviour resumes.
    """

    def __init__(self, script: dict[str, list[Any]] | None = None):
        super().__init__()
        self.script = defaultdict(list, {k: list(v) for k, v in (script or {}).items()})

    def complete(self, *, task: str, model: str, system: str, user: str,
                 temperature: float, json_mode: bool = True) -> str:
        if self.script[task]:
            item = self.script[task].pop(0)
            self.calls.append({"task": task, "model": model, "scripted": True})
            if callable(item):
                item = item(_payload(user))
            return item if isinstance(item, str) else json.dumps(item)
        return super().complete(task=task, model=model, system=system, user=user,
                                temperature=temperature, json_mode=json_mode)
