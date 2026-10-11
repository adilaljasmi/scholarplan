"""Retriever: reactive, deterministic, no LLM.

Trigger: a subtask in state ``open``. Output: ``sources`` rows carrying
provenance (origin, identifier, URL, fetch time), then state ``retrieved``.

Why no LLM here: searching an API is a well-defined operation, and keeping it
deterministic means it can be unit-tested conventionally (Unit 6, section 4)
and that a retrieval failure is never confused with a reasoning failure, the
"separate failure surfaces" argument for role specialisation (Guo et al., 2024).
"""

from __future__ import annotations

from scholarplan.agents.base import Agent
from scholarplan.blackboard import Blackboard
from scholarplan.config import Settings


class RetrieverAgent(Agent):
    role = "retriever"

    def __init__(self, settings: Settings, clients: dict[str, object]):
        super().__init__(settings)
        self.clients = clients

    def can_act(self, bb: Blackboard) -> bool:
        return bool(bb.subtasks("open"))

    def act(self, bb: Blackboard) -> str:
        sb = self.scoped(bb)
        st = sb.claim_subtask("open", "retrieving", owner=self.role)
        if st is None:  # another worker won the race; nothing to do
            return "no open subtask"
        names = list(self.clients) if st["strategy"] == "all" else [st["strategy"]]
        sb.log("action", f"search {', '.join(names)} for subtask {st['id']}: \"{st['query']}\"",
               {"subtask_id": st["id"], "attempt": st["attempts"]})

        n, errors, per_api = 0, [], {}
        for name in names:
            try:
                records = self.clients[name].search(st["query"], self.settings.results_per_source)
            except Exception as exc:  # one API failing must not sink the others
                errors.append(f"{name}: {exc}")
                continue
            per_api[name] = len(records)
            for rec in records:
                sb.add_source(st["id"], st["attempts"], rec)
                n += 1

        if errors:
            sb.log("error", "; ".join(errors))
        if n == 0:
            # Writing feedback (not raising) turns an empty search into
            # something the Planner can reason about and re-plan from.
            msg = (f"No results for query \"{st['query']}\" on {', '.join(names)}."
                   + (f" Errors: {'; '.join(errors)}" if errors else " Try a broader query."))
            sb.add_feedback(self.role, "no_results", msg, subtask_id=st["id"])
            sb.set_subtask_status(st["id"], "rejected")
            sb.log("feedback", msg, {"subtask_id": st["id"]})
            return f"subtask {st['id']}: no results"

        sb.set_subtask_status(st["id"], "retrieved")
        sb.log("observation", f"retrieved {n} records for subtask {st['id']} {per_api}",
               {"subtask_id": st["id"], "per_api": per_api})
        return f"subtask {st['id']}: {n} sources"
