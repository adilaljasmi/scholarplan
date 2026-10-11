"""Planner: the single deliberative agent, running a bounded ReAct loop.

Architecture (Unit 6, section 3): ScholarPlan is a *hybrid* system. One
deliberative LLM agent reasons about the goal while reactive specialists do
the work (Wooldridge, 2009). The Planner is that deliberative layer.

How the ReAct loop maps onto the blackboard (Yao et al., 2023):
  * Observation: the Planner reads the blackboard (subtask states, feedback,
    verified-claim counts, remaining budget).
  * Thought: the LLM writes its reasoning, persisted to the trace.
  * Action: exactly one of issue / refine / alternate source / abandon /
    synthesise, executed as a blackboard write. The specialists then react,
    and their results become the next Observation.

Reflexion-style re-planning (Shinn et al., 2023): when a subtask is rejected,
the Retriever, Processor or Critic writes a natural-language *Feedback* entry.
The Planner reads it before its next Thought and chooses to narrow or broaden
the query, switch database, or abandon.

LLM-Modulo safeguards (Kambhampati et al., 2024): LLMs plan unreliably, so
the LLM never decides what is *allowed*. Code computes the allowed actions,
validates the LLM's choice, enforces the 20-step budget and the 3-re-plan cap
(Unit 6 feedback #3), and falls back to a deterministic policy if the LLM
fails twice. The LLM supplies judgement; the code guarantees termination.
"""

from __future__ import annotations

from typing import Any

from scholarplan.agents import prompts
from scholarplan.agents.base import Agent
from scholarplan.blackboard import TERMINAL_STATES, Blackboard
from scholarplan.config import Settings
from scholarplan.llm import LLMClient, LLMError, call_json
from scholarplan.llm.schemas import PlanOutput, ReActOutput, SynthesisOutput
from scholarplan.sources.records import keywords

IN_FLIGHT = frozenset({"open", "retrieving", "retrieved", "processing", "processed", "verifying"})
SOURCES = ("arxiv", "openalex", "crossref")


def coverage(subtasks: list[dict[str, Any]]) -> float:
    """Share of subtasks verified: the Unit 6 'coverage threshold' metric."""
    return sum(s["status"] == "verified" for s in subtasks) / len(subtasks) if subtasks else 0.0


def allowed_actions(subtasks: list[dict[str, Any]], rejected_with_feedback: set[int],
                    threshold: float) -> list[str]:
    """Which actions are legal right now (pure function, unit-tested).

    Rejected subtasks are dealt with first, because feedback is the signal
    that the plan is wrong and ignoring it would waste the step budget.
    """
    if rejected_with_feedback:
        return ["refine_subtask", "alternate_source", "abandon_subtask"]
    acts = []
    if any(s["status"] == "pending" for s in subtasks):
        acts.append("issue_subtask")
    all_terminal = all(s["status"] in TERMINAL_STATES for s in subtasks)
    if all_terminal or coverage(subtasks) >= threshold:
        acts.append("synthesise")
    return acts


def default_action(subtasks: list[dict[str, Any]], feedback: list[dict[str, Any]],
                   allowed: list[str]) -> ReActOutput:
    """Deterministic fallback policy used when the LLM cannot produce a valid step."""
    if "abandon_subtask" in allowed:
        fb = feedback[0]
        st = next(s for s in subtasks if s["id"] == fb["subtask_id"])
        if st["attempts"] < 2:
            src = SOURCES[(st["attempts"] + 1) % len(SOURCES)]
            return ReActOutput(thought="[fallback policy] retry on a different database",
                               action="alternate_source", subtask_id=st["id"], source=src)
        return ReActOutput(thought="[fallback policy] abandon after repeated failure",
                           action="abandon_subtask", subtask_id=st["id"])
    if "issue_subtask" in allowed:
        nxt = next(s for s in subtasks if s["status"] == "pending")
        return ReActOutput(thought="[fallback policy] issue next pending subtask",
                           action="issue_subtask", subtask_id=nxt["id"])
    return ReActOutput(thought="[fallback policy] synthesise", action="synthesise")


class PlannerAgent(Agent):
    role = "planner"
    uses_llm = True

    def __init__(self, settings: Settings, llm: LLMClient):
        super().__init__(settings)
        self.llm = llm
        self.query_history: dict[int, list[str]] = {}

    # ---------------------------------------------------------------- trigger
    def can_act(self, bb: Blackboard) -> bool:
        run = bb.get_run()
        if run is None or run["status"] in ("done", "failed"):
            return False
        if run["status"] == "planning":
            return True
        syns = bb.syntheses()
        if any(s["status"] in ("pending", "verified", "fallback") for s in syns):
            return False  # waiting for the Critic or the Archivist
        if bb.feedback(unconsumed_only=True):
            return True
        subtasks = bb.subtasks()
        return not any(s["status"] in IN_FLIGHT for s in subtasks)

    # ------------------------------------------------------------------- act
    def act(self, bb: Blackboard) -> str:
        sb = self.scoped(bb)
        run = sb.get_run()
        if run["status"] == "planning":
            return self._plan(sb, run["goal"])

        feedback = sb.feedback(unconsumed_only=True)
        syn_fb = [f for f in feedback if f["kind"] == "synthesis_rejected"]
        if syn_fb:
            sb.consume_feedback([f["id"] for f in syn_fb])
            return self._synthesise(sb, run["goal"], previous_feedback=syn_fb[-1]["message"])

        subtasks = sb.subtasks()
        # Hard rule, enforced in code: the re-plan cap (Unit 6 feedback #3).
        for f in feedback:
            st = next((s for s in subtasks if s["id"] == f["subtask_id"]), None)
            if st and st["status"] == "rejected" and st["attempts"] >= self.settings.max_replans_per_subtask:
                sb.set_subtask_status(st["id"], "unresolvable")
                sb.consume_feedback([x["id"] for x in feedback if x["subtask_id"] == st["id"]])
                sb.log("thought", f"Subtask {st['id']} failed {st['attempts'] + 1} times; the re-plan cap "
                       f"({self.settings.max_replans_per_subtask}) is reached, so it is marked unresolvable.",
                       {"subtask_id": st["id"], "rule": "max_replans_per_subtask"})
                return f"subtask {st['id']} unresolvable"

        if sb.budget_left() <= 0:
            return self._exhausted(sb, run["goal"], subtasks, feedback)

        return self._react_step(sb, run["goal"], subtasks, feedback)

    # ------------------------------------------------------------- planning
    def _plan(self, sb, goal: str) -> str:
        step = sb.consume_step()
        try:
            plan = call_json(self.llm, PlanOutput, task="plan", model=self.settings.models.planner,
                             system=prompts.PLAN.format(max_subtasks=self.settings.max_subtasks),
                             payload={"goal": goal, "max_subtasks": self.settings.max_subtasks},
                             temperature=self.settings.planner_temperature,
                             max_retries=self.settings.llm_max_retries)
            thought, items = plan.thought, plan.subtasks
        except LLMError as exc:
            # Fallback: a single subtask built from the goal's keywords keeps
            # the run alive rather than failing on the very first call.
            thought = f"[fallback] planning LLM failed ({exc}); searching the goal directly"
            items = [type("S", (), {"query": " ".join(keywords(goal)) or goal, "rationale": "fallback"})()]
        seen: set[str] = set()
        created = []
        for it in items:
            q = it.query.strip()
            if q.lower() in seen:
                continue  # duplicate subtasks waste budget
            seen.add(q.lower())
            created.append(sb.add_subtask(len(created) + 1, q, it.rationale))
            self.query_history[created[-1]] = [q]
            if len(created) >= self.settings.max_subtasks:
                break  # bound decomposition even if the LLM over-decomposes
        sb.log("thought", thought, {"step": step})
        sb.log("action", f"create_plan: {len(created)} subtasks",
               {"subtasks": [s["query"] for s in sb.subtasks()]})
        sb.set_run_status("running")
        return f"plan with {len(created)} subtasks"

    # ------------------------------------------------------------ ReAct step
    def _react_step(self, sb, goal: str, subtasks: list[dict[str, Any]],
                    feedback: list[dict[str, Any]]) -> str:
        rejected_fb = {f["subtask_id"] for f in feedback if f["subtask_id"] is not None}
        allowed = allowed_actions(subtasks, rejected_fb, self.settings.coverage_threshold)
        if not allowed:
            return "nothing to decide"
        step = sb.consume_step()
        payload = {
            "goal": goal,
            "observation": {
                "step": step, "budget_left": sb.budget_left(),
                "coverage": round(coverage(subtasks), 2),
                "coverage_threshold": self.settings.coverage_threshold,
                "verified_claims": len(sb.claims(status="supported")),
            },
            "subtasks": [{"id": s["id"], "query": s["query"], "status": s["status"],
                          "attempts": s["attempts"], "strategy": s["strategy"],
                          "previous_queries": self.query_history.get(s["id"], [])} for s in subtasks],
            "feedback": [{"subtask_id": f["subtask_id"], "from": f["author"], "kind": f["kind"],
                          "message": f["message"]} for f in feedback],
            "allowed_actions": allowed,
        }
        decision, error = None, ""
        for _ in range(2):
            if error:
                payload["previous_error"] = error
            try:
                cand = call_json(self.llm, ReActOutput, task="react", model=self.settings.models.planner,
                                 system=prompts.REACT, payload=payload,
                                 temperature=self.settings.planner_temperature,
                                 max_retries=self.settings.llm_max_retries)
            except LLMError as exc:
                error = str(exc)
                continue
            error = self._validate(cand, allowed, subtasks, rejected_fb)
            if not error:
                decision = cand
                break
        if decision is None:
            sb.log("error", f"planner LLM gave no valid action ({error}); using fallback policy")
            feedback_for_rejected = [f for f in feedback if f["subtask_id"] in rejected_fb]
            decision = default_action(subtasks, feedback_for_rejected, allowed)

        sb.log("thought", decision.thought, {"step": step})
        return self._execute(sb, goal, decision, feedback)

    def _validate(self, d: ReActOutput, allowed: list[str], subtasks: list[dict[str, Any]],
                  rejected_fb: set[int]) -> str:
        """Semantic validation the JSON schema cannot express."""
        if d.action not in allowed:
            return f"action '{d.action}' is not in allowed_actions {allowed}"
        by_id = {s["id"]: s for s in subtasks}
        if d.action == "issue_subtask":
            if d.subtask_id not in by_id or by_id[d.subtask_id]["status"] != "pending":
                return "issue_subtask needs the id of a pending subtask"
        if d.action in ("refine_subtask", "alternate_source", "abandon_subtask"):
            if d.subtask_id not in rejected_fb:
                return f"{d.action} needs the id of a rejected subtask with feedback {sorted(rejected_fb)}"
        if d.action == "refine_subtask":
            if not d.query or len(d.query.strip()) < 3:
                return "refine_subtask needs a new query"
            if d.query.strip().lower() in (q.lower() for q in self.query_history.get(d.subtask_id, [])):
                return "that query was already tried; propose a different one"
        if d.action == "alternate_source" and not d.source:
            return "alternate_source needs a source"
        return ""

    def _execute(self, sb, goal: str, d: ReActOutput, feedback: list[dict[str, Any]]) -> str:
        fb_ids = [f["id"] for f in feedback if f["subtask_id"] == d.subtask_id]
        if d.action == "issue_subtask":
            sb.set_subtask_status(d.subtask_id, "open")
            desc = f"issue_subtask {d.subtask_id}"
        elif d.action == "refine_subtask":
            q = d.query.strip()
            self.query_history.setdefault(d.subtask_id, []).append(q)
            sb.set_subtask_status(d.subtask_id, "open", query=q, bump_attempt=True, strategy="all")
            sb.consume_feedback(fb_ids)
            desc = f"refine_subtask {d.subtask_id} -> \"{q}\""
        elif d.action == "alternate_source":
            sb.set_subtask_status(d.subtask_id, "open", bump_attempt=True, strategy=d.source)
            sb.consume_feedback(fb_ids)
            desc = f"alternate_source {d.subtask_id} -> {d.source}"
        elif d.action == "abandon_subtask":
            sb.set_subtask_status(d.subtask_id, "abandoned")
            sb.consume_feedback(fb_ids)
            desc = f"abandon_subtask {d.subtask_id}"
        else:
            sb.log("action", "synthesise")
            return self._synthesise(sb, goal)
        sb.log("action", desc, d.model_dump())
        return desc

    # ------------------------------------------------------------ synthesis
    def _synthesise(self, sb, goal: str, previous_feedback: str | None = None) -> str:
        attempt = len(sb.syntheses()) + 1
        claims = sb.claims(status="supported")
        sources = {s["id"]: s for s in sb.sources()}
        if not claims:
            sb.add_synthesis(attempt, "No claim could be verified against its source within the step "
                             "budget, so no answer is given. See the trace for what was tried.", "fallback")
            sb.set_run_status("synthesising")
            return "no verified evidence: empty synthesis"
        if attempt > self.settings.max_synthesis_attempts:
            sb.add_synthesis(attempt, self._extractive(claims, sources), "fallback")
            sb.log("thought", f"Synthesis rejected {attempt - 1} times; falling back to an extractive "
                   "list of verified claims, which cannot contain unsupported statements.")
            return "fallback synthesis"
        payload = {"goal": goal,
                   "claims": [{"claim_id": c["id"], "text": c["text"],
                               "source": self._cite(sources[c["source_id"]])} for c in claims]}
        if previous_feedback:
            payload["critic_feedback_on_previous_attempt"] = previous_feedback
        try:
            out = call_json(self.llm, SynthesisOutput, task="synthesise", model=self.settings.models.planner,
                            system=prompts.SYNTHESISE, payload=payload,
                            temperature=self.settings.planner_temperature,
                            max_retries=self.settings.llm_max_retries)
            sb.add_synthesis(attempt, out.text, "pending")
        except LLMError as exc:
            sb.log("error", f"synthesis failed ({exc}); using extractive fallback")
            sb.add_synthesis(attempt, self._extractive(claims, sources), "fallback")
        sb.set_run_status("synthesising")
        return f"synthesis attempt {attempt}"

    def _exhausted(self, sb, goal: str, subtasks, feedback) -> str:
        for s in subtasks:
            if s["status"] not in TERMINAL_STATES and s["status"] not in IN_FLIGHT:
                sb.set_subtask_status(s["id"], "abandoned")
        if feedback:
            sb.consume_feedback([f["id"] for f in feedback])
        sb.log("thought", f"Step budget of {self.settings.step_budget} exhausted: abandoning unfinished "
               "subtasks and synthesising what has been verified.", {"rule": "step_budget"})
        return self._synthesise(sb, goal)

    @staticmethod
    def _cite(src: dict[str, Any]) -> str:
        first = (src["authors"] or ["Anon."])[0].split()[-1] if src["authors"] else "Anon."
        etal = " et al." if len(src["authors"]) > 1 else ""
        return f"{first}{etal} ({src.get('year') or 'n.d.'}) {src['title']}"

    def _extractive(self, claims, sources) -> str:
        return "Verified findings:\n" + "\n".join(
            f"- {c['text']} [C{c['id']}] ({self._cite(sources[c['source_id']])})" for c in claims)
