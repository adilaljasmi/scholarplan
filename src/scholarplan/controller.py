"""Blackboard control loop.

In a blackboard system the *control component* decides which applicable
knowledge source runs next (Hayes-Roth, 1985). Here that component is
deliberately small and contains no domain logic: each cycle it asks every
agent ``can_act`` and runs the first applicable one in priority order. All
domain decisions stay inside the agents, so agents can be added or replaced
without touching the controller (open/closed principle).

Priority is "downstream first" (Critic > Processor > Retriever > Archivist >
Planner): work already in flight is finished before the Planner deliberates
again, so the Planner always observes complete results and never spends a
ReAct step on stale state.

Termination is guaranteed independently of any LLM: the Planner's step budget
bounds deliberation, the re-plan cap bounds each subtask, and ``max_cycles``
is a final safety net against a scheduling bug.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from scholarplan.agents import (Agent, ArchivistAgent, CriticAgent, PlannerAgent, ProcessorAgent,
                                RetrieverAgent)
from scholarplan.blackboard import Blackboard
from scholarplan.config import Settings
from scholarplan.embeddings import Embedder, get_embedder
from scholarplan.llm import LLMClient, build_llm
from scholarplan.sources import build_clients


@dataclass
class RunResult:
    run_id: str
    status: str
    output_dir: Path | None
    cycles: int
    summary: dict[str, Any] = field(default_factory=dict)


class Controller:
    def __init__(self, settings: Settings, llm: LLMClient | None = None,
                 clients: dict[str, object] | None = None, embedder: Embedder | None = None,
                 db_path: str | Path | None = None, max_cycles: int = 300,
                 on_cycle: Callable[[str, str], None] | None = None):
        self.settings = settings
        self.llm = llm or build_llm(settings)
        self.embedder = embedder or get_embedder(settings.embedding_backend, settings.embedding_model)
        clients = clients if clients is not None else build_clients(settings)
        self.db_path = db_path or Path(settings.output_dir) / "blackboard.db"
        self.bb = Blackboard(self.db_path)
        self.agents: list[Agent] = [
            CriticAgent(settings, self.llm),
            ProcessorAgent(settings, self.llm, self.embedder),
            RetrieverAgent(settings, clients),
            ArchivistAgent(settings, self.embedder, self.llm),
            PlannerAgent(settings, self.llm),
        ]
        self.max_cycles = max_cycles
        self.on_cycle = on_cycle
        self.stop_event = threading.Event()

    def start(self, goal: str) -> str:
        run_id = self.bb.create_run(goal.strip(), self.settings.step_budget, self.settings.llm_backend)
        self.bb.log("system", "event", f"run started: \"{goal.strip()}\"",
                    {"settings": {k: str(v) for k, v in vars(self.settings).items() if k != "groq_api_key"}})
        return run_id

    def step(self) -> tuple[str, str] | None:
        """Run one applicable agent. Returns (role, description) or None if idle."""
        for agent in self.agents:
            if agent.can_act(self.bb):
                return agent.role, agent.act(self.bb)
        return None

    def run(self, goal: str) -> RunResult:
        run_id = self.start(goal)
        return self.run_existing(run_id)

    def run_existing(self, run_id: str) -> RunResult:
        self.bb.run_id = run_id
        cycles = 0
        try:
            while cycles < self.max_cycles and not self.stop_event.is_set():
                run = self.bb.get_run()
                if run["status"] in ("done", "failed"):
                    break
                res = self.step()
                cycles += 1
                if res is None:
                    # No agent applicable but the run is unfinished: a
                    # coordination dead-end. Fail loudly rather than spin.
                    self.bb.log("system", "error", "no agent can act; run cannot progress")
                    self.bb.set_run_status("failed", "deadlock: no applicable agent")
                    break
                if self.on_cycle:
                    self.on_cycle(*res)
            else:
                reason = "stopped by user" if self.stop_event.is_set() else f"max_cycles={self.max_cycles} reached"
                self.bb.log("system", "error", reason)
                self.bb.set_run_status("failed", reason)
        except Exception as exc:  # surface the failure in the trace and UI, then re-raise
            self.bb.log("system", "error", f"{type(exc).__name__}: {exc}")
            self.bb.set_run_status("failed", f"{type(exc).__name__}: {exc}")
            raise
        run = self.bb.get_run()
        out = Path(self.settings.output_dir) / run_id
        summary_file = out / "run_summary.json"
        summary: dict[str, Any] = {}
        if summary_file.exists():
            import json
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        return RunResult(run_id, run["status"], out if out.exists() else None, cycles, summary)
