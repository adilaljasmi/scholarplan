"""Processor: de-duplicate, rank, filter, then summarise into atomic claims.

Trigger: a subtask in state ``retrieved``.

The work is deliberately split into a deterministic stage and an LLM stage:

1. **Deterministic filtering (no LLM).** Irrelevant context measurably
   degrades LLM reasoning (Shi et al., 2023), so noise is removed *before* the
   model sees it: duplicates are dropped (normalised DOI, normalised title,
   or embedding cosine >= ``dedup_similarity``), sources below
   ``relevance_floor`` cosine to the subtask query are discarded, and only
   the ``top_k_sources`` best are kept. These thresholds are explicit numbers
   in ``config.py`` because the Unit 6 feedback asked for measurable
   de-duplication criteria.
2. **LLM summarisation.** The model turns each kept abstract into a short
   summary plus *atomic* claims. Atomic claims are what make independent
   verification possible: the Critic checks one small proposition at a time,
   which is how fact-checkers such as MiniCheck operate (Tang, Laban and
   Durrett, 2024).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scholarplan.agents import prompts
from scholarplan.agents.base import Agent
from scholarplan.blackboard import Blackboard
from scholarplan.config import Settings
from scholarplan.embeddings import Embedder, cosine_matrix
from scholarplan.llm import LLMClient, LLMError, call_json
from scholarplan.llm.schemas import ProcessorOutput
from scholarplan.sources.records import normalise_doi, normalise_title

MIN_ABSTRACT_CHARS = 100  # shorter "abstracts" are usually stubs that cannot ground a claim


def doc_text(src: dict[str, Any]) -> str:
    return f"{src['title']}. {(src.get('abstract') or '')[:1000]}"


def select_sources(query: str, candidates: list[dict[str, Any]], already_used: list[dict[str, Any]],
                   embedder: Embedder, settings: Settings) -> dict[int, dict[str, Any]]:
    """Pure function deciding the fate of every candidate source.

    Kept separate from the agent so the thresholds can be unit-tested without
    a blackboard or an LLM. Returns ``{source_id: {relevance, duplicate_of,
    selected, reason}}``.
    """
    decisions: dict[int, dict[str, Any]] = {}
    if not candidates:
        return decisions
    # Prefer records that can be verified and cited: abstract first, then DOI.
    ordered = sorted(candidates, key=lambda s: (not s.get("abstract"), not s.get("doi"), s["id"]))
    vecs = embedder.embed([doc_text(s) for s in ordered])
    qvec = embedder.embed([query])
    relevance = cosine_matrix(vecs, qvec)[:, 0]
    used_vecs = embedder.embed([doc_text(s) for s in already_used]) if already_used else None

    kept_idx: list[int] = []
    seen_doi = {normalise_doi(s.get("doi")): s["id"] for s in already_used if s.get("doi")}
    seen_title = {normalise_title(s["title"]): s["id"] for s in already_used}
    for i, s in enumerate(ordered):
        d = {"relevance": float(relevance[i]), "duplicate_of": None, "selected": False, "reason": ""}
        decisions[s["id"]] = d
        doi, title = normalise_doi(s.get("doi")), normalise_title(s["title"])
        dup = seen_doi.get(doi) if doi else None
        dup = dup or seen_title.get(title)
        if dup is None and kept_idx:
            sims = cosine_matrix(vecs[i:i + 1], vecs[kept_idx])[0]
            j = int(np.argmax(sims))
            if sims[j] >= settings.dedup_similarity:
                dup = ordered[kept_idx[j]]["id"]
        if dup is None and used_vecs is not None:
            sims = cosine_matrix(vecs[i:i + 1], used_vecs)[0]
            j = int(np.argmax(sims))
            if sims[j] >= settings.dedup_similarity:
                dup = already_used[j]["id"]
        if dup is not None:
            d["duplicate_of"], d["reason"] = dup, "duplicate"
            continue
        if doi:
            seen_doi[doi] = s["id"]
        seen_title[title] = s["id"]
        kept_idx.append(i)
        if len(s.get("abstract") or "") < MIN_ABSTRACT_CHARS:
            d["reason"] = "no usable abstract"
        elif d["relevance"] < settings.relevance_floor:
            d["reason"] = f"relevance {d['relevance']:.2f} < floor {settings.relevance_floor:.2f}"
        else:
            d["reason"] = "eligible"

    eligible = sorted((sid for sid, d in decisions.items() if d["reason"] == "eligible"),
                      key=lambda sid: -decisions[sid]["relevance"])
    for rank, sid in enumerate(eligible):
        if rank < settings.top_k_sources:
            decisions[sid]["selected"], decisions[sid]["reason"] = True, f"selected (rank {rank + 1})"
        else:
            decisions[sid]["reason"] = f"below top-{settings.top_k_sources}"
    return decisions


class ProcessorAgent(Agent):
    role = "processor"
    uses_llm = True

    def __init__(self, settings: Settings, llm: LLMClient, embedder: Embedder):
        super().__init__(settings)
        self.llm, self.embedder = llm, embedder

    def can_act(self, bb: Blackboard) -> bool:
        return bool(bb.subtasks("retrieved"))

    def _reject(self, sb, st: dict[str, Any], kind: str, msg: str) -> str:
        sb.add_feedback(self.role, kind, msg, subtask_id=st["id"])
        sb.set_subtask_status(st["id"], "rejected")
        sb.log("feedback", msg, {"subtask_id": st["id"]})
        return f"subtask {st['id']}: {kind}"

    def act(self, bb: Blackboard) -> str:
        sb = self.scoped(bb)
        st = sb.claim_subtask("retrieved", "processing", owner=self.role)
        if st is None:
            return "no retrieved subtask"
        attempt = st["attempts"]
        candidates = sb.sources(subtask_id=st["id"], attempt=attempt)
        already_used = [s for s in sb.sources(selected_only=True) if s["subtask_id"] != st["id"]]

        decisions = select_sources(st["query"], candidates, already_used, self.embedder, self.settings)
        for sid, d in decisions.items():
            sb.annotate_source(sid, relevance=round(d["relevance"], 4),
                               duplicate_of=d["duplicate_of"], selected=d["selected"])
        n_dup = sum(1 for d in decisions.values() if d["duplicate_of"])
        selected = [s for s in candidates if decisions.get(s["id"], {}).get("selected")]
        sb.log("observation",
               f"subtask {st['id']}: {len(candidates)} candidates, {n_dup} duplicates, "
               f"{len(selected)} selected",
               {"subtask_id": st["id"],
                "decisions": {sid: {"rel": round(d["relevance"], 3), "why": d["reason"]}
                              for sid, d in decisions.items()}})
        if not selected:
            best = max((d["relevance"] for d in decisions.values()), default=0.0)
            return self._reject(sb, st, "no_relevant",
                                f"None of {len(candidates)} results for \"{st['query']}\" passed the "
                                f"relevance floor {self.settings.relevance_floor:.2f} (best {best:.2f}) "
                                "or had a usable abstract. Use more specific domain terms.")

        payload = {"query": st["query"], "max_claims": self.settings.max_claims_per_source,
                   "sources": [{"source_id": s["id"], "title": s["title"], "abstract": s["abstract"]}
                               for s in selected]}
        try:
            out = call_json(self.llm, ProcessorOutput, task="summarise",
                            model=self.settings.models.processor,
                            system=prompts.SUMMARISE.format(max_claims=self.settings.max_claims_per_source),
                            payload=payload, temperature=self.settings.worker_temperature,
                            max_retries=self.settings.llm_max_retries)
        except LLMError as exc:
            # State the cause explicitly. In a live run the Planner read a bare
            # "Summarisation failed" as a bad query and rewrote a good one (R15).
            return self._reject(sb, st, "processing_failed",
                                f"The Processor model failed to return valid output ({exc}). This is a model "
                                f"output-format failure, NOT evidence that the query \"{st['query']}\" is "
                                f"poor: {len(selected)} relevant sources were found. Prefer retrying via "
                                "alternate_source over rewriting the query.")

        valid_ids = {s["id"] for s in selected}
        n_claims = 0
        for item in out.summaries:
            if item.source_id not in valid_ids:
                # A summary attributed to a source we did not give the model is
                # a fabricated citation: drop it rather than trust it.
                sb.log("error", f"processor returned unknown source_id {item.source_id}; ignored")
                continue
            sb.add_summary(st["id"], item.source_id, item.summary)
            for claim in item.claims[: self.settings.max_claims_per_source]:
                sb.add_claim(st["id"], item.source_id, attempt, claim)
                n_claims += 1
        if self.settings.demo_inject_fabrication and st["seq"] == 1 and attempt == 0 and selected:
            # Demonstration only (see config.demo_inject_fabrication): a
            # fabricated, over-claiming statement attributed to a real source.
            fake = (f"This study reports that its method eliminates hallucination entirely and improves "
                    f"accuracy by 99% on every benchmark.")
            sb.add_claim(st["id"], selected[0]["id"], attempt, fake)
            n_claims += 1
            sb.log("event", f"[DEMO] injected a fabricated claim for source {selected[0]['id']}: \"{fake}\"",
                   {"subtask_id": st["id"], "demo": True})
        if n_claims == 0:
            return self._reject(sb, st, "no_claims",
                                f"No claims relevant to \"{st['query']}\" could be extracted from the abstracts.")
        sb.set_subtask_status(st["id"], "processed")
        sb.log("observation", f"subtask {st['id']}: {n_claims} claims from {len(selected)} sources",
               {"subtask_id": st["id"]})
        return f"subtask {st['id']}: {n_claims} claims"
