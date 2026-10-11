"""Evaluation harness (Unit 6, section 4, and Unit 6 feedback #2).

Three measurements, each with a target fixed *before* running:

* retrieval: recall of known-relevant papers for fixed queries
  (target >= 0.75), plus how many cross-API duplicates the Processor's
  de-duplication removes;
* critic: precision / recall of the Critic on hand-labelled claim-source
  pairs (targets: precision >= 0.85, recall >= 0.75). The production Critic
  model is compared with the fallback model and with a lexical-overlap
  baseline, which is how the Critic model choice is justified empirically
  (benchmark-driven model selection, as in Unit 6);
* e2e: full live runs on fixed goals: completion within the step budget,
  steps, re-plans, Critic pass rate, tokens and wall-clock time.

Unit tests answer "does the code do what it says?"; this harness answers
"is the agent any good?", which for non-deterministic LLM components can only
be estimated empirically.

Usage:  python eval/run_eval.py retrieval|critic|e2e|all
Results are written to evidence/evaluation/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholarplan.agents import prompts  # noqa: E402
from scholarplan.agents.processor import select_sources  # noqa: E402
from scholarplan.config import load_settings  # noqa: E402
from scholarplan.embeddings import get_embedder  # noqa: E402
from scholarplan.llm import MockLLM, build_llm, call_json  # noqa: E402
from scholarplan.llm.schemas import CriticOutput  # noqa: E402
from scholarplan.sources import build_clients  # noqa: E402
from scholarplan.sources.clients import ArxivClient, CrossrefClient, OpenAlexClient  # noqa: E402
from scholarplan.sources.records import normalise_doi  # noqa: E402

OUT = ROOT / "evidence" / "evaluation"
FIX = ROOT / "tests" / "fixtures"

E2E_GOALS = [
    "How does verifying LLM outputs against retrieved sources reduce hallucination?",
    "What coordination mechanisms do multi-agent LLM systems use, and how do blackboard architectures compare?",
    "How are tool-using LLM agents evaluated, and what are the main benchmarks?",
]


def save(name: str, data: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ retrieval
def ids_of(rec: dict) -> set[str]:
    ids = {normalise_doi(rec.get("doi")) or "", (rec.get("external_id") or "").lower()}
    return {i for i in ids if i}


def eval_retrieval() -> dict:
    settings = load_settings()
    bench = json.loads((ROOT / "eval" / "retrieval_benchmark.json").read_text(encoding="utf-8"))
    clients = build_clients(settings)
    embedder = get_embedder(settings.embedding_backend, settings.embedding_model)
    rows, hits = [], 0
    for q in bench["queries"]:
        expect = {e.lower() for e in q["expect"]}
        found_by, records = [], []
        for name, client in clients.items():
            try:
                recs = client.search(q["query"], settings.results_per_source)
            except Exception as exc:  # report, do not abort the benchmark
                recs = []
                found_by.append(f"{name}:ERROR {exc}")
            records += recs
            if any(ids_of(r) & expect for r in recs):
                found_by.append(name)
        hit = any(not f.endswith("ERROR") and ":" not in f for f in found_by)
        hits += hit
        cands = [r | {"id": i} for i, r in enumerate(records)]
        dec = select_sources(q["query"], cands, [], embedder, settings)
        dups = sum(1 for d in dec.values() if d["duplicate_of"] is not None)
        rows.append({"id": q["id"], "query": q["query"], "cites": q["cites"], "hit": hit,
                     "found_by": found_by, "records": len(records), "duplicates_removed": dups})
        print(f"{q['id']:<4} {'HIT ' if hit else 'MISS'} {','.join(found_by) or '-':<28} "
              f"records={len(records):>2} dups={dups}  {q['query']}")
    recall = hits / len(bench["queries"])
    result = {"recall": round(recall, 3), "target": bench["targets"]["recall"],
              "met": recall >= bench["targets"]["recall"], "queries": rows,
              "dedup_threshold": settings.dedup_similarity, "relevance_floor": settings.relevance_floor}
    print(f"\nRetrieval recall {recall:.2f} (target {bench['targets']['recall']}) -> "
          f"{'MET' if result['met'] else 'NOT MET'}")
    return result


# --------------------------------------------------------------------- critic
def load_source(ref: str) -> dict:
    fname, idx = ref.split("#")
    text = (FIX / fname).read_text(encoding="utf-8")
    if fname.startswith("arxiv"):
        recs = ArxivClient.parse(text)
    elif fname.startswith("openalex"):
        recs = OpenAlexClient.parse(json.loads(text))
    else:
        recs = CrossrefClient.parse(json.loads(text))
    return recs[int(idx)]


def metrics(pairs: list[tuple[str, str]]) -> dict:
    tp = sum(g == p == "supported" for g, p in pairs)
    fp = sum(g == "unsupported" and p == "supported" for g, p in pairs)
    fn = sum(g == "supported" and p == "unsupported" for g, p in pairs)
    tn = sum(g == p == "unsupported" for g, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "hallucinations_caught": f"{tn}/{tn + fp}", "accuracy": round((tp + tn) / len(pairs), 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def eval_critic() -> dict:
    settings = load_settings()
    bench = json.loads((ROOT / "eval" / "critic_benchmark.json").read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in bench["items"]:
        groups[it["source"]].append(it)
    candidates = {"lexical-baseline (mock)": (MockLLM(), "mock")}
    if settings.llm_backend == "groq":
        live = build_llm(settings)
        candidates[f"{settings.models.critic} (production)"] = (live, settings.models.critic)
        candidates[f"{settings.models.critic_fallback} (fallback)"] = (live, settings.models.critic_fallback)
    results = {}
    for label, (llm, model) in candidates.items():
        pairs, errors, detail = [], [], []
        t0 = time.time()
        for ref, items in groups.items():
            src = load_source(ref)
            payload = {"items": [{"claim_id": it["id"], "claim": it["claim"],
                                  "source_text": f"{src['title']}. {src['abstract']}"} for it in items]}
            try:
                out = call_json(llm, CriticOutput, task="verify_claims", model=model,
                                system=prompts.VERIFY_CLAIMS, payload=payload, temperature=0)
                pred = {v.claim_id: v for v in out.verdicts}
            except Exception as exc:
                errors.append(str(exc))
                pred = {}
            for it in items:
                v = pred.get(it["id"])
                p = v.label if v else "unsupported"  # same conservative default as production
                pairs.append((it["label"], p))
                detail.append({"id": it["id"], "type": it["type"], "gold": it["label"], "pred": p,
                               "reason": v.reason if v else "no verdict"})
        m = metrics(pairs) | {"seconds": round(time.time() - t0, 1), "errors": errors}
        m["met"] = m["precision"] >= bench["targets"]["precision"] and m["recall"] >= bench["targets"]["recall"]
        m["errors_by_type"] = sorted({d["type"] for d in detail if d["gold"] != d["pred"]})
        m["items"] = detail
        results[label] = m
        print(f"{label:<40} P={m['precision']:.2f} R={m['recall']:.2f} acc={m['accuracy']:.2f} "
              f"caught={m['hallucinations_caught']} {'MET' if m['met'] else 'NOT MET'} "
              f"misjudged={m['errors_by_type']}")
    return {"targets": bench["targets"], "n_items": len(bench["items"]), "models": results}


# ------------------------------------------------------------------------ e2e
def eval_e2e(goals: list[str]) -> dict:
    from scholarplan.controller import Controller
    settings = load_settings(output_dir=ROOT / "output" / "eval")
    runs = []
    for g in goals:
        t0 = time.time()
        ctl = Controller(settings)
        res = ctl.run(g)
        s = res.summary
        row = {"goal": g, "status": res.status, "seconds": round(time.time() - t0, 1),
               "run_id": res.run_id, **{k: s.get(k) for k in (
                   "steps_used", "step_budget", "replans", "subtasks", "sources_retrieved",
                   "duplicates_removed", "claims_total", "claims_supported", "synthesis_status",
                   "synthesis_attempts", "llm_usage")}}
        runs.append(row)
        print(f"{res.status:<6} steps={row['steps_used']}/{row['step_budget']} replans={row['replans']} "
              f"claims={row['claims_supported']}/{row['claims_total']} {row['seconds']}s  {g}")
    done = [r for r in runs if r["status"] == "done"]
    tot_claims = sum(r["claims_total"] or 0 for r in done)
    return {"backend": settings.llm_backend, "runs": runs,
            "completed_within_budget": f"{len(done)}/{len(runs)}",
            "mean_steps": round(sum(r["steps_used"] for r in done) / max(len(done), 1), 1),
            "critic_pass_rate": round(sum(r["claims_supported"] for r in done) / tot_claims, 3) if tot_claims else None,
            "mean_seconds": round(sum(r["seconds"] for r in done) / max(len(done), 1), 1)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["retrieval", "critic", "e2e", "all"])
    ap.add_argument("--goals", type=int, default=len(E2E_GOALS))
    a = ap.parse_args()
    stamp = datetime.now().isoformat(timespec="seconds")
    if a.which in ("retrieval", "all"):
        print(f"Saved {save('retrieval', {'run_at': stamp} | eval_retrieval())}\n")
    if a.which in ("critic", "all"):
        print(f"Saved {save('critic', {'run_at': stamp} | eval_critic())}\n")
    if a.which in ("e2e", "all"):
        print(f"Saved {save('e2e', {'run_at': stamp} | eval_e2e(E2E_GOALS[:a.goals]))}\n")


if __name__ == "__main__":
    main()
