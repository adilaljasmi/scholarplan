"""Command-line interface.

    scholarplan run "goal" [--llm groq|mock] [--budget 20]
    scholarplan search output/<run_id> "query"     # query a saved vector index
    scholarplan ui                                  # launch the Streamlit dashboard

The CLI prints the trace as it happens, so a terminal run is itself evidence of
the ReAct loop (Thought -> Action -> Observation) and of agent interaction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COLOURS = {"planner": "\033[94m", "retriever": "\033[92m", "processor": "\033[95m",
           "critic": "\033[93m", "archivist": "\033[91m", "system": "\033[90m"}
RESET = "\033[0m"


def _print_trace(bb, after: int, colour: bool) -> int:
    for t in bb.trace(after):
        c = COLOURS.get(t["agent"], "") if colour else ""
        r = RESET if colour else ""
        print(f"{c}[step {t['step']:>2}] {t['agent']:<9} {t['kind']:<11}{r} {t['content']}", flush=True)
        after = t["id"]
    return after


def cmd_run(args: argparse.Namespace) -> int:
    from scholarplan.config import load_settings
    from scholarplan.controller import Controller

    overrides = {"step_budget": args.budget}
    if args.llm:
        overrides["llm_backend"] = args.llm
    if args.out:
        overrides["output_dir"] = Path(args.out)
    if args.embeddings:
        overrides["embedding_backend"] = args.embeddings
    if args.inject_fabrication:
        overrides["demo_inject_fabrication"] = True
    settings = load_settings(**overrides)
    if settings.llm_backend == "groq" and not settings.groq_api_key:
        print("GROQ_API_KEY is not set (see .env.example). Use --llm mock to run offline.", file=sys.stderr)
        return 2
    colour = sys.stdout.isatty()
    print(f"ScholarPlan | LLM backend: {settings.llm_backend} | step budget: {settings.step_budget}")
    ctl = Controller(settings)
    state = {"last": 0}

    def on_cycle(role: str, desc: str) -> None:
        state["last"] = _print_trace(ctl.bb, state["last"], colour)

    ctl.on_cycle = on_cycle
    result = ctl.run(args.goal)
    _print_trace(ctl.bb, state["last"], colour)
    print(f"\nRun {result.run_id}: {result.status} after {result.cycles} agent activations")
    if result.output_dir:
        print(f"Outputs: {result.output_dir}")
        s = result.summary
        if s:
            print(f"Steps {s['steps_used']}/{s['step_budget']} | sources {s['sources_retrieved']} "
                  f"(dupes {s['duplicates_removed']}) | claims {s['claims_supported']}/{s['claims_total']} "
                  f"supported | synthesis {s['synthesis_status']}")
    return 0 if result.status == "done" else 1


def cmd_search(args: argparse.Namespace) -> int:
    from scholarplan.agents.archivist import search_index
    from scholarplan.config import load_settings
    from scholarplan.embeddings import get_embedder

    s = load_settings()
    path = Path(args.run_dir) / "index.sqlite"
    if not path.exists():
        print(f"No index at {path}", file=sys.stderr)
        return 2
    for hit in search_index(path, args.query, get_embedder(s.embedding_backend, s.embedding_model), args.k):
        print(f"[{hit['distance']:.3f}] {hit['kind']:<6} {hit['citation']}: {hit['text'][:160]}")
    return 0


def cmd_ui(_: argparse.Namespace) -> int:
    app = Path(__file__).resolve().parents[2] / "app" / "dashboard.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(prog="scholarplan", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the multi-agent system on a research goal")
    r.add_argument("goal")
    r.add_argument("--llm", choices=["groq", "mock"], help="override SCHOLARPLAN_LLM")
    r.add_argument("--budget", type=int, default=20, help="ReAct step budget (default 20)")
    r.add_argument("--out", help="output directory (default ./output)")
    r.add_argument("--embeddings", choices=["minilm", "hashing"], help="embedding backend")
    r.add_argument("--inject-fabrication", action="store_true",
                   help="demo: plant one false claim to show the Critic rejecting it")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("search", help="semantic search over a finished run's vector index")
    s.add_argument("run_dir")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(fn=cmd_search)
    u = sub.add_parser("ui", help="launch the Streamlit dashboard")
    u.set_defaults(fn=cmd_ui)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
