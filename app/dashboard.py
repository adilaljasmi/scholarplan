"""ScholarPlan live dashboard (Streamlit).

Run with:  scholarplan ui    (or: streamlit run app/dashboard.py)

Why a dashboard: the blackboard is the system's only coordination channel,
so *showing the blackboard* is the most direct way to make the multi-agent
behaviour observable: who acted, what the Planner thought, which claims the
Critic rejected and why. This serves explainability (a Unit 6 challenge) and
the live demonstration.

Concurrency design: the agents run in a background thread with their own
SQLite connection; the page polls the same database file through a second,
read-only connection every second (``st.fragment(run_every=...)``). SQLite's
WAL mode allows one writer and many readers without blocking, so the UI
never freezes the agents and the agents never wait for the UI.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholarplan.blackboard import Blackboard  # noqa: E402
from scholarplan.config import load_settings  # noqa: E402

OUTPUT = ROOT / "output" / "ui"
DB = OUTPUT / "blackboard.db"

AGENTS = {
    "planner": ("🧭", "Planner", "#2563eb", "Deliberative · ReAct"),
    "retriever": ("🔎", "Retriever", "#16a34a", "arXiv · OpenAlex · Crossref"),
    "processor": ("⚙️", "Processor", "#9333ea", "Dedup · rank · summarise"),
    "critic": ("🛡️", "Critic", "#ea580c", "Independent verifier"),
    "archivist": ("🗄️", "Archivist", "#dc2626", "Report · BibTeX · index"),
    "system": ("▪️", "System", "#6b7280", ""),
}
KIND_LABEL = {"thought": "Thought", "action": "Action", "observation": "Observation", "verdict": "Verdict",
              "feedback": "Feedback", "event": "Event", "error": "Error"}
STATUS_COLOUR = {"verified": "🟢", "rejected": "🔴", "abandoned": "⚫", "unresolvable": "🟤", "pending": "⚪",
                 "open": "🔵", "retrieving": "🔵", "retrieved": "🔵", "processing": "🟣", "processed": "🟣",
                 "verifying": "🟠"}

st.set_page_config(page_title="ScholarPlan", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.agent-card {border-radius:10px;padding:10px 12px;border:2px solid #e5e7eb;text-align:center;min-height:92px}
.agent-card.active {box-shadow:0 0 0 3px var(--c);border-color:var(--c)}
.agent-card .name {font-weight:700;font-size:1.05rem}
.agent-card .sub {font-size:.78rem;opacity:.75}
.tr {padding:6px 10px;margin:3px 0;border-left:4px solid var(--c);border-radius:4px;background:rgba(127,127,127,.06)}
.tr .meta {font-size:.75rem;opacity:.7}
.bb {border:2px dashed #9ca3af;border-radius:12px;padding:6px 12px;text-align:center;font-size:.85rem;margin:6px 0 2px}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------- run control
def start_run(goal: str, backend: str, budget: int, max_subtasks: int) -> None:
    from scholarplan.controller import Controller
    settings = load_settings(llm_backend=backend, step_budget=budget, max_subtasks=max_subtasks,
                             output_dir=OUTPUT)
    ctl = Controller(settings, db_path=DB)
    run_id = ctl.start(goal)
    st.session_state.run_id = run_id
    st.session_state.controller = ctl

    def work() -> None:
        # No Streamlit calls in this thread: it has no script context. Errors
        # are written to the blackboard trace by the controller and shown by
        # the polling panel.
        try:
            ctl.run_existing(run_id)
        except Exception:
            pass

    t = threading.Thread(target=work, daemon=True, name=f"scholarplan-{run_id}")
    st.session_state.thread = t
    t.start()


def is_running() -> bool:
    t = st.session_state.get("thread")
    return bool(t and t.is_alive())


# -------------------------------------------------------------------- sidebar
settings0 = load_settings()
with st.sidebar:
    st.title("🧭 ScholarPlan")
    st.caption("LLM-powered multi-agent planning for academic research")
    goal = st.text_area("Research goal", height=100,
                        value="How does verifying LLM outputs against retrieved sources reduce hallucination?")
    has_key = bool(settings0.groq_api_key)
    backend = st.radio("LLM backend", ["groq", "mock"], index=0 if has_key else 1, horizontal=True,
                       help="groq = live models (needs GROQ_API_KEY). mock = deterministic offline rules.")
    if backend == "groq" and not has_key:
        st.warning("No GROQ_API_KEY found in .env; use mock.")
    budget = st.slider("ReAct step budget", 5, 30, settings0.step_budget)
    max_sub = st.slider("Max subtasks", 1, 6, settings0.max_subtasks)
    c1, c2 = st.columns(2)
    if c1.button("▶ Run agents", type="primary", disabled=is_running() or not goal.strip()
                 or (backend == "groq" and not has_key), use_container_width=True):
        start_run(goal, backend, budget, max_sub)
    if c2.button("■ Stop", disabled=not is_running(), use_container_width=True):
        st.session_state.controller.stop_event.set()

    st.divider()
    past = Blackboard(DB).list_runs() if DB.exists() else []
    if past:
        labels = {r["id"]: f"{r['status']:<5} · {r['goal'][:42]}" for r in past}
        default = st.session_state.get("run_id", past[0]["id"])
        chosen = st.selectbox("Inspect run", list(labels), format_func=labels.get,
                              index=list(labels).index(default) if default in labels else 0)
        if not is_running():
            st.session_state.run_id = chosen
    with st.expander("Models & thresholds"):
        m = settings0.models
        st.markdown(f"- Planner: `{m.planner}`\n- Processor: `{m.processor}`\n- Critic: `{m.critic}`\n"
                    f"- Re-plan cap: **{settings0.max_replans_per_subtask}** per subtask\n"
                    f"- Coverage threshold: **{settings0.coverage_threshold:.0%}**\n"
                    f"- Dedup cosine ≥ **{settings0.dedup_similarity}**\n"
                    f"- Relevance floor: **{settings0.relevance_floor}**")


# --------------------------------------------------------------------- views
def agent_strip(last_agent: str | None) -> None:
    cols = st.columns(5)
    for col, key in zip(cols, ["planner", "retriever", "processor", "critic", "archivist"]):
        icon, name, colour, sub = AGENTS[key]
        active = "active" if key == last_agent else ""
        col.markdown(f'<div class="agent-card {active}" style="--c:{colour}"><div style="font-size:1.6rem">'
                     f'{icon}</div><div class="name" style="color:{colour}">{name}</div>'
                     f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="bb">⇅ all agents read & write only the shared <b>SQLite blackboard</b> '
                '(plan · subtasks · sources · claims · verdicts · feedback · trace) ⇅</div>',
                unsafe_allow_html=True)


def trace_view(bb: Blackboard, kinds: list[str]) -> None:
    rows = [t for t in bb.trace() if t["kind"] in kinds]
    box = st.container(height=520)
    for t in rows:
        icon, name, colour, _ = AGENTS.get(t["agent"], AGENTS["system"])
        content = t["content"].replace("<", "&lt;")
        box.markdown(f'<div class="tr" style="--c:{colour}"><span class="meta">step {t["step"]} · {icon} '
                     f'<b style="color:{colour}">{name}</b> · {KIND_LABEL.get(t["kind"], t["kind"])}</span>'
                     f'<br>{content}</div>', unsafe_allow_html=True)


def subtasks_view(bb: Blackboard) -> None:
    subs = bb.subtasks()
    if not subs:
        st.info("No plan yet.")
        return
    df = pd.DataFrame([{"#": s["seq"], "status": f"{STATUS_COLOUR.get(s['status'], '')} {s['status']}",
                        "query": s["query"], "attempts": s["attempts"] + 1, "databases": s["strategy"],
                        "rationale": s["rationale"]} for s in subs])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("Life-cycle: pending → open → retrieving → retrieved → processing → processed → verifying → "
               "verified | rejected → (Planner re-plans, max 3) → abandoned | unresolvable")


def sources_view(bb: Blackboard) -> None:
    src = bb.sources()
    if not src:
        st.info("No sources retrieved yet.")
        return
    df = pd.DataFrame([{"id": s["id"], "subtask": s["subtask_id"], "api": s["origin"],
                        "title": s["title"], "year": s["year"],
                        "relevance": s["relevance"], "duplicate of": s["duplicate_of"],
                        "selected": "✅" if s["selected"] else "", "doi": s["doi"], "url": s["url"]}
                       for s in src])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"relevance": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "url": st.column_config.LinkColumn()})


def claims_view(bb: Blackboard) -> None:
    claims = bb.claims()
    if not claims:
        st.info("No claims yet.")
        return
    verdicts = {v["target_id"]: v for v in bb.verdicts() if v["target_type"] == "claim"}
    sources = {s["id"]: s for s in bb.sources()}
    n_ok = sum(c["status"] == "supported" for c in claims)
    st.markdown(f"**{n_ok}/{len(claims)}** claims supported by their source. Unsupported claims never reach "
                "the report.")
    for c in claims:
        mark = {"supported": "✅", "unsupported": "❌", "pending": "⏳"}[c["status"]]
        with st.expander(f"{mark} C{c['id']}: {c['text'][:140]}"):
            v = verdicts.get(c["id"])
            st.markdown(f"**Critic:** {v['reason'] if v else 'pending'}")
            s = sources.get(c["source_id"])
            if s:
                st.markdown(f"**Source:** [{s['title']}]({s['url']}) ({s['year']}, {s['origin']})")
                st.caption((s["abstract"] or "")[:1200])


def report_view(bb: Blackboard) -> None:
    arts = {a["kind"]: Path(a["path"]) for a in bb.artefacts()}
    if "report.md" not in arts or not arts["report.md"].exists():
        syn = bb.syntheses()
        if syn:
            st.info(f"Synthesis attempt {syn[-1]['attempt']}: {syn[-1]['status']}")
        else:
            st.info("The report appears here when the Archivist has finished.")
        return
    cols = st.columns(4)
    for col, (kind, mime) in zip(cols, [("report.md", "text/markdown"), ("references.bib", "text/plain"),
                                        ("trace.jsonl", "application/json"), ("run_summary.json", "application/json")]):
        if kind in arts and arts[kind].exists():
            col.download_button(f"⬇ {kind}", arts[kind].read_bytes(), file_name=kind, mime=mime,
                                use_container_width=True)
    st.markdown(arts["report.md"].read_text(encoding="utf-8"), unsafe_allow_html=True)
    st.divider()
    q = st.text_input("🔍 Semantic search over this run's saved vector index (sqlite-vec)")
    if q and "index.sqlite" in arts:
        from scholarplan.agents.archivist import search_index
        from scholarplan.embeddings import get_embedder
        emb = get_embedder(settings0.embedding_backend, settings0.embedding_model)
        for h in search_index(arts["index.sqlite"], q, emb, k=5):
            st.markdown(f"- `{h['distance']:.3f}` **{h['kind']}** {h['citation']}: {h['text'][:220]}")


def usage_view(bb: Blackboard) -> None:
    arts = {a["kind"]: Path(a["path"]) for a in bb.artefacts()}
    if "run_summary.json" in arts and arts["run_summary.json"].exists():
        s = json.loads(arts["run_summary.json"].read_text(encoding="utf-8"))
        st.json(s, expanded=False)
    else:
        st.info("Metrics appear when the run is archived.")


@st.fragment(run_every=1.0)
def live_panel() -> None:
    run_id = st.session_state.get("run_id")
    if not run_id or not DB.exists():
        st.info("Enter a research goal in the sidebar and press **Run agents**.")
        agent_strip(None)
        return
    bb = Blackboard(DB, run_id=run_id)
    run = bb.get_run()
    if run is None:
        return
    subs, claims, sources = bb.subtasks(), bb.claims(), bb.sources()
    last = bb.trace()[-1:] or [{}]
    running = is_running() and st.session_state.get("run_id") == run_id
    # Only this fragment re-runs every second; when a run finishes, trigger
    # one full re-run so the sidebar (Stop button, run list) catches up.
    if st.session_state.get("was_running") and not running:
        st.session_state.was_running = False
        st.rerun()
    st.session_state.was_running = running

    st.markdown(f"### {'⏳' if running else ('✅' if run['status'] == 'done' else '⚠️')} {run['goal']}")
    m = st.columns(6)
    m[0].metric("Run status", run["status"])
    m[1].metric("ReAct steps", f"{run['steps_used']}/{run['step_budget']}")
    m[2].metric("Subtasks verified", f"{sum(s['status'] == 'verified' for s in subs)}/{len(subs)}")
    m[3].metric("Sources (dupes)", f"{len(sources)} ({sum(1 for s in sources if s['duplicate_of'])})")
    m[4].metric("Claims supported", f"{sum(c['status'] == 'supported' for c in claims)}/{len(claims)}")
    m[5].metric("Re-plans", sum(s["attempts"] for s in subs))
    if run.get("note") and run["status"] == "failed":
        st.error(run["note"])
    agent_strip(last[0].get("agent") if running else None)

    tabs = st.tabs(["🧠 ReAct trace", "📋 Subtasks", "📚 Sources", "🛡️ Claims & verdicts",
                    "🔁 Feedback", "📄 Report", "📈 Metrics"])
    with tabs[0]:
        show = st.multiselect("Show", list(KIND_LABEL), default=list(KIND_LABEL), key="kinds",
                              format_func=KIND_LABEL.get, label_visibility="collapsed")
        trace_view(bb, show)
    with tabs[1]:
        subtasks_view(bb)
    with tabs[2]:
        sources_view(bb)
    with tabs[3]:
        claims_view(bb)
    with tabs[4]:
        fb = bb.feedback()
        if not fb:
            st.info("No feedback yet. Feedback is written when an agent rejects work; the Planner reads it "
                    "before its next Thought (Reflexion-style re-planning).")
        for f in fb:
            icon, name, colour, _ = AGENTS.get(f["author"], AGENTS["system"])
            st.markdown(f'<div class="tr" style="--c:{colour}"><span class="meta">{icon} {name} · {f["kind"]} · '
                        f'subtask {f["subtask_id"]} · {"read by Planner" if f["consumed"] else "unread"}</span>'
                        f'<br>{f["message"]}</div>', unsafe_allow_html=True)
    with tabs[5]:
        report_view(bb)
    with tabs[6]:
        usage_view(bb)
    bb.close()


live_panel()
