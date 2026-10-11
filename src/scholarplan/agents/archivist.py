"""Archivist: the only agent that writes to disk (Unit 6, Figure 3).

Trigger: a synthesis that is ``verified`` (or a ``fallback`` synthesis).

Outputs, in ``output/<run_id>/`` (the "offline location" required by the brief):
  * report.md: the answer, with Harvard in-text citations and an evidence table;
  * references.bib: BibTeX for every source that backs a verified claim;
  * index.sqlite: a sqlite-vec vector index of verified claims and sources,
    so the evidence can be semantically searched later without re-running
    the agents (``scholarplan search``);
  * trace.jsonl: every Thought, Action, Observation, Verdict and Feedback,
    the auditable reasoning record promised for explainability;
  * run_summary.json: metrics (steps, tokens, counts) used by the evaluation.

Confining file I/O to one role keeps the side-effects of the system in one
place, which simplifies both testing and the security/ethics review: no
other agent can write outside the blackboard.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import sqlite_vec

from scholarplan.agents.base import Agent
from scholarplan.blackboard import Blackboard
from scholarplan.config import Settings
from scholarplan.embeddings import Embedder
from scholarplan.llm import LLMClient


# ------------------------------------------------------------------ citations
def surname(name: str) -> str:
    parts = name.replace(",", " ").split()
    return parts[-1] if parts else "Anon."


def initials(name: str) -> str:
    parts = name.replace(",", " ").split()[:-1]
    return "".join(f"{p[0]}." for p in parts if p and p[0].isalpha())


def harvard_intext(src: dict[str, Any]) -> str:
    """UoEO Harvard in-text form: (Surname, Year) / (A and B, Year) / (A et al., Year)."""
    a = src.get("authors") or []
    year = src.get("year_label") or src.get("year") or "n.d."
    if not a:
        return f"({src['title'][:40]}, {year})"
    if len(a) == 1:
        return f"({surname(a[0])}, {year})"
    if len(a) == 2:
        return f"({surname(a[0])} and {surname(a[1])}, {year})"
    return f"({surname(a[0])} et al., {year})"


def disambiguate_years(sources: list[dict[str, Any]]) -> None:
    """Add 2024a / 2024b labels when two cited works would share an in-text
    citation, as Harvard requires; otherwise readers cannot tell them apart."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in sources:
        s.pop("year_label", None)
        groups.setdefault(harvard_intext(s), []).append(s)
    for same in groups.values():
        if len(same) > 1:
            for i, s in enumerate(sorted(same, key=lambda x: x["title"].lower())):
                s["year_label"] = f"{s.get('year') or 'n.d.'}{chr(97 + i)}"


def harvard_reference(src: dict[str, Any], accessed: str) -> str:
    """UoEO Harvard reference-list entry."""
    a = src.get("authors") or []
    names = [f"{surname(n)}, {initials(n)}".strip() for n in a]
    if len(names) > 3:
        author = f"{names[0]} et al."
    elif len(names) > 1:
        author = ", ".join(names[:-1]) + " and " + names[-1]
    else:
        author = names[0] if names else "Anon."
    venue = f", {src['venue']}." if src.get("venue") else "."
    link = f"https://doi.org/{src['doi']}" if src.get("doi") else (src.get("url") or "")
    year = src.get("year_label") or src.get("year") or "n.d."
    return f"{author} ({year}) '{src['title']}'{venue} Available at: {link} (Accessed: {accessed})."


def bibtex_key(src: dict[str, Any]) -> str:
    first = re.sub(r"[^a-z]", "", surname((src.get("authors") or ["anon"])[0]).lower()) or "anon"
    word = next((w for w in re.findall(r"[a-z]+", src["title"].lower()) if len(w) > 3), "work")
    return f"{first}{src.get('year') or 'nd'}{word}"


def bibtex_entry(src: dict[str, Any], key: str) -> str:
    def esc(s: str) -> str:
        return s.replace("{", "\\{").replace("}", "\\}")
    is_arxiv = src["origin"] == "arxiv" and not src.get("doi")
    fields = {"title": "{" + esc(src["title"]) + "}",
              "author": " and ".join(src.get("authors") or ["Anonymous"]),
              "year": str(src.get("year") or "")}
    if is_arxiv:
        kind = "misc"
        fields.update(eprint=src.get("external_id") or "", archivePrefix="arXiv", url=src.get("url") or "")
    else:
        kind = "article"
        if src.get("venue"):
            fields["journal"] = esc(src["venue"])
        if src.get("doi"):
            fields["doi"] = src["doi"]
        if src.get("url"):
            fields["url"] = src["url"]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}"


# ------------------------------------------------------------- vector index
def build_index(path: Path, items: list[dict[str, Any]], embedder: Embedder) -> int:
    """Write a sqlite-vec index. sqlite-vec was chosen in Unit 6 so the index is a
    single portable file in the same technology as the blackboard."""
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, kind TEXT, text TEXT, citation TEXT, link TEXT)")
    conn.execute(f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{embedder.dim}])")
    if items:
        vecs = embedder.embed([it["text"] for it in items])
        for i, (it, v) in enumerate(zip(items, vecs), start=1):
            conn.execute("INSERT INTO items VALUES (?,?,?,?,?)",
                         (i, it["kind"], it["text"], it["citation"], it["link"]))
            conn.execute("INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
                         (i, sqlite_vec.serialize_float32(v.tolist())))
    conn.commit()
    conn.close()
    return len(items)


def search_index(path: Path, query: str, embedder: Embedder, k: int = 5) -> list[dict[str, Any]]:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    q = sqlite_vec.serialize_float32(embedder.embed([query])[0].tolist())
    rows = conn.execute(
        "SELECT items.kind, items.text, items.citation, items.link, v.distance FROM "
        "(SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?) v "
        "JOIN items ON items.id = v.rowid ORDER BY v.distance", (q, k)).fetchall()
    conn.close()
    return [dict(zip(("kind", "text", "citation", "link", "distance"), r)) for r in rows]


class ArchivistAgent(Agent):
    role = "archivist"

    def __init__(self, settings: Settings, embedder: Embedder, llm: LLMClient | None = None):
        super().__init__(settings)
        self.embedder = embedder
        self.llm = llm  # only read for token-usage reporting

    def can_act(self, bb: Blackboard) -> bool:
        run = bb.get_run()
        return (run is not None and run["status"] not in ("done", "failed")
                and any(s["status"] in ("verified", "fallback") for s in bb.syntheses()))

    def act(self, bb: Blackboard) -> str:
        sb = self.scoped(bb)
        run = sb.get_run()
        out = Path(self.settings.output_dir) / run["id"]
        out.mkdir(parents=True, exist_ok=True)
        accessed = date.today().strftime("%d %B %Y").lstrip("0")

        syn = next(s for s in reversed(sb.syntheses()) if s["status"] in ("verified", "fallback"))
        claims = sb.claims(status="supported")
        sources = {s["id"]: s for s in sb.sources()}
        cited_ids = sorted({c["source_id"] for c in claims})
        cited = [sources[i] for i in cited_ids]
        disambiguate_years(cited)  # mutates the shared dicts, so all later citations agree

        # BibTeX, de-duplicated by DOI and by key (same work found by two APIs).
        keys: dict[int, str] = {}
        bib, seen_doi, used_keys = [], set(), set()
        for s in cited:
            if s.get("doi") and s["doi"] in seen_doi:
                continue
            key = base = bibtex_key(s)
            n = 1
            while key in used_keys:
                n += 1
                key = f"{base}{chr(96 + n)}"
            used_keys.add(key)
            seen_doi.add(s.get("doi"))
            keys[s["id"]] = key
            bib.append(bibtex_entry(s, key))
        (out / "references.bib").write_text("\n\n".join(bib) + "\n", encoding="utf-8")

        report = self._report(run, syn, claims, sources, cited, accessed, sb)
        (out / "report.md").write_text(report, encoding="utf-8")

        items = [{"kind": "claim", "text": c["text"], "citation": harvard_intext(sources[c["source_id"]]),
                  "link": sources[c["source_id"]].get("url") or ""} for c in claims]
        items += [{"kind": "source", "text": f"{s['title']}. {s.get('abstract') or ''}"[:2000],
                   "citation": harvard_intext(s), "link": s.get("url") or ""} for s in cited]
        n_idx = build_index(out / "index.sqlite", items, self.embedder)

        with (out / "trace.jsonl").open("w", encoding="utf-8") as fh:
            for t in sb.trace():
                fh.write(json.dumps({k: t[k] for k in ("id", "step", "agent", "kind", "content", "created_at")}
                                    | {"data": json.loads(t["data"]) if t["data"] else None}) + "\n")

        summary = self._summary(run, sb, n_idx, syn)
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        for kind in ("report.md", "references.bib", "index.sqlite", "trace.jsonl", "run_summary.json"):
            sb.add_artefact(kind, str(out / kind))
        try:  # log a relative path: absolute paths leak the user's account name into shared traces
            shown = out.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            shown = Path(out.name)
        sb.log("action", f"wrote report, {len(bib)} BibTeX entries, {n_idx} index vectors and trace to {shown}")
        sb.set_run_status("done")
        return f"archived to {out}"

    # ------------------------------------------------------------ helpers
    def _report(self, run, syn, claims, sources, cited, accessed, sb) -> str:
        def render(text: str) -> str:
            # Replace [C12] markers with Harvard in-text citations for readers,
            # while keeping the claim id so every statement stays auditable.
            by_id = {c["id"]: c for c in claims}

            def sub(m: re.Match) -> str:
                c = by_id.get(int(m.group(1)))
                return f"{harvard_intext(sources[c['source_id']])}<sup>C{c['id']}</sup>" if c else m.group(0)
            return re.sub(r"\[C(\d+)\]", sub, text)

        subtasks = sb.subtasks()
        lines = [f"# ScholarPlan research report", "",
                 f"**Research goal:** {run['goal']}  ",
                 f"**Run:** `{run['id']}` | **Date:** {accessed} | **LLM backend:** {run['llm_backend']} | "
                 f"**Synthesis status:** {syn['status']}", "",
                 "> Generated by an LLM-based multi-agent system. Every statement below cites a claim that an "
                 "independent Critic model judged to be supported by the cited abstract. Abstract-level "
                 "verification is not a substitute for reading the full papers.", "",
                 "## Answer", "", render(syn["text"]), "",
                 "## Evidence (verified claims)", "",
                 "| ID | Claim | Source |", "|---|---|---|"]
        for c in claims:
            s = sources[c["source_id"]]
            lines.append(f"| C{c['id']} | {c['text'].replace('|', '/')} | {harvard_intext(s)} |")
        lines += ["", "## How the evidence was gathered", "",
                  "| # | Subtask query | Outcome | Attempts |", "|---|---|---|---|"]
        for s in subtasks:
            lines.append(f"| {s['seq']} | {s['query']} | {s['status']} | {s['attempts'] + 1} |")
        rejected = sb.claims(status="unsupported")
        lines += ["", f"Claims rejected by the Critic and excluded from this report: **{len(rejected)}**.", "",
                  "## References", ""]
        # Harvard lists references alphabetically by first author, then by year label (2024a before 2024b).
        def ref_order(x: dict[str, Any]) -> tuple[str, str]:
            return surname((x.get("authors") or ["~"])[0]).lower(), str(x.get("year_label") or x.get("year") or "")
        lines += [harvard_reference(s, accessed) for s in sorted(cited, key=ref_order)]
        return "\n".join(lines) + "\n"

    def _summary(self, run, sb, n_idx, syn) -> dict[str, Any]:
        subtasks = sb.subtasks()
        sources = sb.sources()
        usage = getattr(self.llm, "usage", None)
        return {
            "run_id": run["id"], "goal": run["goal"], "llm_backend": run["llm_backend"],
            "steps_used": sb.get_run()["steps_used"], "step_budget": run["step_budget"],
            "subtasks": {st: sum(1 for s in subtasks if s["status"] == st)
                         for st in sorted({s["status"] for s in subtasks})},
            "replans": sum(s["attempts"] for s in subtasks),
            "sources_retrieved": len(sources),
            "duplicates_removed": sum(1 for s in sources if s["duplicate_of"]),
            "sources_selected": sum(1 for s in sources if s["selected"]),
            "claims_total": len(sb.claims()),
            "claims_supported": len(sb.claims(status="supported")),
            "claims_unsupported": len(sb.claims(status="unsupported")),
            "synthesis_attempts": len(sb.syntheses()), "synthesis_status": syn["status"],
            "index_vectors": n_idx,
            "llm_usage": ({"calls": usage.calls, "prompt_tokens": usage.prompt_tokens,
                           "completion_tokens": usage.completion_tokens, "by_model": usage.by_model}
                          if usage else None),
        }
