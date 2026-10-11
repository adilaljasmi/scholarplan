"""The shared blackboard: the only channel through which agents cooperate.

Design rationale
----------------
* Agents never call one another; they post and read typed entries here. The
  blackboard model decouples heterogeneous knowledge sources and lets control
  be opportunistic (Hayes-Roth, 1985), and it has recently outperformed a
  central orchestrator for LLM agents doing information discovery
  (Salemi et al., 2025).
* SQLite was chosen over an in-memory dict because every intermediate artefact
  (plans, sources, claims, verdicts, feedback) then persists and can be
  inspected after a run. That is the explainability mechanism promised in the
  Unit 6 design, and it is what the Streamlit dashboard reads live.
* The blackboard is a single point of failure (Salemi et al., 2025). We contain
  that risk with transactions (``BEGIN IMMEDIATE``) so that two agents can never
  claim the same subtask, and with WAL mode so the dashboard can read while the
  agents write.
* Gaia methodology (Wooldridge, Jennings and Kinny, 2000): each role has
  explicit *permissions*. :class:`ScopedBlackboard` enforces them at runtime,
  so a Retriever that tried to write a verdict would fail loudly instead of
  silently corrupting shared state.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    goal         TEXT NOT NULL,
    status       TEXT NOT NULL,          -- planning|running|synthesising|archiving|done|failed
    step_budget  INTEGER NOT NULL,
    steps_used   INTEGER NOT NULL DEFAULT 0,
    llm_backend  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    note         TEXT
);
CREATE TABLE IF NOT EXISTS subtasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id),
    seq         INTEGER NOT NULL,
    query       TEXT NOT NULL,
    rationale   TEXT,
    status      TEXT NOT NULL,           -- see SUBTASK_STATES
    attempts    INTEGER NOT NULL DEFAULT 0,
    strategy    TEXT NOT NULL DEFAULT 'all', -- which APIs the Retriever should query
    owner       TEXT,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    subtask_id   INTEGER NOT NULL,
    attempt      INTEGER NOT NULL,
    origin       TEXT NOT NULL,          -- arxiv|openalex|crossref
    external_id  TEXT,
    doi          TEXT,
    title        TEXT NOT NULL,
    authors      TEXT NOT NULL,          -- JSON list
    year         INTEGER,
    venue        TEXT,
    abstract     TEXT,
    url          TEXT,
    fetched_at   TEXT NOT NULL,          -- provenance captured at fetch time
    relevance    REAL,
    duplicate_of INTEGER,
    selected     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    subtask_id  INTEGER NOT NULL,
    source_id   INTEGER NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    subtask_id  INTEGER NOT NULL,
    source_id   INTEGER NOT NULL,
    attempt     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'   -- pending|supported|unsupported
);
CREATE TABLE IF NOT EXISTS verdicts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    target_type  TEXT NOT NULL,          -- claim|synthesis
    target_id    INTEGER NOT NULL,
    label        TEXT NOT NULL,          -- supported|unsupported
    reason       TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    subtask_id  INTEGER,                 -- NULL when it concerns the synthesis
    author      TEXT NOT NULL,
    kind        TEXT NOT NULL,           -- no_results|no_relevant|unsupported|synthesis_rejected
    message     TEXT NOT NULL,
    consumed    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS syntheses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    attempt     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|verified|rejected|fallback
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trace (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    step        INTEGER NOT NULL,
    agent       TEXT NOT NULL,
    kind        TEXT NOT NULL,           -- thought|action|observation|verdict|feedback|event|error
    content     TEXT NOT NULL,
    data        TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artefacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_subtasks_run ON subtasks(run_id, status);
CREATE INDEX IF NOT EXISTS ix_sources_sub ON sources(subtask_id, attempt);
CREATE INDEX IF NOT EXISTS ix_claims_sub ON claims(subtask_id, attempt);
CREATE INDEX IF NOT EXISTS ix_trace_run ON trace(run_id, id);
"""

# Subtask life-cycle. Each agent is *triggered* by one state and moves the
# subtask to the next, which is what makes the specialists reactive
# (Wooldridge, 2009): they act only when their input appears.
SUBTASK_STATES = (
    "pending",       # in the plan, not yet issued by the Planner
    "open",          # issued: Retriever's trigger
    "retrieving",
    "retrieved",     # Processor's trigger
    "processing",
    "processed",     # Critic's trigger
    "verifying",
    "verified",      # terminal (success)
    "rejected",      # Planner's trigger: read feedback, then re-plan
    "abandoned",     # terminal (Planner judged it not worth pursuing)
    "unresolvable",  # terminal (re-plan cap reached, Unit 6 feedback #3)
)
TERMINAL_STATES = frozenset({"verified", "abandoned", "unresolvable"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def writes(*tables: str) -> Callable:
    """Tag a Blackboard method with the tables it mutates.

    The tag is metadata only; :class:`ScopedBlackboard` reads it to enforce the
    Gaia permission model without duplicating the method list.
    """

    def deco(fn: Callable) -> Callable:
        fn._writes = frozenset(tables)  # type: ignore[attr-defined]
        return fn

    return deco


# Gaia "permissions": which shared tables each role may change. Derived
# directly from the Unit 6 sequence diagram (Figure 3). Trace is writable by
# everyone because explainability requires every role to record its reasoning.
PERMISSIONS: dict[str, frozenset[str]] = {
    "planner": frozenset({"runs", "subtasks", "feedback", "syntheses", "trace"}),
    "retriever": frozenset({"subtasks", "sources", "feedback", "trace"}),
    "processor": frozenset({"subtasks", "sources", "summaries", "claims", "feedback", "trace"}),
    "critic": frozenset({"subtasks", "claims", "verdicts", "feedback", "syntheses", "trace"}),
    "archivist": frozenset({"runs", "artefacts", "trace"}),
    "system": frozenset({"runs", "subtasks", "sources", "summaries", "claims", "verdicts",
                         "feedback", "syntheses", "trace", "artefacts"}),
}


class Blackboard:
    """Typed access to the SQLite blackboard for one run."""

    def __init__(self, db_path: str | Path, run_id: str | None = None):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None gives explicit control of transactions, which
        # we need for the atomic "claim" operations below.
        self.conn = sqlite3.connect(self.db_path, isolation_level=None,
                                    check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if self.db_path != ":memory:":
            # WAL lets the dashboard read while the agent thread writes.
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._lock = threading.RLock()
        self.run_id = run_id
        self.current_step = 0

    # ------------------------------------------------------------------ utils
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """IMMEDIATE transaction: takes the write lock up front so a
        read-modify-write cannot interleave with another writer."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def _rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def _row(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = self._rows(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------- runs
    @writes("runs")
    def create_run(self, goal: str, step_budget: int, llm_backend: str) -> str:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self.transaction() as c:
            c.execute("INSERT INTO runs (id, goal, status, step_budget, llm_backend, created_at)"
                      " VALUES (?,?,?,?,?,?)",
                      (run_id, goal, "planning", step_budget, llm_backend, _now()))
        self.run_id = run_id
        return run_id

    def get_run(self, run_id: str | None = None) -> dict[str, Any] | None:
        return self._row("SELECT * FROM runs WHERE id=?", (run_id or self.run_id,))

    def list_runs(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM runs ORDER BY created_at DESC")

    @writes("runs")
    def set_run_status(self, status: str, note: str | None = None) -> None:
        finished = _now() if status in ("done", "failed") else None
        with self.transaction() as c:
            c.execute("UPDATE runs SET status=?, note=COALESCE(?, note),"
                      " finished_at=COALESCE(?, finished_at) WHERE id=?",
                      (status, note, finished, self.run_id))

    @writes("runs")
    def consume_step(self) -> int:
        """Spend one unit of the ReAct step budget and return the new count."""
        with self.transaction() as c:
            c.execute("UPDATE runs SET steps_used = steps_used + 1 WHERE id=?", (self.run_id,))
            used = c.execute("SELECT steps_used FROM runs WHERE id=?", (self.run_id,)).fetchone()[0]
        self.current_step = used
        return used

    def budget_left(self) -> int:
        r = self.get_run()
        return (r["step_budget"] - r["steps_used"]) if r else 0

    # --------------------------------------------------------------- subtasks
    @writes("subtasks")
    def add_subtask(self, seq: int, query: str, rationale: str = "", status: str = "pending") -> int:
        with self.transaction() as c:
            cur = c.execute("INSERT INTO subtasks (run_id, seq, query, rationale, status, updated_at)"
                            " VALUES (?,?,?,?,?,?)",
                            (self.run_id, seq, query, rationale, status, _now()))
            return int(cur.lastrowid)

    def subtasks(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return self._rows("SELECT * FROM subtasks WHERE run_id=? AND status=? ORDER BY seq",
                              (self.run_id, status))
        return self._rows("SELECT * FROM subtasks WHERE run_id=? ORDER BY seq", (self.run_id,))

    def get_subtask(self, subtask_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM subtasks WHERE id=?", (subtask_id,))

    @writes("subtasks")
    def claim_subtask(self, from_status: str, to_status: str, owner: str) -> dict[str, Any] | None:
        """Atomically move the next subtask in ``from_status`` to ``to_status``.

        Returns the claimed row, or None if nothing was waiting. Because the
        SELECT and UPDATE share one IMMEDIATE transaction and the UPDATE
        re-checks the status, two agents can never claim the same entry: this
        is the "atomic claim" mitigation from the Unit 6 challenges list.
        """
        with self.transaction() as c:
            row = c.execute("SELECT id FROM subtasks WHERE run_id=? AND status=? ORDER BY seq LIMIT 1",
                            (self.run_id, from_status)).fetchone()
            if row is None:
                return None
            cur = c.execute("UPDATE subtasks SET status=?, owner=?, updated_at=? WHERE id=? AND status=?",
                            (to_status, owner, _now(), row["id"], from_status))
            if cur.rowcount != 1:
                return None
        return self.get_subtask(row["id"])

    @writes("subtasks")
    def set_subtask_status(self, subtask_id: int, status: str, query: str | None = None,
                           bump_attempt: bool = False, strategy: str | None = None) -> None:
        if status not in SUBTASK_STATES:
            raise ValueError(f"unknown subtask status {status!r}")
        with self.transaction() as c:
            c.execute("UPDATE subtasks SET status=?, query=COALESCE(?, query),"
                      " strategy=COALESCE(?, strategy), attempts = attempts + ?, updated_at=?"
                      " WHERE id=?",
                      (status, query, strategy, 1 if bump_attempt else 0, _now(), subtask_id))

    # ---------------------------------------------------------------- sources
    @writes("sources")
    def add_source(self, subtask_id: int, attempt: int, rec: dict[str, Any]) -> int:
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO sources (run_id, subtask_id, attempt, origin, external_id, doi, title,"
                " authors, year, venue, abstract, url, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, subtask_id, attempt, rec["origin"], rec.get("external_id"),
                 rec.get("doi"), rec["title"], json.dumps(rec.get("authors", [])), rec.get("year"),
                 rec.get("venue"), rec.get("abstract"), rec.get("url"),
                 rec.get("fetched_at") or _now()))
            return int(cur.lastrowid)

    def sources(self, subtask_id: int | None = None, attempt: int | None = None,
                selected_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sources WHERE run_id=?"
        params: list[Any] = [self.run_id]
        if subtask_id is not None:
            sql += " AND subtask_id=?"
            params.append(subtask_id)
        if attempt is not None:
            sql += " AND attempt=?"
            params.append(attempt)
        if selected_only:
            sql += " AND selected=1"
        rows = self._rows(sql + " ORDER BY id", tuple(params))
        for r in rows:
            r["authors"] = json.loads(r["authors"] or "[]")
        return rows

    def get_source(self, source_id: int) -> dict[str, Any] | None:
        r = self._row("SELECT * FROM sources WHERE id=?", (source_id,))
        if r:
            r["authors"] = json.loads(r["authors"] or "[]")
        return r

    @writes("sources")
    def annotate_source(self, source_id: int, relevance: float | None = None,
                        duplicate_of: int | None = None, selected: bool | None = None) -> None:
        with self.transaction() as c:
            c.execute("UPDATE sources SET relevance=COALESCE(?, relevance),"
                      " duplicate_of=COALESCE(?, duplicate_of), selected=COALESCE(?, selected) WHERE id=?",
                      (relevance, duplicate_of, None if selected is None else int(selected), source_id))

    # -------------------------------------------------------- summaries/claims
    @writes("summaries")
    def add_summary(self, subtask_id: int, source_id: int, text: str) -> int:
        with self.transaction() as c:
            cur = c.execute("INSERT INTO summaries (run_id, subtask_id, source_id, text, created_at)"
                            " VALUES (?,?,?,?,?)", (self.run_id, subtask_id, source_id, text, _now()))
            return int(cur.lastrowid)

    def summaries(self, subtask_id: int | None = None) -> list[dict[str, Any]]:
        if subtask_id is None:
            return self._rows("SELECT * FROM summaries WHERE run_id=? ORDER BY id", (self.run_id,))
        return self._rows("SELECT * FROM summaries WHERE subtask_id=? ORDER BY id", (subtask_id,))

    @writes("claims")
    def add_claim(self, subtask_id: int, source_id: int, attempt: int, text: str) -> int:
        with self.transaction() as c:
            cur = c.execute("INSERT INTO claims (run_id, subtask_id, source_id, attempt, text)"
                            " VALUES (?,?,?,?,?)", (self.run_id, subtask_id, source_id, attempt, text))
            return int(cur.lastrowid)

    def claims(self, subtask_id: int | None = None, attempt: int | None = None,
               status: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM claims WHERE run_id=?", [self.run_id]
        if subtask_id is not None:
            sql += " AND subtask_id=?"
            params.append(subtask_id)
        if attempt is not None:
            sql += " AND attempt=?"
            params.append(attempt)
        if status is not None:
            sql += " AND status=?"
            params.append(status)
        return self._rows(sql + " ORDER BY id", tuple(params))

    @writes("claims", "verdicts")
    def record_claim_verdict(self, claim_id: int, label: str, reason: str) -> None:
        if label not in ("supported", "unsupported"):
            raise ValueError(f"invalid verdict label {label!r}")
        with self.transaction() as c:
            c.execute("UPDATE claims SET status=? WHERE id=?", (label, claim_id))
            c.execute("INSERT INTO verdicts (run_id, target_type, target_id, label, reason, created_at)"
                      " VALUES (?,?,?,?,?,?)", (self.run_id, "claim", claim_id, label, reason, _now()))

    def verdicts(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM verdicts WHERE run_id=? ORDER BY id", (self.run_id,))

    # --------------------------------------------------------------- feedback
    @writes("feedback")
    def add_feedback(self, author: str, kind: str, message: str, subtask_id: int | None = None) -> int:
        with self.transaction() as c:
            cur = c.execute("INSERT INTO feedback (run_id, subtask_id, author, kind, message, created_at)"
                            " VALUES (?,?,?,?,?,?)",
                            (self.run_id, subtask_id, author, kind, message, _now()))
            return int(cur.lastrowid)

    def feedback(self, unconsumed_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM feedback WHERE run_id=?" + (" AND consumed=0" if unconsumed_only else "")
        return self._rows(sql + " ORDER BY id", (self.run_id,))

    @writes("feedback")
    def consume_feedback(self, feedback_ids: list[int]) -> None:
        with self.transaction() as c:
            c.executemany("UPDATE feedback SET consumed=1 WHERE id=?", [(i,) for i in feedback_ids])

    # -------------------------------------------------------------- synthesis
    @writes("syntheses")
    def add_synthesis(self, attempt: int, text: str, status: str = "pending") -> int:
        with self.transaction() as c:
            cur = c.execute("INSERT INTO syntheses (run_id, attempt, text, status, created_at)"
                            " VALUES (?,?,?,?,?)", (self.run_id, attempt, text, status, _now()))
            return int(cur.lastrowid)

    def syntheses(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM syntheses WHERE run_id=? ORDER BY id", (self.run_id,))

    @writes("syntheses", "verdicts")
    def record_synthesis_verdict(self, synthesis_id: int, label: str, reason: str) -> None:
        status = "verified" if label == "supported" else "rejected"
        with self.transaction() as c:
            c.execute("UPDATE syntheses SET status=? WHERE id=?", (status, synthesis_id))
            c.execute("INSERT INTO verdicts (run_id, target_type, target_id, label, reason, created_at)"
                      " VALUES (?,?,?,?,?,?)", (self.run_id, "synthesis", synthesis_id, label, reason, _now()))

    # ------------------------------------------------------------------ trace
    @writes("trace")
    def log(self, agent: str, kind: str, content: str, data: Any = None) -> None:
        """Append to the execution trace (Thought / Action / Observation / ...).

        Persisting every reasoning step is how ScholarPlan answers the
        explainability challenge: the chain of decisions is auditable after
        the fact, in the spirit of chain-of-thought (Wei et al., 2022).
        """
        with self.transaction() as c:
            c.execute("INSERT INTO trace (run_id, step, agent, kind, content, data, created_at)"
                      " VALUES (?,?,?,?,?,?,?)",
                      (self.run_id, self.current_step, agent, kind, content,
                       json.dumps(data, default=str) if data is not None else None, _now()))

    def trace(self, after_id: int = 0) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM trace WHERE run_id=? AND id>? ORDER BY id", (self.run_id, after_id))

    # -------------------------------------------------------------- artefacts
    @writes("artefacts")
    def add_artefact(self, kind: str, path: str) -> None:
        with self.transaction() as c:
            c.execute("INSERT INTO artefacts (run_id, kind, path) VALUES (?,?,?)", (self.run_id, kind, path))

    def artefacts(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM artefacts WHERE run_id=? ORDER BY id", (self.run_id,))

    # ----------------------------------------------------------- permissions
    def view(self, role: str) -> "ScopedBlackboard":
        return ScopedBlackboard(self, role)


class ScopedBlackboard:
    """A role's window onto the blackboard (Gaia permissions enforced).

    Reads are always allowed; any method tagged with ``@writes`` is checked
    against :data:`PERMISSIONS`. ``log`` is bound to the role automatically so
    the trace always records *who* reasoned.
    """

    def __init__(self, bb: Blackboard, role: str):
        if role not in PERMISSIONS:
            raise ValueError(f"unknown role {role!r}")
        self._bb = bb
        self.role = role

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._bb, name)
        needed = getattr(attr, "_writes", None)
        if needed is not None and not needed <= PERMISSIONS[self.role]:
            missing = ", ".join(sorted(needed - PERMISSIONS[self.role]))
            raise PermissionError(f"role '{self.role}' may not write to: {missing}")
        return attr

    def log(self, kind: str, content: str, data: Any = None) -> None:
        self._bb.log(self.role, kind, content, data)
