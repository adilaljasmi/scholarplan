"""Blackboard: persistence, atomic claims, Gaia permissions, trace."""

from __future__ import annotations

import threading

import pytest

from scholarplan.blackboard import PERMISSIONS, Blackboard


def test_run_lifecycle_and_step_budget(bb):
    assert bb.get_run()["status"] == "planning"
    assert bb.budget_left() == 20
    assert bb.consume_step() == 1
    assert bb.budget_left() == 19
    bb.set_run_status("done")
    assert bb.get_run()["finished_at"] is not None


def test_claim_is_atomic_single_thread(bb):
    sid = bb.add_subtask(1, "q", status="open")
    first = bb.claim_subtask("open", "retrieving", "retriever")
    second = bb.claim_subtask("open", "retrieving", "retriever")
    assert first["id"] == sid and first["status"] == "retrieving"
    assert second is None  # the same entry can never be claimed twice


def test_claim_is_atomic_under_concurrency(tmp_path):
    """Eight threads race for one open subtask; exactly one must win."""
    db = tmp_path / "bb.db"
    board = Blackboard(db)
    board.create_run("race", 20, "mock")
    board.add_subtask(1, "q", status="open")
    wins: list[int] = []

    def worker():
        mine = Blackboard(db, run_id=board.run_id)  # separate connection, as a real agent process would have
        if mine.claim_subtask("open", "retrieving", "retriever"):
            wins.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1


def test_claim_order_follows_plan_sequence(bb):
    bb.add_subtask(2, "second", status="open")
    bb.add_subtask(1, "first", status="open")
    assert bb.claim_subtask("open", "retrieving", "retriever")["query"] == "first"


def test_permissions_block_out_of_role_writes(bb):
    sid = bb.add_subtask(1, "q")
    src = bb.add_source(sid, 0, {"origin": "arxiv", "title": "t", "authors": []})
    cid = bb.add_claim(sid, src, 0, "a claim")
    with pytest.raises(PermissionError, match="retriever"):
        bb.view("retriever").record_claim_verdict(cid, "supported", "x")
    with pytest.raises(PermissionError):
        bb.view("critic").add_source(sid, 0, {"origin": "arxiv", "title": "t"})
    with pytest.raises(PermissionError):
        bb.view("planner").add_artefact("report", "x")
    bb.view("critic").record_claim_verdict(cid, "supported", "ok")  # allowed
    assert bb.claims(status="supported")[0]["id"] == cid


def test_reads_are_always_allowed(bb):
    assert bb.view("archivist").subtasks() == []


def test_unknown_role_rejected(bb):
    with pytest.raises(ValueError):
        bb.view("intruder")


def test_every_role_can_write_trace():
    assert all("trace" in perms for perms in PERMISSIONS.values())


def test_scoped_log_records_role(bb):
    bb.view("critic").log("verdict", "C1 SUPPORTED")
    t = bb.trace()[-1]
    assert (t["agent"], t["kind"], t["content"]) == ("critic", "verdict", "C1 SUPPORTED")


def test_invalid_states_and_labels_rejected(bb):
    sid = bb.add_subtask(1, "q")
    with pytest.raises(ValueError):
        bb.set_subtask_status(sid, "teleported")
    with pytest.raises(ValueError):
        bb.record_claim_verdict(1, "maybe", "")


def test_replan_updates_query_attempts_and_strategy(bb):
    sid = bb.add_subtask(1, "old")
    bb.set_subtask_status(sid, "open", query="new", bump_attempt=True, strategy="openalex")
    st = bb.get_subtask(sid)
    assert (st["query"], st["attempts"], st["strategy"], st["status"]) == ("new", 1, "openalex", "open")


def test_feedback_consumption(bb):
    f1 = bb.add_feedback("critic", "unsupported", "bad", subtask_id=1)
    bb.add_feedback("retriever", "no_results", "none", subtask_id=2)
    bb.consume_feedback([f1])
    left = bb.feedback(unconsumed_only=True)
    assert [f["kind"] for f in left] == ["no_results"]


def test_transaction_rolls_back_on_error(bb):
    with pytest.raises(RuntimeError):
        with bb.transaction() as c:
            c.execute("INSERT INTO subtasks (run_id, seq, query, status, updated_at) VALUES (?,?,?,?,?)",
                      (bb.run_id, 1, "q", "pending", "now"))
            raise RuntimeError("boom")
    assert bb.subtasks() == []
