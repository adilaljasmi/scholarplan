"""Shared test fixtures.

Testing strategy (Unit 6, section 4, "testing by agent type"):
  * deterministic components (blackboard, API parsers, ranking, citation
    formatting, controller rules) get conventional unit tests;
  * LLM-dependent agents are tested through the deterministic MockLLM /
    ScriptedLLM, which lets us *force* the failure paths (rejection,
    re-planning, invalid JSON, budget exhaustion) that a real model would
    only produce occasionally;
  * real-service behaviour is covered by tests marked ``live``, run only with
    ``pytest --live`` because they cost API quota and are non-deterministic,
    and by the evaluation harness in ``eval/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scholarplan.blackboard import Blackboard
from scholarplan.config import Settings
from scholarplan.embeddings import HashingEmbedder

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="run tests that call Groq and the real scholarly APIs")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live test: run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Hashing embeddings: no model download, fully deterministic. Its cosine
    # values are lower than MiniLM's, hence the lower relevance floor.
    return Settings(llm_backend="mock", embedding_backend="hashing", relevance_floor=0.05,
                    output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")


@pytest.fixture
def bb() -> Blackboard:
    board = Blackboard(":memory:")
    board.create_run("test goal", step_budget=20, llm_backend="mock")
    return board


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


# ---------------------------------------------------------------- fake APIs
def make_record(i: int, title: str, abstract: str, origin: str = "arxiv", doi: str | None = None,
                authors: list[str] | None = None, year: int = 2023) -> dict[str, Any]:
    return {"origin": origin, "external_id": f"{origin}-{i}", "doi": doi, "title": title,
            "authors": authors or [f"Author{i} Example"], "year": year, "venue": "Test Venue",
            "abstract": abstract, "url": f"https://example.org/{origin}/{i}"}


# Synthetic corpus written for the tests (not copied from any paper). Each
# abstract mentions its topic words so that relevance ranking is predictable.
CORPUS = {
    "react": [
        make_record(1, "Reasoning and acting with language model agents",
                    "Language model agents interleave reasoning traces and actions. Interleaving reasoning "
                    "and acting improves task success on question answering benchmarks. The agents call "
                    "external tools such as search engines during reasoning.", doi="10.1/react"),
        make_record(2, "Tool use by language model agents",
                    "Language model agents can learn to call external tools. Tool use reduces factual errors "
                    "in question answering. Agents decide when to call a tool during reasoning.",
                    doi="10.1/tools", origin="openalex"),
    ],
    "evaluation": [
        make_record(3, "Evaluating language model agents on benchmarks",
                    "Benchmarks for language model agents measure task success and cost. Evaluation of "
                    "language model agents requires repeated runs because outputs vary. Agent evaluation "
                    "reports success rates across many tasks.", doi="10.1/eval"),
    ],
    "challenges": [
        make_record(4, "Limitations and challenges of language model agents",
                    "Language model agents hallucinate facts and fail at long horizon planning. These "
                    "challenges limit reliability of language model agents. Verification of agent outputs "
                    "mitigates hallucination.", doi="10.1/limits", origin="crossref"),
    ],
}


class FakeClient:
    """Stands in for an API client; returns corpus records whose key appears in the query."""

    def __init__(self, name: str, corpus: dict[str, list[dict[str, Any]]] | None = None):
        self.name = name
        self.corpus = CORPUS if corpus is None else corpus
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        self.queries.append(query)
        q = query.lower()
        out = []
        for key, recs in self.corpus.items():
            if key in q:
                out += [dict(r) for r in recs if r["origin"] == self.name]
        return out[:limit]


@pytest.fixture
def fake_clients() -> dict[str, FakeClient]:
    return {n: FakeClient(n) for n in ("arxiv", "openalex", "crossref")}
