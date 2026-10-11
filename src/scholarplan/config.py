"""Central configuration.

Every tunable number in ScholarPlan lives here, not scattered through the agents.
The Unit 6 feedback asked for *measurable* targets and explicit fallback rules;
keeping them in one frozen dataclass means the README, the tests and the
evaluation harness all refer to the same values, so a claim such as "at most
three re-plans per subtask" is checkable in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import so keys never have to be typed on the command line,
# where they would leak into shell history (a professional-practice concern).
load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers per agent role.

    Deviation from the Unit 6 design (documented in README section 5):
    the design named GPT-5-mini on Azure for the Planner and Processor. No Azure
    subscription was available, so the OpenAI open-weight gpt-oss family is
    served through Groq's free tier instead. The Critic keeps exactly the model
    the design chose Llama-3.3-70B-Instruct (Dubey et al., 2024) for the Critic,
    but Groq no longer offers it to new free accounts (HTTP 404 on 11 Oct 2026,
    see docs/REMEDIATION_LOG.md). The Critic therefore uses Qwen, which keeps
    the property that actually mattered in the design: the verifier comes
    from a *different model family* than the generator, so it does not share
    the generator's blind spots (Kambhampati et al., 2024). Qwen is a preview
    model on Groq, so ``critic_fallback`` is used automatically if it is
    withdrawn.
    The Planner and Processor use different sizes of the same family because
    Groq meters tokens per model; splitting them doubles usable throughput,
    mirroring the design's GPT-5-mini / GPT-5-nano split.
    """

    planner: str = "openai/gpt-oss-120b"
    processor: str = "openai/gpt-oss-20b"
    critic: str = "qwen/qwen3.8-27b"
    critic_fallback: str = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class Settings:
    # --- LLM backend -------------------------------------------------------
    llm_backend: str = "mock"  # "groq" or "mock"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    models: ModelConfig = field(default_factory=ModelConfig)
    # Low temperature reduces run-to-run variance (Unit 6 challenge:
    # non-deterministic behaviour). 0 is avoided for the Planner because a
    # little diversity helps it propose a *different* query after rejection.
    planner_temperature: float = 0.2
    worker_temperature: float = 0.0
    llm_max_retries: int = 3  # JSON-repair / HTTP retry attempts per call

    # --- ReAct control (Yao et al., 2023) ----------------------------------
    step_budget: int = 20  # hard cap on Planner Thought/Action steps per run
    max_subtasks: int = 4  # bounds decomposition (Huang et al., 2024)
    # Unit 6 feedback #3: a subtask that keeps being rejected must not loop
    # forever. After this many re-plans it is marked "unresolvable".
    max_replans_per_subtask: int = 3
    # Coverage threshold: fraction of subtasks that must be verified (or
    # formally resolved) before the Planner may synthesise.
    coverage_threshold: float = 0.75
    max_synthesis_attempts: int = 2

    # --- Retrieval ----------------------------------------------------------
    results_per_source: int = 5  # per API per subtask
    contact_email: str = ""
    http_timeout_s: float = 20.0
    # arXiv's terms of use ask for no more than one request every 3 seconds.
    arxiv_min_interval_s: float = 3.0
    # 1 request/s for OpenAlex and Crossref: well inside their published
    # limits, and slow enough that anonymous (no-mailto) use is not throttled.
    api_min_interval_s: float = 1.0

    # --- Processing (Unit 6 feedback #2: explicit thresholds) ---------------
    # Cosine similarity at or above which two records are treated as the same
    # work (e.g. an arXiv preprint and its published version).
    dedup_similarity: float = 0.92
    # Sources less similar than this to the subtask query are discarded as
    # noise before any LLM sees them (Shi et al., 2023).
    relevance_floor: float = 0.30
    top_k_sources: int = 3  # summarised per subtask
    max_claims_per_source: int = 2

    # --- Verification -------------------------------------------------------
    # A subtask passes only if at least this share of its claims is entailed.
    min_supported_ratio: float = 0.5

    # --- Demonstration ------------------------------------------------------
    # When True, the Processor appends one deliberately false claim to the
    # first subtask. It is clearly labelled in the trace and exists only to
    # demonstrate, live, that the independent Critic catches hallucinations.
    demo_inject_fabrication: bool = False

    # --- Embeddings ---------------------------------------------------------
    embedding_backend: str = "minilm"  # "minilm" or "hashing" (offline tests)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage ------------------------------------------------------------
    output_dir: Path = Path("output")
    cache_dir: Path = Path(".cache")

    def with_overrides(self, **kwargs) -> "Settings":
        """Return a copy with some fields changed (used by CLI, UI and tests)."""
        return replace(self, **kwargs)


def load_settings(**overrides) -> Settings:
    """Build Settings from environment variables, then apply explicit overrides."""
    env_backend = os.getenv("SCHOLARPLAN_LLM", "").strip().lower()
    key = os.getenv("GROQ_API_KEY", "").strip()
    # Fall back to the mock backend rather than crashing when no key exists,
    # so a marker can always run the system end-to-end offline.
    backend = env_backend or ("groq" if key else "mock")
    if backend == "groq" and not key:
        backend = "mock"
    base = Settings(
        llm_backend=backend,
        groq_api_key=key,
        contact_email=os.getenv("SCHOLARPLAN_CONTACT_EMAIL", "").strip(),
    )
    return base.with_overrides(**overrides) if overrides else base
