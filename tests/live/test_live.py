"""Live smoke tests against the real services. Run with:  pytest --live

They are excluded from the default run because they need GROQ_API_KEY and
network access, consume free-tier quota, and depend on external data that can
change. Their purpose is to detect *drift* (an API changing its format, a
model being withdrawn, see remediation R8) rather than to test our logic.
"""

from __future__ import annotations

import pytest

from scholarplan.agents import prompts
from scholarplan.config import load_settings
from scholarplan.llm import build_llm, call_json
from scholarplan.llm.schemas import CriticOutput
from scholarplan.sources import build_clients

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_settings():
    s = load_settings()
    if s.llm_backend != "groq":
        pytest.skip("GROQ_API_KEY not configured")
    return s


@pytest.mark.parametrize("api", ["arxiv", "openalex", "crossref"])
def test_each_api_still_returns_parseable_records(api, tmp_path):
    s = load_settings(cache_dir=tmp_path)
    recs = build_clients(s)[api].search("retrieval augmented generation hallucination", 3)
    assert recs and all(r["title"] for r in recs)


def test_live_critic_separates_supported_from_fabricated(live_settings):
    source = ("ReAct generates reasoning traces and task-specific actions in an interleaved manner. "
              "On ALFWorld and WebShop it was prompted with only one or two in-context examples.")
    items = [{"claim_id": 1, "claim": "ReAct interleaves reasoning traces with actions.", "source_text": source},
             {"claim_id": 2, "claim": "ReAct needs fifty in-context examples.", "source_text": source}]
    out = call_json(build_llm(live_settings), CriticOutput, task="verify_claims",
                    model=live_settings.models.critic, system=prompts.VERIFY_CLAIMS,
                    payload={"items": items}, temperature=0)
    labels = {v.claim_id: v.label for v in out.verdicts}
    assert labels == {1: "supported", 2: "unsupported"}
