"""LLM layer: JSON extraction, validate-and-retry, live client protocol (mocked transport)."""

from __future__ import annotations

import json

import httpx
import pytest

from scholarplan.llm import LLMError, ScriptedLLM, call_json
from scholarplan.llm.base import extract_json
from scholarplan.llm.openai_compat import OpenAICompatibleClient
from scholarplan.llm.schemas import PlanOutput, ReActOutput


@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here is the JSON: {"a": 1} Hope this helps.',
])
def test_extract_json_tolerates_wrappers(raw):
    assert extract_json(raw) == {"a": 1}


def test_call_json_repairs_invalid_output():
    llm = ScriptedLLM({"plan": ["not json at all",
                                {"thought": "t", "subtasks": [{"query": "valid query"}]}]})
    out = call_json(llm, PlanOutput, task="plan", model="m", system="s", payload={"goal": "g"},
                    temperature=0)
    assert out.subtasks[0].query == "valid query"
    assert len(llm.calls) == 2  # one failure, one successful repair


def test_call_json_gives_up_after_retries():
    llm = ScriptedLLM({"react": ['{"thought": "x", "action": "fly_to_moon"}'] * 3})
    with pytest.raises(LLMError, match="ReActOutput"):
        call_json(llm, ReActOutput, task="react", model="m", system="s", payload={}, temperature=0,
                  max_retries=3)


def test_schema_rejects_empty_plan():
    with pytest.raises(Exception):
        PlanOutput.model_validate({"thought": "t", "subtasks": []})


def _ok(content: str, prompt=10, completion=5):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                     "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})


def _client(handler, **kw):
    return OpenAICompatibleClient("https://llm.test/v1", kw.pop("key", "KEY"), min_interval_s=0,
                                  transport=httpx.MockTransport(handler), **kw)


def test_live_client_request_shape_and_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _ok('{"ok": true}')

    c = _client(handler)
    out = c.complete(task="plan", model="openai/gpt-oss-120b", system="s", user="u", temperature=0.2)
    assert out == '{"ok": true}'
    assert seen["auth"] == "Bearer KEY"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["reasoning_effort"] == "low"  # only for reasoning (gpt-oss) models
    assert c.usage.by_model["openai/gpt-oss-120b"] == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5}


def test_live_client_omits_reasoning_effort_for_other_models():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok("{}")

    _client(handler).complete(task="verify_claims", model="qwen/qwen3.8-27b", system="s", user="u",
                              temperature=0)
    assert "reasoning_effort" not in seen["body"]


def test_live_client_backs_off_on_429(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("scholarplan.llm.openai_compat.time.sleep", lambda s: slept.append(s))
    responses = iter([httpx.Response(429, headers={"retry-after": "2"}), _ok("{}")])
    c = _client(lambda r: next(responses))
    assert c.complete(task="t", model="m", system="s", user="u", temperature=0) == "{}"
    assert 2.5 in slept  # honoured the server's retry-after hint (+0.5 s margin)


def test_live_client_substitutes_withdrawn_model():
    """Regression test for remediation R8: the planned Critic model was withdrawn."""
    models: list[str] = []

    def handler(request):
        body = json.loads(request.content)
        models.append(body["model"])
        if body["model"] == "gone-model":
            return httpx.Response(404, json={"error": {"code": "model_not_found"}})
        return _ok("{}")

    c = _client(handler, fallbacks={"gone-model": "backup-model"})
    c.complete(task="t", model="gone-model", system="s", user="u", temperature=0)
    c.complete(task="t", model="gone-model", system="s", user="u", temperature=0)
    assert models == ["gone-model", "backup-model", "backup-model"]  # remembered after first 404
    assert c.substitutions == {"gone-model": "backup-model"}


def test_live_client_fails_fast_on_auth_error():
    c = _client(lambda r: httpx.Response(401, text="invalid key"), key="BAD")
    with pytest.raises(LLMError, match="401"):
        c.complete(task="t", model="m", system="s", user="u", temperature=0)


def test_live_client_requires_key():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        OpenAICompatibleClient("https://llm.test/v1", "")
