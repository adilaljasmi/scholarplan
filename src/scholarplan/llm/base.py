"""LLM client interface and the validate-or-retry wrapper.

Why an interface: the agents depend on ``LLMClient`` (an abstraction), not on
Groq. That lets the same agent code run against the live service, against a
deterministic mock for testing, or against another OpenAI-compatible endpoint
(Azure, Ollama) by configuration alone: the dependency-inversion principle
applied to model providers, which also limits the vendor lock-in risk noted
in Unit 6.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a model cannot produce valid output within the retry budget."""


@dataclass
class Usage:
    """Token accounting, per model. Unit 6 promised to measure real per-run
    cost rather than extrapolate from benchmarks; this is where it is measured."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, model: str, prompt: int, completion: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        m = self.by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        m["calls"] += 1
        m["prompt_tokens"] += prompt
        m["completion_tokens"] += completion


class LLMClient(Protocol):
    usage: Usage

    def complete(self, *, task: str, model: str, system: str, user: str,
                 temperature: float, json_mode: bool = True) -> str:
        """Return the raw text of one chat completion.

        ``task`` names the call site (plan, react, summarise, verify_claims,
        synthesise, verify_synthesis). Live clients only log it; the mock uses
        it to decide what to answer.
        """
        ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences or chatter.

    Even in JSON mode some models wrap output in ```json fences or add a
    sentence before it; stripping that is cheaper than a retry.
    """
    t = _FENCE.sub("", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            return json.loads(t[start:end + 1])
        raise


def call_json(client: LLMClient, schema: type[T], *, task: str, model: str, system: str,
              payload: dict[str, Any], temperature: float, max_retries: int = 3) -> T:
    """Ask for JSON matching ``schema``; on failure, show the model its error and retry.

    Feeding the validation error back is a small-scale version of the
    critique-and-retry idea (Shinn et al., 2023): the model is told *why* its
    output was rejected, which usually fixes it in one round. After
    ``max_retries`` we raise rather than guess, so the caller can fall back
    to a deterministic policy (LLM-Modulo: the LLM proposes, an external
    component decides; Kambhampati et al., 2024).
    """
    user = "INPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=1)
    last_error = ""
    for attempt in range(max_retries):
        prompt = user if not last_error else (
            user + f"\n\nYour previous reply was invalid: {last_error}\n"
                   "Reply again with ONLY a JSON object that matches the schema.")
        raw = client.complete(task=task, model=model, system=system, user=prompt,
                              temperature=temperature, json_mode=True)
        try:
            return schema.model_validate(extract_json(raw))
        except ValidationError as exc:
            # Name the offending field(s); "1 validation error" alone gives the
            # model nothing to correct (remediation R15).
            last_error = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
    raise LLMError(f"{task}: no valid {schema.__name__} after {max_retries} attempts ({last_error})")
