"""Live client for any OpenAI-compatible chat endpoint (Groq by default).

Implemented with plain ``httpx`` instead of a vendor SDK, in keeping with the
Unit 6 decision to avoid heavy abstractions: the request is visible in a dozen
lines, which matters when the behaviour under assessment is the agent loop
itself. Groq, Azure OpenAI and Ollama all expose this same API shape, so
switching provider is a base-URL change.
"""

from __future__ import annotations

import threading
import time

import httpx

from scholarplan.llm.base import LLMError, Usage


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 90.0,
                 min_interval_s: float = 2.1, max_retries: int = 4,
                 transport: httpx.BaseTransport | None = None,
                 fallbacks: dict[str, str] | None = None):
        if not api_key:
            raise LLMError("No API key configured. Put GROQ_API_KEY in .env or use --llm mock.")
        self.client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport,
                                   headers={"Authorization": f"Bearer {api_key}"})
        self.usage = Usage()
        # Groq's free tier allows ~30 requests/minute per model. Spacing calls
        # client-side (2.1 s) avoids burning retries on predictable 429s.
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()
        # Model substitutions used when a provider withdraws a model (404).
        # Free tiers change without notice; degrading to a known-good model
        # keeps a demo alive, and the switch is recorded in ``substitutions``.
        self.fallbacks = dict(fallbacks or {})
        self.substitutions: dict[str, str] = {}

    def _throttle(self, model: str) -> None:
        with self._lock:
            wait = self._last_call.get(model, 0) + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call[model] = time.monotonic()

    def complete(self, *, task: str, model: str, system: str, user: str,
                 temperature: float, json_mode: bool = True) -> str:
        model = self.substitutions.get(model, model)
        body: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": 2048,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if model.startswith("openai/gpt-oss"):
            # gpt-oss models reason before answering; "low" effort keeps the
            # hidden reasoning short. Free-tier limits are per *token*, and the
            # structured tasks here do not need long deliberation.
            body["reasoning_effort"] = "low"

        last = ""
        for attempt in range(self.max_retries):
            self._throttle(model)
            try:
                resp = self.client.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                last = f"transport error: {exc}"
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 429:
                # Respect the server's own back-off hint when present.
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else 5.0
                last = "rate limited (429)"
                time.sleep(min(delay, 60.0) + 0.5)
                continue
            if resp.status_code >= 500:
                last = f"server error {resp.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 400 and "response_format" in body and "json" in resp.text.lower():
                # Groq returns 400 "json_validate_failed" when the model's
                # output was not valid JSON. Retry; on the final attempt drop
                # JSON mode altogether and let extract_json() recover the
                # object from free text (remediation R15: in a live run all
                # four JSON-mode attempts failed for gpt-oss-20b).
                last = "provider JSON validation failed"
                if attempt == self.max_retries - 2:
                    body.pop("response_format")
                continue
            if resp.status_code == 404 and "model_not_found" in resp.text and model in self.fallbacks:
                substitute = self.fallbacks[model]
                self.substitutions[model] = substitute
                return self.complete(task=task, model=substitute, system=system, user=user,
                                     temperature=temperature, json_mode=json_mode)
            if resp.status_code >= 400:
                raise LLMError(f"{task}: HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            u = data.get("usage") or {}
            self.usage.add(model, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
            return data["choices"][0]["message"].get("content") or ""
        raise LLMError(f"{task}: giving up after {self.max_retries} attempts ({last})")
