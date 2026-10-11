"""Polite, cached HTTP access to the scholarly APIs.

Three concerns are handled here once, instead of in every API client:

1. **Rate limits / terms of use.** arXiv asks for at most one request every
   three seconds; Crossref and OpenAlex ask clients to identify themselves.
   Respecting those terms is a legal and professional obligation, not only a
   technical one, so the limiter is enforced centrally and cannot be bypassed
   by an individual client.
2. **Caching.** The Unit 6 design listed source caching as a mitigation for
   rate limits and cost. Responses are cached on disk by URL; scholarly
   metadata is effectively immutable, so no expiry is needed for a single
   project. A cache also makes repeated demo runs reproducible.
3. **Testability.** An ``httpx`` transport can be injected, so the unit tests
   replay recorded fixtures instead of touching the network (deterministic,
   free and fast).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from scholarplan import __version__


class RateLimiter:
    """Minimum spacing between requests to one host (thread-safe)."""

    def __init__(self, min_interval_s: float, clock=time.monotonic, sleep=time.sleep):
        self.min_interval_s = min_interval_s
        self._last = float("-inf")
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def wait(self) -> float:
        """Block until the next request is allowed; return seconds slept."""
        with self._lock:
            now = self._clock()
            delay = self._last + self.min_interval_s - now
            if delay > 0:
                self._sleep(delay)
            self._last = max(now, self._last + self.min_interval_s)
            return max(delay, 0.0)


class ScholarlyHTTP:
    """Shared GET helper used by the arXiv, OpenAlex and Crossref clients."""

    def __init__(self, cache_dir: Path | None, contact_email: str = "", timeout_s: float = 20.0,
                 limiters: dict[str, RateLimiter] | None = None,
                 transport: httpx.BaseTransport | None = None):
        ua = f"ScholarPlan/{__version__} (academic coursework"
        ua += f"; mailto:{contact_email})" if contact_email else ")"
        self.contact_email = contact_email
        self.client = httpx.Client(timeout=timeout_s, headers={"User-Agent": ua},
                                   follow_redirects=True, transport=transport)
        self.cache_dir = Path(cache_dir) / "http" if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiters = limiters or {}
        self.cache_hits = 0
        self.network_calls = 0

    def _cache_path(self, url: str, params: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:32]
        return self.cache_dir / f"{key}.txt"

    def get_text(self, host_key: str, url: str, params: dict[str, Any]) -> str:
        path = self._cache_path(url, params)
        if path and path.exists():
            self.cache_hits += 1
            return path.read_text(encoding="utf-8")
        limiter = self.limiters.get(host_key)
        last_exc: Exception | None = None
        # Two retries with back-off: transient 429/5xx responses are common on
        # free public APIs and should not abort a whole research run.
        for attempt in range(3):
            if limiter:
                limiter.wait()
            try:
                self.network_calls += 1
                resp = self.client.get(url, params=params)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request,
                                                response=resp)
                resp.raise_for_status()
                text = resp.text
                if path:
                    path.write_text(text, encoding="utf-8")
                return text
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if status is not None and status < 500 and status != 429:
                    break  # a 4xx other than 429 will not fix itself on retry
                # Honour Retry-After when the server sends one (remediation R10:
                # the first live run hit OpenAlex 429s because a fixed 1.5 s
                # back-off was shorter than the server's own window).
                hint = response.headers.get("retry-after", "") if response is not None else ""
                delay = float(hint) if hint.replace(".", "", 1).isdigit() else 2.0 * 2 ** attempt
                time.sleep(min(delay, 30.0))
        raise RuntimeError(f"{host_key} request failed: {last_exc}") from last_exc

    def get_json(self, host_key: str, url: str, params: dict[str, Any]) -> Any:
        return json.loads(self.get_text(host_key, url, params))

    def close(self) -> None:
        self.client.close()
