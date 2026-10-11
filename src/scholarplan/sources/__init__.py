"""Scholarly data sources (arXiv, OpenAlex, Crossref)."""

from __future__ import annotations

from scholarplan.config import Settings
from scholarplan.sources.clients import ArxivClient, CrossrefClient, OpenAlexClient
from scholarplan.sources.http import RateLimiter, ScholarlyHTTP


def build_clients(settings: Settings, transport=None) -> dict[str, object]:
    """Wire the three clients to one shared, rate-limited HTTP helper."""
    http = ScholarlyHTTP(
        cache_dir=settings.cache_dir,
        contact_email=settings.contact_email,
        timeout_s=settings.http_timeout_s,
        limiters={
            "arxiv": RateLimiter(settings.arxiv_min_interval_s),
            "openalex": RateLimiter(settings.api_min_interval_s),
            "crossref": RateLimiter(settings.api_min_interval_s),
        },
        transport=transport,
    )
    return {"arxiv": ArxivClient(http), "openalex": OpenAlexClient(http), "crossref": CrossrefClient(http)}


__all__ = ["build_clients", "ArxivClient", "OpenAlexClient", "CrossrefClient", "ScholarlyHTTP", "RateLimiter"]
