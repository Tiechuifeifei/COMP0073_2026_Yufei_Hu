"""Minimal Massive REST client: auth, retries, no bulk download helpers beyond pagination."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from massive_audit.config import API_BASE, api_key


@dataclass
class ProbeResult:
    endpoint: str
    url: str
    status_code: int | None
    accessible: bool
    plan_required: str
    error: str
    n_results: int
    sample_keys: str
    extra: dict[str, Any] = field(default_factory=dict)
    body_excerpt: str = ""


class MassiveClient:
    def __init__(self, min_interval_s: float = 0.25) -> None:
        self.key = api_key()
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self.calls = 0
        self.rate_limit_hits = 0

    def _sleep(self) -> None:
        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)

    def _adapt_rate(self, headers: dict[str, str]) -> None:
        lim = headers.get("x-ratelimit-limit") or headers.get("x-ratelimit-limit-minute")
        if not lim:
            return
        try:
            n = int(float(str(lim).split(",")[0]))
        except ValueError:
            return
        if n > 0:
            self.min_interval_s = max(self.min_interval_s, 60.0 / n + 0.05)

    def get(self, path: str, params: dict[str, Any] | None = None, timeout: int = 60) -> tuple[int, dict[str, Any] | list[Any] | None, dict[str, str]]:
        self._sleep()
        params = dict(params or {})
        params.setdefault("apiKey", self.key)
        if path.startswith("http"):
            url = path
            if "apiKey=" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}apiKey={self.key}"
        else:
            url = f"{API_BASE}{path}"
            if params:
                url = f"{url}?{urlencode(params, doseq=True)}"
        req = Request(url, headers={"Accept": "application/json", "Authorization": f"Bearer {self.key}"})
        headers: dict[str, str] = {}
        for attempt in range(6):
            try:
                with urlopen(req, timeout=timeout) as resp:
                    self._last = time.time()
                    self.calls += 1
                    raw = resp.read()
                    headers = {k.lower(): v for k, v in resp.headers.items()}
                    self._adapt_rate(headers)
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                    return int(resp.status), payload, headers
            except HTTPError as exc:
                self._last = time.time()
                self.calls += 1
                body = exc.read().decode("utf-8", errors="replace")
                headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
                self._adapt_rate(headers)
                if exc.code == 429:
                    self.rate_limit_hits += 1
                    retry_after = float(headers.get("retry-after", 2 ** attempt))
                    time.sleep(min(max(retry_after, 1.0), 60.0))
                    continue
                try:
                    payload = json.loads(body) if body else {"error": body}
                except json.JSONDecodeError:
                    payload = {"error": body[:2000]}
                return int(exc.code), payload, headers
            except URLError as exc:
                self._last = time.time()
                time.sleep(1.5 * (attempt + 1))
                if attempt == 5:
                    return None, {"error": str(exc.reason)}, {}  # type: ignore[return-value]
        return 429, {"error": "rate_limited"}, headers

    def probe(self, name: str, path: str, params: dict[str, Any] | None = None) -> ProbeResult:
        status, payload, headers = self.get(path, params)
        accessible = status is not None and 200 <= status < 300
        plan_required = ""
        error = ""
        n_results = 0
        sample_keys = ""
        excerpt = ""
        extra: dict[str, Any] = {
            "x-ratelimit-remaining": headers.get("x-ratelimit-remaining", ""),
            "x-ratelimit-limit": headers.get("x-ratelimit-limit", ""),
            "x-ratelimit-reset": headers.get("x-ratelimit-reset", ""),
        }
        if isinstance(payload, dict):
            error = str(payload.get("error") or payload.get("message") or "")
            status_text = str(payload.get("status") or "")
            if not accessible:
                plan_required = status_text or error or f"HTTP {status}"
            results = payload.get("results")
            if isinstance(results, list):
                n_results = len(results)
                if results and isinstance(results[0], dict):
                    sample_keys = ",".join(sorted(results[0].keys())[:20])
            elif isinstance(results, dict):
                n_results = 1
                sample_keys = ",".join(sorted(results.keys())[:20])
            extra["api_status"] = status_text
            extra["resultsCount"] = payload.get("resultsCount")
            extra["next_url"] = bool(payload.get("next_url"))
            extra["adjusted"] = payload.get("adjusted")
            excerpt = json.dumps(payload, default=str)[:800]
        elif payload is None:
            error = "no_response"
            plan_required = "NO_RESPONSE"
        url = f"{API_BASE}{path}"
        return ProbeResult(
            endpoint=name,
            url=url,
            status_code=status,
            accessible=accessible,
            plan_required=plan_required or ("included" if accessible else f"HTTP {status}"),
            error=error,
            n_results=n_results,
            sample_keys=sample_keys,
            extra=extra,
            body_excerpt=excerpt,
        )

    def paginate(self, path: str, params: dict[str, Any] | None = None, max_pages: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        status, payload, _ = self.get(path, params)
        pages = 0
        while isinstance(payload, dict) and pages < max_pages:
            pages += 1
            results = payload.get("results") or []
            if isinstance(results, list):
                rows.extend([r for r in results if isinstance(r, dict)])
            next_url = payload.get("next_url")
            if not next_url or status != 200:
                break
            status, payload, _ = self.get(str(next_url), params=None)
        return rows
