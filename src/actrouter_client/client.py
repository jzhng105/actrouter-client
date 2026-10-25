"""Talk to the ActRouter API.

Calls return the API's JSON shapes as plain dicts. :func:`connect` reads the
environment:

* ``ACTROUTER_API_KEY`` -- your API key, sent as ``x-api-key``. The hosted
  API rejects requests without one (HTTP 401).
* ``ACTROUTER_URL`` -- the API's base URL; unset means :data:`DEFAULT_URL`,
  the hosted ActRouter.

The client uses only the standard library.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Optional, Protocol

#: the hosted ActRouter API
DEFAULT_URL = "https://r93hdy7zti.execute-api.us-east-1.amazonaws.com"
DEFAULT_TIMEOUT = 10.0


class ActRouterError(RuntimeError):
    """ActRouter is unreachable, rejected the request, or answered nonsense.
    `status` is the HTTP status when there was one."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


_SLUG_WORD = re.compile(r"[a-z0-9]+")


def task_as_function(task: dict[str, Any]) -> dict[str, Any]:
    """A task as function metadata, the way the server's
    ``TaskInfo.to_function_info`` does it; for deployments that predate
    ``POST /v1/route/task``. The task kind travels as a ``task:`` tag, which
    an older inventory ignores instead of rejecting."""
    words = _SLUG_WORD.findall(str(task.get("title") or task["description"]).lower())
    fn: dict[str, Any] = {
        "name": "_".join(words[:8]) or "agent_task",
        "docstring": task["description"],
        "tags": list(task.get("tags") or []),
    }
    if task.get("task_kind") and task["task_kind"] != "unknown":
        fn["tags"].append(f"task:{task['task_kind']}")
    for key in (
        "estimated_prompt_tokens", "estimated_completion_tokens",
        "latency_budget_ms", "cost_budget_usd", "privacy_tier",
    ):
        if task.get(key) is not None:
            fn[key] = task[key]
    return fn


class Router(Protocol):
    """What every backend offers. Each method returns the API's JSON."""

    @property
    def where(self) -> str: ...

    def route_function(
        self,
        function: dict[str, Any],
        *,
        router: str = "heuristic",
        candidate_models: Optional[list[str]] = None,
        top_k: int = 3,
        baseline_model: Optional[str] = None,
    ) -> dict[str, Any]: ...

    def route_task(
        self,
        task: dict[str, Any],
        *,
        router: str = "heuristic",
        preference: str = "balanced",
        candidate_models: Optional[list[str]] = None,
        top_k: int = 3,
        baseline_model: Optional[str] = None,
    ) -> dict[str, Any]: ...

    def list_models(self) -> list[dict[str, Any]]: ...

    def list_task_types(self) -> dict[str, Any]: ...


def _body(**fields: Any) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


class HttpRouter:
    """ActRouter's HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"content-type": "application/json", "accept": "application/json"}
        if headers:
            self.headers.update(headers)

    @property
    def where(self) -> str:
        return self.base_url

    def route_function(
        self, function, *, router="heuristic", candidate_models=None, top_k=3, baseline_model=None
    ):
        body = _body(
            function=function, router=router, candidate_models=candidate_models,
            top_k=top_k, baseline_model=baseline_model,
        )
        return self._post_route("/v1/route", body)

    def route_task(
        self, task, *, router="heuristic", preference="balanced", candidate_models=None, top_k=3,
        baseline_model=None,
    ):
        body = _body(
            task=task, router=router, preference=preference,
            candidate_models=candidate_models, top_k=top_k, baseline_model=baseline_model,
        )
        try:
            return self._post_route("/v1/route/task", body)
        except ActRouterError as exc:
            if exc.status != 404:
                raise
        # An older deployment: route the task as a function. `needs` become
        # a candidate filter here; `preference` has no equivalent there.
        needs = {"tools": "supports_tools", "vision": "supports_vision"}
        flags = [needs.get(n, n) for n in task.get("needs") or []]
        if flags:
            models = [
                m["id"] for m in self.list_models()
                if all(m.get(f) for f in flags)
                and (candidate_models is None or m["id"] in candidate_models)
            ]
            if not models:
                raise ActRouterError(f"no candidate model has {', '.join(flags)}", 400)
            candidate_models = models
        resp = self.route_function(
            task_as_function(task), router=router,
            candidate_models=candidate_models, top_k=top_k, baseline_model=baseline_model,
        )
        resp["compat"] = "routed via /v1/route: this deployment predates /v1/route/task"
        return resp

    def _post_route(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST a routing request. A deployment that predates `baseline_model`
        rejects it as an unknown field (422); route without it rather than
        fail, and say so -- the baseline is then compared client-side only."""
        try:
            return self._request("POST", path, body)
        except ActRouterError as exc:
            if exc.status != 422 or "baseline_model" not in str(exc) or "baseline_model" not in body:
                raise
        resp = self._request("POST", path, {k: v for k, v in body.items() if k != "baseline_model"})
        resp["compat_baseline"] = "this deployment predates baseline_model; no server-side estimate"
        return resp

    def list_models(self):
        data = self._request("GET", "/v1/models").get("data")
        return list(data) if isinstance(data, list) else []

    def list_task_types(self):
        return self._request("GET", "/v1/task-types")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def billing_link(self, kind: str) -> str:
        """A Stripe link for this key's customer: ``upgrade`` (Checkout for
        Pro) or ``portal`` (card, invoices, cancelling)."""
        if kind not in ("upgrade", "portal"):
            raise ValueError(f"unknown billing link {kind!r}")
        url = self._request("POST", f"/billing/{kind}", {}).get("url")
        if not isinstance(url, str):
            raise ActRouterError(f"ActRouter returned no {kind} link")
        return url

    def _request(
        self, method: str, path: str, payload: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive
                pass
            if exc.code == 402:  # the plan's limit; the server says what to do
                try:
                    message = json.loads(detail)["detail"]
                except (ValueError, KeyError, TypeError):
                    message = detail or "ActRouter plan limit reached: run `actrouter-route upgrade`"
                raise ActRouterError(str(message), 402) from exc
            if exc.code == 401:
                hint = (
                    "ACTROUTER_API_KEY was rejected" if "x-api-key" in self.headers
                    else "set ACTROUTER_API_KEY (the Claude Code plugin's api_key setting)"
                )
                raise ActRouterError(
                    f"ActRouter needs a valid API key: {hint}. Server said: {detail}", 401
                ) from exc
            raise ActRouterError(
                f"ActRouter returned HTTP {exc.code} for {path}: {detail}", exc.code
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ActRouterError(f"could not reach ActRouter at {url}: {reason}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActRouterError(f"ActRouter returned non-JSON from {path}: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise ActRouterError(f"ActRouter returned an unexpected payload from {path}")
        return parsed


def env(name: str) -> Optional[str]:
    """An environment setting, or None when it is unset, empty, or a
    placeholder its host left unexpanded (``${user_config.router_url}``
    when a Claude Code plugin is installed into another agent)."""
    value = os.environ.get(name, "").strip()
    return value if value and "${" not in value else None


def connect(url: Optional[str] = None, *, timeout: Optional[float] = None) -> Router:
    """The router `url` names, else the one the environment names (see the
    module docstring)."""
    url = url if url is not None else (env("ACTROUTER_URL") or "")
    if url == "local":
        raise ActRouterError(
            "ACTROUTER_URL=local is no longer supported: routing runs on the hosted "
            "ActRouter API. Unset ACTROUTER_URL and set ACTROUTER_API_KEY."
        )
    headers = {}
    if key := env("ACTROUTER_API_KEY"):
        headers["x-api-key"] = key
    if timeout is None:
        timeout = float(env("ACTROUTER_TIMEOUT") or DEFAULT_TIMEOUT)
    return HttpRouter(url or DEFAULT_URL, timeout=timeout, headers=headers)
