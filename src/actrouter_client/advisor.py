"""Turn an ActRouter response into advice an agent can act on.

:class:`Advisor` is what the MCP server, the Claude Code hook and the CLI
share: it restricts routing to models the host can run, then translates the
answer into the host's spelling, with the reasons in a compact form an LLM
can read and relay.
"""

from __future__ import annotations

from typing import Any, Optional

from actrouter_client.baseline import to_catalog_id
from actrouter_client.client import ActRouterError, Router, connect
from actrouter_client.hosts import Host, candidate_models, get_host
from actrouter_client.metadata import function_metadata_from_source

#: reasons worth showing; the rest are score components
_KEEP_REASONS = ("task=", "needs=", "no candidate", "route-invariant")


class Advisor:
    def __init__(self, router: Optional[Router] = None, host: Optional[Host | str] = None) -> None:
        self.router = router or connect()
        self.host = host if isinstance(host, Host) else get_host(host)
        self._catalog: Optional[list[dict[str, Any]]] = None

    # --- catalog ---------------------------------------------------------------

    def catalog(self) -> list[dict[str, Any]]:
        if self._catalog is None:
            self._catalog = self.router.list_models()
        return self._catalog

    def baseline_id(self, model: Optional[str]) -> Optional[str]:
        """The catalog id for the host's spelling of a model (``sonnet``,
        ``claude-sonnet-4-6-20260205``), or None if the catalog lacks it."""
        if not model:
            return None
        try:
            return to_catalog_id(model, self.catalog())
        except ActRouterError:
            return None

    def candidates(self, override: Optional[list[str]] = None) -> Optional[list[str]]:
        if override:
            return list(override)
        return candidate_models(self.host, self.catalog())

    # --- routing ---------------------------------------------------------------

    def for_task(
        self,
        description: str,
        *,
        title: Optional[str] = None,
        task_kind: Optional[str] = None,
        needs: Optional[list[str]] = None,
        preference: str = "balanced",
        cost_budget_usd: Optional[float] = None,
        latency_budget_ms: Optional[int] = None,
        privacy_tier: Optional[str] = None,
        candidate_models: Optional[list[str]] = None,
        top_k: int = 3,
        baseline: Optional[str] = None,
    ) -> dict[str, Any]:
        """`baseline` is the model that would run without routing, as the
        host spells it; the answer then compares the two (see
        :mod:`actrouter_client.baseline`)."""
        task = {
            "description": description,
            "title": title,
            "task_kind": task_kind,
            "needs": needs,
            "cost_budget_usd": cost_budget_usd,
            "latency_budget_ms": latency_budget_ms,
            "privacy_tier": privacy_tier,
        }
        baseline_id = self.baseline_id(baseline)
        resp = self.router.route_task(
            {k: v for k, v in task.items() if v not in (None, [])},
            preference=preference,
            candidate_models=self.candidates(candidate_models),
            top_k=top_k,
            baseline_model=baseline_id,
        )
        return self.advice(resp, baseline=baseline, baseline_id=baseline_id)

    def for_function(
        self,
        function: dict[str, Any],
        *,
        candidate_models: Optional[list[str]] = None,
        top_k: int = 3,
        baseline: Optional[str] = None,
    ) -> dict[str, Any]:
        baseline_id = self.baseline_id(baseline)
        resp = self.router.route_function(
            {k: v for k, v in function.items() if v not in (None, [])},
            candidate_models=self.candidates(candidate_models),
            top_k=top_k,
            baseline_model=baseline_id,
        )
        return self.advice(resp, baseline=baseline, baseline_id=baseline_id)

    def for_source(
        self,
        path: str,
        qualname: str,
        *,
        root: Optional[str] = None,
        candidate_models: Optional[list[str]] = None,
        top_k: int = 3,
        baseline: Optional[str] = None,
        **hints: Any,
    ) -> dict[str, Any]:
        meta = function_metadata_from_source(path, qualname, root=root, **hints)
        advice = self.for_function(
            meta, candidate_models=candidate_models, top_k=top_k, baseline=baseline
        )
        advice["function"] = {k: meta[k] for k in ("name", "qualname", "module", "signature")}
        return advice

    # --- presentation ----------------------------------------------------------

    def advice(
        self,
        resp: dict[str, Any],
        *,
        baseline: Optional[str] = None,
        baseline_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """The response, reshaped for an agent: the model in the host's
        spelling first, then why, then the runners-up -- and, when a
        baseline was given, what routing changed against it."""
        chosen = resp.get("chosen_model")
        if not chosen:
            raise ActRouterError(f"ActRouter response has no chosen_model: {resp!r}")
        c = resp.get("classification") or {}
        ranking = resp.get("ranking") or []
        top_reasons = ranking[0].get("reasons", []) if ranking else []
        out: dict[str, Any] = {
            "model": chosen,
            "host": self.host.name,
            "host_model": self.host.spell(chosen),
            "use_it_as": self.host.usage,
            "task_type": c.get("task_type"),
            "task_confidence": c.get("confidence"),
            "task_source": c.get("source"),
            "reasons": [r for r in top_reasons if r.startswith(_KEEP_REASONS)] + c.get("evidence", [])[:3],
            "alternatives": [
                {
                    "model": r["model_id"],
                    "host_model": self.host.spell(r["model_id"]),
                    "score": r.get("score"),
                }
                for r in ranking[1:]
            ],
            "estimated_cost_usd": resp.get("estimated_cost_usd"),
            #: "given" or "prior" (the task type's typical agent work), or
            #: None from a deployment that predates it
            "estimate_basis": (resp.get("token_estimate") or {}).get("basis"),
            "router": resp.get("router"),
            "routed_by": self.router.where,
            "request_id": resp.get("request_id"),
        }
        if baseline:
            b = resp.get("baseline") or {}
            out["baseline_model"] = baseline_id
            out["baseline_host_model"] = baseline
            out["baseline_estimated_cost_usd"] = b.get("estimated_cost_usd")
            out["estimated_savings_usd"] = resp.get("estimated_savings_usd")
        if resp.get("compat"):
            out["note"] = resp["compat"]
        if out["host_model"] is None:
            out["note"] = (
                f"{chosen} cannot run on {self.host.name}; set ACTROUTER_FAMILIES or "
                f"ACTROUTER_CANDIDATES to route only over models it can run"
            )
        return out
