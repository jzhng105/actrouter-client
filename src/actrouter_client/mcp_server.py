"""ActRouter as an MCP server, for any MCP-capable agent (Claude Code, Codex,
Cursor, Gemini CLI, ...).

Run it over stdio::

    actrouter-mcp                       # or: python -m actrouter_client.mcp_server

Configuration is all environment (see :mod:`actrouter_client.client` and
:mod:`actrouter_client.hosts`): ``ACTROUTER_API_KEY``, ``ACTROUTER_URL``, ``ACTROUTER_HOST``,
``ACTROUTER_FAMILIES``, ``ACTROUTER_CANDIDATES``, ``ACTROUTER_PROJECT_DIR``.

No tool changes the project. ``route_function`` reads the named Python file
locally and sends only its static metadata; nothing of the function body
leaves the machine. Routing decisions are noted in the local routing ledger
(model ids and costs only, see :mod:`actrouter_client.ledger`), which
``routing_report`` summarizes.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore[no-redef]
from mcp.types import ToolAnnotations

from actrouter_client import baseline as baselines
from actrouter_client import ledger
from actrouter_client.advisor import Advisor
from actrouter_client.client import ActRouterError, env

INSTRUCTIONS = """\
ActRouter recommends which LLM fits a piece of work, from the work's \
description or a function's static metadata. Use it:
- before delegating to a subagent or starting a sizeable task: call \
route_task with a one or two sentence description of the *kind* of work \
(never paste code, secrets or file contents), then use `host_model`;
- when writing or changing code that calls an LLM: call route_function on \
that function and use `model` in the code;
- prefer preference="quality" for hard debugging, design or security work \
and "cost" for bulk or mechanical work.
Pass `current_model` (the model you would otherwise use, e.g. your own) so \
the answer shows what routing saves; routing_report sums it up.
The answer is advice: follow an explicit model choice by the user over it."""

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)

Preference = Literal["quality", "balanced", "cost"]
Need = Literal["tools", "vision"]
Privacy = Literal["public", "internal", "confidential"]


@contextmanager
def _tool_errors() -> Iterator[None]:
    """Report anticipated failures to the agent in words it can act on
    (the SDK hides the text of any other exception)."""
    try:
        yield
    except (ActRouterError, ValueError, LookupError, OSError, SyntaxError) as exc:
        raise ToolError(str(exc)) from exc


def _project_dir() -> Optional[str]:
    # Claude Code sets CLAUDE_PROJECT_DIR for hooks but not for MCP servers,
    # and can pass the plugin's ${CLAUDE_PROJECT_DIR} through unexpanded; it
    # starts stdio servers in the project, so the working directory stands in.
    return env("ACTROUTER_PROJECT_DIR") or env("CLAUDE_PROJECT_DIR") or os.getcwd()


def _project_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(_project_dir() or os.getcwd()) / p
    if p.suffix != ".py":
        raise ValueError(f"route_function reads Python source only; got {p.name}")
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    return p


def _baseline(current_model: Optional[str]) -> tuple[Optional[str], str]:
    if current_model:
        return current_model, "current_model (agent)"
    return baselines.override() or baselines.UNKNOWN


def _recorded(advice: dict[str, Any], baseline: tuple[Optional[str], str], tool: str) -> dict[str, Any]:
    ledger.route_event(advice, surface="mcp", baseline=baseline[0], baseline_source=baseline[1], tool=tool)
    if tool == "route_task":
        # the subagent the agent starts on this answer claims it (PreToolUse),
        # so its real usage is measured
        ledger.offer(advice, project=_project_dir(), baseline=baseline[0], baseline_source=baseline[1])
    return advice


def build_server(advisor: Optional[Advisor] = None) -> Any:
    server = _Server("actrouter", instructions=INSTRUCTIONS)
    state: dict[str, Advisor] = {}

    def adv() -> Advisor:
        # Built lazily: the server must start (and list its tools) even when
        # the router is unreachable.
        if "a" not in state:
            state["a"] = advisor or Advisor()
        return state["a"]

    @server.tool(annotations=READ_ONLY)
    def route_task(
        description: str,
        title: Optional[str] = None,
        task_kind: Optional[str] = None,
        needs: Optional[list[Need]] = None,
        preference: Preference = "balanced",
        cost_budget_usd: Optional[float] = None,
        privacy_tier: Optional[Privacy] = None,
        top_k: int = 3,
        current_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Recommend a model for a unit of agent work, e.g. a subagent task.

        description: one or two sentences on the kind of work ("Find where
          the config is loaded", "Fix the failing auth test"). Describe the
          work; do not include code, logs, secrets or file contents.
        title: optional 2-6 word label, weighted most heavily.
        task_kind: optional task type id or alias (see list_task_types).
        needs: hard requirements, "tools" and/or "vision".
        preference: quality | balanced | cost.
        current_model: the model this work would otherwise run on (yours, if
          you would do it yourself); the answer then includes
          `estimated_savings_usd`.
        Returns `host_model` (the value this agent host takes), the catalog
        `model`, the classified task and the alternatives.
        """
        baseline = _baseline(current_model)
        with _tool_errors():
            return _recorded(adv().for_task(
                description,
                title=title,
                task_kind=task_kind,
                needs=list(needs) if needs else None,
                preference=preference,
                cost_budget_usd=cost_budget_usd,
                privacy_tier=privacy_tier,
                top_k=top_k,
                baseline=baseline[0],
            ), baseline, "route_task")

    @server.tool(annotations=READ_ONLY)
    def route_function(
        path: str,
        function: str,
        task_kind: Optional[str] = None,
        estimated_prompt_tokens: Optional[int] = None,
        estimated_completion_tokens: Optional[int] = None,
        cost_budget_usd: Optional[float] = None,
        latency_budget_ms: Optional[int] = None,
        privacy_tier: Optional[Privacy] = None,
        top_k: int = 3,
        current_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Recommend the model a Python function that calls an LLM should use.

        path: the .py file (relative to the project root or absolute).
        function: its name, or Class.method.
        The file is parsed locally; only the function's name, signature,
        decorators and docstring are sent, never its body. Use the returned
        `model` in the code (e.g. actdecor's `model=` or your LLM client).
        current_model: the model the code uses now, if any.
        """
        baseline = _baseline(current_model)
        with _tool_errors():
            return _recorded(adv().for_source(
                str(_project_path(path)),
                function,
                root=_project_dir(),
                task_kind=task_kind,
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_completion_tokens=estimated_completion_tokens,
                cost_budget_usd=cost_budget_usd,
                latency_budget_ms=latency_budget_ms,
                privacy_tier=privacy_tier,
                top_k=top_k,
                baseline=baseline[0],
            ), baseline, "route_function")

    @server.tool(annotations=READ_ONLY)
    def route_function_metadata(
        name: str,
        signature: Optional[str] = None,
        docstring: Optional[str] = None,
        task_kind: Optional[str] = None,
        tags: Optional[list[str]] = None,
        estimated_prompt_tokens: Optional[int] = None,
        estimated_completion_tokens: Optional[int] = None,
        cost_budget_usd: Optional[float] = None,
        latency_budget_ms: Optional[int] = None,
        top_k: int = 3,
        current_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Recommend a model for an LLM-calling function in any language,
        described by its static metadata: name, signature, doc comment.
        Never pass argument values or outputs. current_model: the model the
        code uses now, if any."""
        meta = {
            "name": name,
            "signature": signature,
            "docstring": docstring,
            "task_kind": task_kind,
            "tags": tags,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "estimated_completion_tokens": estimated_completion_tokens,
            "cost_budget_usd": cost_budget_usd,
            "latency_budget_ms": latency_budget_ms,
        }
        baseline = _baseline(current_model)
        with _tool_errors():
            return _recorded(
                adv().for_function(meta, top_k=top_k, baseline=baseline[0]),
                baseline, "route_function_metadata",
            )

    @server.tool(annotations=READ_ONLY)
    def list_models(family: Optional[str] = None, host_only: bool = True) -> dict[str, Any]:
        """The candidate models with their size tier, strengths and prices.
        host_only=true (default) lists only the models this agent host can
        run; family filters by vendor (e.g. "anthropic", "openai")."""
        with _tool_errors():
            a = adv()
            allowed = set(a.candidates() or []) if host_only else set()
            models = [
                {**m, "host_model": a.host.spell(m["id"])}
                for m in a.catalog()
                if (not allowed or m["id"] in allowed) and (family is None or m.get("family") == family)
            ]
            return {"host": a.host.name, "models": models}

    @server.tool(annotations=READ_ONLY)
    def routing_report(days: int = 30) -> dict[str, Any]:
        """What model routing has saved on this machine: routed calls, the
        baseline models they replaced, measured and estimated savings, by
        task type and model switch. days=0 covers the whole ledger. For a
        dashboard, run `actrouter-route report --html savings.html`."""
        from actrouter_client.report import load

        with _tool_errors():
            return load(days or None, None, adv().catalog())

    @server.tool(annotations=READ_ONLY)
    def list_task_types() -> dict[str, Any]:
        """The task types ActRouter classifies work into (ids, aliases,
        descriptions); any id or alias is a valid `task_kind`."""
        with _tool_errors():
            return adv().router.list_task_types()

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
