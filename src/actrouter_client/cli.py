"""``actrouter-route``: ActRouter from the shell.

    actrouter-route task "Fix the failing auth test" --host claude-code
    actrouter-route task "Search the repo for usages of parse_args" --host codex --model-only
    actrouter-route function src/news.py summarize_article
    actrouter-route models --host codex
    actrouter-route launch codex -- "add retries to the http client"
    actrouter-route launch claude --describe "write release notes" -- -p "..."
    actrouter-route report --days 30 --html savings.html

``launch`` routes on the agent's prompt (or ``--describe``), then replaces
itself with the agent CLI, passing the model (``codex -m``, ``claude
--model``). A model given explicitly is kept, and a routing failure starts
the agent unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Optional

from actrouter_client import baseline as baselines
from actrouter_client import ledger
from actrouter_client.advisor import Advisor
from actrouter_client.client import ActRouterError

#: agent CLI -> (host, flags that already choose a model, flag to pass ours)
LAUNCHERS = {
    "codex": ("codex", ("-m", "--model"), "-m"),
    "claude": ("claude-code", ("--model",), "--model"),
}

#: how much of a launch prompt is sent when no --describe is given
LAUNCH_PROMPT_CHARS = 600


def _print(advice: dict[str, Any], args: argparse.Namespace) -> None:
    if args.model_only:
        print(advice.get("host_model") or advice["model"])
    elif args.json:
        print(json.dumps(advice, indent=2))
    else:
        print(f"model:      {advice.get('host_model') or advice['model']}  ({advice['model']})")
        print(f"task:       {advice.get('task_type')} ({advice.get('task_confidence')}, {advice.get('task_source')})")
        alts = ", ".join(a.get("host_model") or a["model"] for a in advice.get("alternatives", []))
        if alts:
            print(f"also:       {alts}")
        if advice.get("note"):
            print(f"note:       {advice['note']}")
        print(f"routed by:  {advice.get('routed_by')}")


def _needs(value: Optional[str]) -> Optional[list[str]]:
    return [n.strip() for n in value.split(",") if n.strip()] if value else None


def _baseline_arg(args: argparse.Namespace) -> tuple[Optional[str], str]:
    if args.baseline:
        return args.baseline, "--baseline"
    return baselines.override() or baselines.UNKNOWN


def _cmd_task(args: argparse.Namespace) -> int:
    baseline, source = _baseline_arg(args)
    advice = Advisor(host=args.host).for_task(
        args.description,
        title=args.title,
        task_kind=args.task_kind,
        needs=_needs(args.needs),
        preference=args.preference,
        top_k=args.top_k,
        baseline=baseline,
    )
    ledger.route_event(advice, surface="cli", baseline=baseline, baseline_source=source)
    _print(advice, args)
    return 0


def _cmd_function(args: argparse.Namespace) -> int:
    baseline, source = _baseline_arg(args)
    advice = Advisor(host=args.host).for_source(
        args.path, args.function, root=args.root, task_kind=args.task_kind, top_k=args.top_k,
        baseline=baseline,
    )
    ledger.route_event(advice, surface="cli", baseline=baseline, baseline_source=source)
    _print(advice, args)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from actrouter_client.report import format_html, format_text, load

    catalog = Advisor(host="generic").catalog()
    summary = load(args.days, Path(args.ledger).expanduser() if args.ledger else None, catalog)
    if args.html:
        Path(args.html).write_text(format_html(summary), encoding="utf-8")
        print(f"wrote {args.html}", file=sys.stderr)
    print(json.dumps(summary, indent=2) if args.json else format_text(summary))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    a = Advisor(host=args.host)
    allowed = set(a.candidates() or [])
    for m in a.catalog():
        if not allowed or m["id"] in allowed:
            print(f"{m['id']:<40} {m.get('family', ''):<12} tier={m.get('size_tier')}  -> {a.host.spell(m['id'])}")
    return 0


def _prompt_of(agent_args: list[str]) -> str:
    positional = [a for a in agent_args if not a.startswith("-")]
    return positional[-1] if positional else ""


def _cmd_launch(args: argparse.Namespace) -> int:
    host, model_flags, flag = LAUNCHERS[args.agent]
    agent_args = list(args.agent_args)
    if agent_args[:1] == ["--"]:
        agent_args = agent_args[1:]
    exe = shutil.which(args.agent)
    if exe is None:
        print(f"actrouter-route: {args.agent} is not on PATH", file=sys.stderr)
        return 127
    chooses_model = any(a in model_flags or a.split("=", 1)[0] in model_flags for a in agent_args)
    description = args.describe or _prompt_of(agent_args)[:LAUNCH_PROMPT_CHARS]
    if not chooses_model and description:
        baseline, source = baselines.agent_cli(args.agent)
        try:
            advice = Advisor(host=host).for_task(
                description, preference=args.preference, top_k=1, baseline=baseline
            )
        except (ActRouterError, ValueError) as exc:
            print(f"actrouter-route: routing failed, starting {args.agent} unchanged: {exc}", file=sys.stderr)
        else:
            if model := advice.get("host_model"):
                print(f"actrouter-route: {advice.get('task_type')} -> {model}", file=sys.stderr)
                ledger.route_event(
                    advice, surface="launch", baseline=baseline, baseline_source=source, agent=args.agent
                )
                agent_args = [flag, model, *agent_args]
    os.execv(exe, [exe, *agent_args])
    return 0  # pragma: no cover - execv does not return


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="actrouter-route", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", default=None, help="claude-code | codex | generic (default: $ACTROUTER_HOST)")
        p.add_argument("--top-k", type=int, default=3)
        out = p.add_mutually_exclusive_group()
        out.add_argument("--model-only", action="store_true", help="Print only the model")
        out.add_argument("--json", action="store_true")

    p = sub.add_parser("task", help="Recommend a model for a unit of agent work")
    p.add_argument("description")
    p.add_argument("--title")
    p.add_argument("--task-kind")
    p.add_argument("--needs", help="Comma-separated: tools,vision")
    p.add_argument("--preference", choices=["quality", "balanced", "cost"], default="balanced")
    p.add_argument("--baseline", help="Model that would run without routing, to compare against")
    common(p)
    p.set_defaults(func=_cmd_task)

    p = sub.add_parser("function", help="Recommend a model for an LLM-calling Python function")
    p.add_argument("path")
    p.add_argument("function", help="Name or Class.method")
    p.add_argument("--root", help="Project root, for the module name")
    p.add_argument("--task-kind")
    p.add_argument("--baseline", help="Model that would run without routing, to compare against")
    common(p)
    p.set_defaults(func=_cmd_function)

    p = sub.add_parser("models", help="List the models a host can run")
    p.add_argument("--host", default=None)
    p.set_defaults(func=_cmd_models)

    p = sub.add_parser("launch", help="Start an agent CLI on the routed model")
    p.add_argument("agent", choices=sorted(LAUNCHERS))
    p.add_argument("--describe", help="Route on this description instead of the prompt")
    p.add_argument("--preference", choices=["quality", "balanced", "cost"], default="balanced")
    p.add_argument("agent_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_launch)

    p = sub.add_parser("report", help="What routing saved, from the routing ledger")
    p.add_argument("--days", type=int, default=30, help="Look back this many days (0 = all)")
    p.add_argument("--ledger", help="Ledger file (default: $ACTROUTER_LEDGER or ~/.actrouter/routing-ledger.jsonl)")
    p.add_argument("--html", help="Also write a self-contained HTML dashboard here")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ActRouterError, ValueError, LookupError, OSError) as exc:
        print(f"actrouter-route: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
