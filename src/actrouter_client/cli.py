"""``actrouter-route``: ActRouter from the shell.

    actrouter-route task "Fix the failing auth test" --host claude-code
    actrouter-route task "Search the repo for usages of parse_args" --host codex --model-only
    actrouter-route function src/news.py summarize_article
    actrouter-route models --host codex
    actrouter-route launch codex -- "add retries to the http client"
    actrouter-route launch claude --describe "write release notes" -- -p "..."
    actrouter-route chat codex                    # route every turn (shadow mode first)
    actrouter-route chat claude --route on "fix the failing auth test"
    actrouter-route chat codex --test-transitions gpt-5.5,gpt-5.4-mini
    actrouter-route report --days 30 --html savings.html
    actrouter-route upgrade              # free -> Pro, in Stripe Checkout
    actrouter-route billing              # card, invoices, cancelling

``launch`` routes on the agent's prompt (or ``--describe``), then replaces
itself with the agent CLI, passing the model (``codex -m``, ``claude
--model``). A model given explicitly is kept, and a routing failure starts
the agent unchanged. ``chat`` keeps the session instead and routes every
prompt, switching the agent's own session model before the turn runs (see
``docs/chat_routing.md``).
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


def _cmd_chat(args: argparse.Namespace) -> int:
    import asyncio

    from actrouter_client import routing_policy as policy
    from actrouter_client.chat import repl
    from actrouter_client.chat.session import ChatConfig, RoutedSession
    from actrouter_client.chat.ui import TerminalUI
    from actrouter_client.session_state import Transitions

    def make_agent():  # type: ignore[no-untyped-def]
        if args.agent == "codex":
            from actrouter_client.chat.codex import CodexAgent

            return CodexAgent(codex_bin=args.codex_bin, effort=args.effort,
                              approval_policy=args.approval_policy, sandbox=args.sandbox,
                              config=args.config)
        from actrouter_client.chat.claude import ClaudeAgent

        return ClaudeAgent(effort=args.effort, permission_mode=args.permission_mode, cli_path=args.claude_bin)

    host = LAUNCHERS[args.agent][0]
    advisor = Advisor(host=host)
    transitions = Transitions(args.agent)
    ui = TerminalUI()

    if args.test_transitions:
        from actrouter_client.chat.matrix import check_transitions

        models = [m.strip() for m in args.test_transitions.split(",") if m.strip()]
        if len(models) < 2:
            print("actrouter-route: --test-transitions needs two or more models", file=sys.stderr)
            return 2

        def report(r) -> None:  # type: ignore[no-untyped-def]
            mark = {True: "ok  ", False: "FAIL", None: "skip"}[r.ok]
            print(f"{mark} {r.source} -> {r.target}: {r.detail}")

        results = asyncio.run(check_transitions(
            make_agent, models, transitions,
            catalog_id=lambda m: advisor.baseline_id(m) or m, report=report,
        ))
        print(f"recorded in {transitions.path}")
        return 0 if all(r.ok is not False for r in results) else 1

    cfg = ChatConfig(
        mode=args.route,
        preference=args.preference,
        prompt_chars=int(os.environ.get("ACTROUTER_CHAT_PROMPT_CHARS", "600") or 600),
        policy=policy.PolicyConfig(
            min_savings_usd=args.min_savings,
            rel_margin=args.margin,
            cooldown_turns=args.cooldown_turns,
            cooldown_s=args.cooldown_seconds,
            upgrades=args.upgrades,
            allow_untested=args.allow_untested,
        ),
        window_cap=args.window_cap,
    )
    first = " ".join(args.prompt).strip() or None

    async def main_loop() -> int:
        session = RoutedSession(make_agent(), advisor, ui, cfg, transitions=transitions)
        await session.start(model=args.model, resume=args.resume, first_prompt=first)
        return await repl.run(session, first_prompt=first)

    try:
        return asyncio.run(main_loop())
    except KeyboardInterrupt:
        return 130


def _cmd_billing_link(args: argparse.Namespace) -> int:
    from actrouter_client.client import connect

    url = connect().billing_link(args.link)  # type: ignore[attr-defined]
    print(url)
    if not args.no_browser:
        import webbrowser

        webbrowser.open(url)
    return 0


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

    p = sub.add_parser("chat", help="Run an agent session whose model is routed every turn")
    p.add_argument("agent", choices=sorted(LAUNCHERS))
    p.add_argument("prompt", nargs="*", help="A first prompt (optional)")
    p.add_argument("--route", choices=["shadow", "on", "off"],
                   default=os.environ.get("ACTROUTER_CHAT_ROUTING", "shadow"),
                   help="shadow (default): decide and record, never switch; on: switch; off: no routing")
    p.add_argument("--model", help="Start on this model and keep it (routing leaves it alone)")
    p.add_argument("--resume", help="Resume this session (Codex thread id / Claude session id)")
    p.add_argument("--effort", help="Reasoning effort, set once and kept (not routed)")
    p.add_argument("--preference", choices=["quality", "balanced", "cost"], default="balanced")
    p.add_argument("--min-savings", type=float, default=0.01, help="$ a switch must save per task (default 0.01)")
    p.add_argument("--margin", type=float, default=0.2,
                   help="Share of staying's cost a switch must save (default 0.2; +0.15 on prior workloads)")
    p.add_argument("--cooldown-turns", type=int, default=2)
    p.add_argument("--cooldown-seconds", type=float, default=120.0)
    p.add_argument("--upgrades", choices=["quality", "cost"], default="quality",
                   help="quality (default): a switch to a larger model needs only the hard gates")
    p.add_argument("--allow-untested", action="store_true",
                   help="Switch over model transitions --test-transitions has not checked")
    p.add_argument("--window-cap", type=int, help="Context window assumed for models the agent has not reported")
    p.add_argument("--test-transitions", metavar="M1,M2,...",
                   help="Check which switches between these models work, record them, and exit")
    p.add_argument("--codex-bin", help="codex executable (default: codex on PATH)")
    p.add_argument("--approval-policy", help="Codex approval policy (untrusted, on-request, never)")
    p.add_argument("--sandbox", help="Codex sandbox (read-only, workspace-write, danger-full-access)")
    p.add_argument("-c", "--config", action="append", default=[], help="Codex -c key=value, repeatable")
    p.add_argument("--permission-mode", help="Claude permission mode (default, acceptEdits, plan, ...)")
    p.add_argument("--claude-bin", help="claude executable for the Agent SDK (default: the bundled one)")
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("report", help="What routing saved, from the routing ledger")
    p.add_argument("--days", type=int, default=30, help="Look back this many days (0 = all)")
    p.add_argument("--ledger", help="Ledger file (default: $ACTROUTER_LEDGER or ~/.actrouter/routing-ledger.jsonl)")
    p.add_argument("--html", help="Also write a self-contained HTML dashboard here")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("upgrade", help="Move your API key to the Pro plan (opens Stripe Checkout)")
    p.add_argument("--no-browser", action="store_true", help="Only print the link")
    p.set_defaults(func=_cmd_billing_link, link="upgrade")

    p = sub.add_parser("billing", help="Manage your plan: card, invoices, cancelling (Stripe portal)")
    p.add_argument("--no-browser", action="store_true", help="Only print the link")
    p.set_defaults(func=_cmd_billing_link, link="portal")

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ActRouterError, ValueError, LookupError, OSError) as exc:
        print(f"actrouter-route: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
