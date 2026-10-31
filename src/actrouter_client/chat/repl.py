"""The ``actrouter-route chat`` loop: a small agent client, not the agent's TUI.

It supports a limited set of commands; everything else is sent to the agent
as a prompt. Ctrl-C during a turn interrupts the turn; Ctrl-D (or ``/quit``)
ends the session, which can be resumed with ``--resume <id>``.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Optional

from actrouter_client.chat.session import MODES, RoutedSession

HELP = """commands:
  /model <name>       switch to this model and keep it (routing leaves it alone)
  /model auto         hand the model back to routing
  /route on|shadow|off  switch per turn | only record what it would do | do not route
  /status             session, model, routing mode and the last decision
  /help               this list
  /quit               end the session (resume it with --resume <id>)
Ctrl-C during a turn interrupts it."""


def _status(session: RoutedSession) -> str:
    st = session.state
    if st is None:
        return "no session"
    lines = [
        f"session  {st.session_id} ({session.agent.name})",
        f"model    {st.current_host_model or '?'} ({st.current_model or '?'})"
        + ("  [chosen with /model]" if st.manual else ""),
        f"routing  {session.cfg.mode}",
        f"context  {st.context_tokens if st.context_tokens is not None else '?'} tokens, turn {st.turn}",
    ]
    d = session.last_decision
    if d is not None:
        lines.append(f"last     {d.reason}: recommended {d.recommended_host_model or d.recommended_model or '-'}"
                     f", applied {d.applied_host_model or d.applied_model or '-'}")
        gate = d.gate
        if "stay_cost_usd" in gate:
            lines.append(f"         stay ${gate['stay_cost_usd']:.4f} vs switch ${gate['switch_cost_usd']:.4f}"
                         f" (needs ${gate['savings_needed_usd']:.4f} saved, {gate['workload']['basis']} workload)")
    return "\n".join(lines)


async def _command(session: RoutedSession, line: str) -> Optional[bool]:
    """Handle a slash command: True to quit, False when handled, None when
    `line` is not a wrapper command (it then goes to the agent as is)."""
    parts = line.split()
    cmd, args = parts[0].lower(), parts[1:]
    ui = session.ui
    if cmd in ("/quit", "/exit"):
        return True
    if cmd == "/help":
        ui.notice(HELP)
        return False
    if cmd == "/status":
        ui.notice(_status(session))
        return False
    if cmd == "/route":
        if len(args) != 1 or args[0] not in MODES:
            ui.notice("usage: /route on|shadow|off")
        else:
            session.cfg.mode = args[0]
            ui.notice(f"routing {args[0]}")
        return False
    if cmd == "/model":
        if len(args) != 1:
            ui.notice("usage: /model <name> | /model auto")
        elif args[0].lower() == "auto":
            await session.set_manual_model(None)
            ui.notice("the model is routed again")
        else:
            await session.set_manual_model(args[0])
            ui.notice(f"model {args[0]}; routing leaves it alone until /model auto")
        return False
    return None


async def run(session: RoutedSession, *, first_prompt: Optional[str] = None) -> int:
    ui = session.ui
    loop = asyncio.get_running_loop()
    agent = session.agent
    st = session.state
    if st is not None:
        ui.notice(f"{agent.name} session {agent.session_id or '(new)'} on {st.current_host_model or 'its default model'}"
                  f", routing {session.cfg.mode}. /help for commands.")
    pending = first_prompt
    try:
        while True:
            line = pending if pending is not None else await ui.read("› ")
            pending = None
            if line is None:
                return 0
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                handled = await _command(session, line)
                if handled is True:
                    return 0
                if handled is False:
                    continue
            interrupting = False

            def on_sigint() -> None:
                nonlocal interrupting
                if not interrupting:
                    interrupting = True
                    ui.notice("interrupting...")
                    asyncio.ensure_future(agent.interrupt())

            try:
                loop.add_signal_handler(signal.SIGINT, on_sigint)
                installed = True
            except (NotImplementedError, RuntimeError, ValueError):
                installed = False
            try:
                result = await session.turn(line)
            finally:
                if installed:
                    loop.remove_signal_handler(signal.SIGINT)
            if result.status == "interrupted":
                ui.notice("turn interrupted")
            elif result.status == "failed":
                ui.notice(f"turn failed: {result.error}")
    finally:
        sid = agent.session_id
        await session.close()
        if sid:
            ui.notice(f"resume with: actrouter-route chat {agent.name} --resume {sid}")
