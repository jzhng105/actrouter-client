"""Test which model transitions work in practice, before routing uses them.

The agent's official model-switching call keeps the runtime informed, but
that does not prove every transition works: reasoning items or thinking
blocks one model wrote may not be accepted by another. For each ordered
pair ``a -> b``, :func:`check_transitions` opens a fresh session on ``a`` and
runs a turn that reasons and makes one tool call. It then switches to ``b``
with the same call routing uses and asks about the earlier turn. A pair works
when the second turn completes on ``b``. The results go to
:class:`~actrouter_client.session_state.Transitions`, which the switch policy
reads (``transition_untested`` / ``transition_unsupported``).

This calls the models for real, so it costs a little: two short turns per pair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from actrouter_client.chat.base import Agent
from actrouter_client.chat.ui import ScriptedUI
from actrouter_client.session_state import Transitions

PROBE = "echo actrouter-transition-probe"

FIRST = (
    "Think this through step by step before answering. Run the shell command "
    f"`{PROBE}` and tell me exactly what it printed. Then say in one sentence whether "
    "391 is prime, and why."
)
SECOND = "What exactly did that command print, and what did you conclude about 391? One sentence."


@dataclass
class PairResult:
    source: str
    target: str
    ok: Optional[bool]
    detail: str


def _approve_probe(kind: str, summary: str) -> bool:
    """Approve the probe command only; anything else the model tries is refused."""
    return PROBE in summary


async def check_pair(make_agent: Callable[[], Agent], a: str, b: str, *,
                    same: Callable[[str, str], bool]) -> PairResult:
    agent = make_agent()
    ui = ScriptedUI(approve_if=_approve_probe)
    try:
        try:
            await agent.start(model=a, resume=None)
            first = await agent.run_turn(FIRST, ui)
        except Exception as exc:  # noqa: BLE001 - says nothing about the transition
            return PairResult(a, b, None, f"setup on {a} failed: {type(exc).__name__}: {exc}")
        if first.status != "completed":
            return PairResult(a, b, None, f"setup turn on {a} did not complete: {first.status} {first.error or ''}")
        await agent.set_model(b)
        second = await agent.run_turn(SECOND, ui)
        if second.status != "completed":
            return PairResult(a, b, False, f"turn on {b} {second.status}: {second.error or ''}".strip())
        served = second.models[-1] if second.models else None
        if served and not same(served, b):
            return PairResult(a, b, False, f"turn ran on {served}, not {b}")
        return PairResult(a, b, True, "ok")
    except Exception as exc:  # noqa: BLE001 - any failure is a result
        return PairResult(a, b, False, f"{type(exc).__name__}: {exc}")
    finally:
        await agent.close()


async def check_transitions(make_agent: Callable[[], Agent], models: list[str], transitions: Transitions, *,
                           catalog_id: Callable[[str], str],
                           report: Callable[[PairResult], None] = lambda r: None) -> list[PairResult]:
    """Test every ordered pair of `models` (as the agent spells them) and
    record the results under their catalog ids."""
    results = []
    for a in models:
        for b in models:
            if a == b:
                continue
            r = await check_pair(make_agent, a, b, same=lambda x, y: catalog_id(x) == catalog_id(y))
            if r.ok is not None:
                transitions.record(catalog_id(a), catalog_id(b), r.ok, source="matrix",
                                   detail=r.detail, at=int(time.time()))
            report(r)
            results.append(r)
    return results
