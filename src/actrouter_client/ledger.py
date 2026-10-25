"""The routing ledger: a local record of what ActRouter decided, and what ran.

Each line is one JSON event in ``~/.actrouter/routing-ledger.jsonl`` (or
``ACTROUTER_LEDGER``; ``off`` disables recording):

``route``    a model was routed: the task type, the routed model, the
             *baseline* (the model that would have run without routing,
             :mod:`actrouter_client.baseline`) and both estimated costs;
``skipped``  a subagent already had a model, so routing left it alone --
             counted for coverage;
``prompt``   a user's prompt fit another model than the session's: the task
             type, both models, whether Claude was told to delegate, and
             and what it was routed on: the prompt, the routing the
             conversation is pinned to, or the exchange it continues.
             The work Claude then hands to that model is recorded as a
             ``route`` with ``surface: prompt_hook``;
``outcome``  a routed subagent finished: the model it actually ran on and
             its real token usage, from Claude Code's ``PostToolUse`` event.

Advice from the MCP ``route_task`` tool is a ``route`` without a tool call:
the agent then starts the subagent with that model itself. So the MCP
server also leaves an *offer* (:func:`offer`), kept per project for
``OFFER_TTL_S``, and the ``PreToolUse`` hook *claims* it (:func:`claim`)
when a subagent starts with the advised model: it records the route again,
with the tool call's ids and ``claims`` set to the advice's request id, so
the subagent's outcome is measured; the report then counts the claimed
advice once.

``actrouter-route report`` turns it into savings (see
:mod:`actrouter_client.report`).

What is recorded is metadata about the decision only: no task description,
prompt, file content or tool output. Recording never raises -- a full disk
or an unwritable home directory must not break routing.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from actrouter_client.client import env

SCHEMA = 1
OFF = frozenset({"0", "off", "false", "no", "none"})

#: how long route_task advice stays claimable by the subagent it advised on
OFFER_TTL_S = 15 * 60

#: the advice fields a route event records (no task text: `reasons` and
#: classifier evidence can quote the description)
ROUTE_FIELDS = (
    "host", "model", "host_model", "task_type", "task_confidence", "task_source",
    "baseline_model", "estimated_cost_usd", "baseline_estimated_cost_usd",
    "estimated_savings_usd", "estimate_basis", "router", "routed_by", "request_id",
)


def path() -> Optional[Path]:
    """Where events go, or None when recording is off."""
    value = env("ACTROUTER_LEDGER")
    if value and value.lower() in OFF:
        return None
    return Path(value).expanduser() if value else Path.home() / ".actrouter" / "routing-ledger.jsonl"


def record(kind: str, **fields: Any) -> None:
    """Append one event. Fields that are None are left out."""
    target = path()
    if target is None:
        return
    event = {
        "v": SCHEMA,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        **{k: v for k, v in fields.items() if v is not None},
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # One write per event on an O_APPEND file: concurrent hooks interleave
        # whole lines, not fragments.
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"actrouter: could not record to {target}: {exc}", file=sys.stderr)


def _offers_file(project: Optional[str]) -> Optional[Path]:
    if not project or path() is None:
        return None
    key = hashlib.sha256(str(Path(project).expanduser().resolve()).encode()).hexdigest()[:16]
    root = env("ACTROUTER_STATE_DIR")
    return (Path(root).expanduser() if root else Path.home() / ".actrouter" / "sessions") / f"offers-{key}.json"


def _live_offers(target: Path, now: float) -> list[dict[str, Any]]:
    try:
        offers = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [o for o in offers if isinstance(o, dict) and now - float(o.get("t", 0)) < OFFER_TTL_S] \
        if isinstance(offers, list) else []


def _write_offers(target: Path, offers: list[dict[str, Any]]) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(offers[-20:]), encoding="utf-8")
    except OSError as exc:
        print(f"actrouter: could not keep route_task advice: {exc}", file=sys.stderr)


def offer(advice: dict[str, Any], *, project: Optional[str], baseline: Optional[str],
          baseline_source: str) -> None:
    """Keep route_task `advice` claimable by the subagent it advises on."""
    target = _offers_file(project)
    if target is None or not advice.get("host_model") or not advice.get("request_id"):
        return
    now = time.time()
    entry = {**{k: advice.get(k) for k in ROUTE_FIELDS}, "baseline": baseline,
             "baseline_source": baseline_source, "t": now}
    _write_offers(target, [*_live_offers(target, now), entry])


def claim(project: Optional[str], model: str) -> Optional[dict[str, Any]]:
    """The newest live offer for `project` whose model is `model` (as the
    host spells it, or the catalog id), removed so it is claimed once."""
    target = _offers_file(project)
    if target is None:
        return None
    now = time.time()
    offers = _live_offers(target, now)
    wanted = model.strip().lower()
    for i in range(len(offers) - 1, -1, -1):
        if wanted in (str(offers[i].get("host_model")).lower(), str(offers[i].get("model")).lower()):
            found = offers.pop(i)
            _write_offers(target, offers)
            return found
    return None


def route_event(advice: dict[str, Any], *, surface: str, baseline: Optional[str],
                baseline_source: str, **context: Any) -> None:
    """Record a routing decision from an :meth:`Advisor.advice` answer."""
    record(
        "route",
        surface=surface,
        host=advice.get("host"),
        task_type=advice.get("task_type"),
        task_confidence=advice.get("task_confidence"),
        task_source=advice.get("task_source"),
        routed_model=advice.get("model"),
        routed_host_model=advice.get("host_model"),
        baseline_model=advice.get("baseline_model"),
        baseline_as_given=baseline,
        baseline_source=baseline_source,
        estimated_cost_usd=advice.get("estimated_cost_usd"),
        baseline_estimated_cost_usd=advice.get("baseline_estimated_cost_usd"),
        estimated_savings_usd=advice.get("estimated_savings_usd"),
        estimate_basis=advice.get("estimate_basis"),
        router=advice.get("router"),
        routed_by=advice.get("routed_by"),
        request_id=advice.get("request_id"),
        **context,
    )


def read(source: Optional[Path] = None, since: Optional[dt.datetime] = None) -> Iterator[dict[str, Any]]:
    """Events from the ledger, oldest first; malformed lines are skipped."""
    target = source or path()
    if target is None or not target.exists():
        return
    with open(target, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or "kind" not in event:
                continue
            if since is not None:
                try:
                    if dt.datetime.fromisoformat(str(event.get("ts"))) < since:
                        continue
                except ValueError:
                    continue
            yield event

