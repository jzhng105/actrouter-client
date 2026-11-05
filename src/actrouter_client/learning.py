"""Let the router update itself from how agents and users responded to it.

The routing ledger records what ActRouter decided and what happened next
(:mod:`actrouter_client.ledger`, :mod:`actrouter_client.feedback`). This
module reads those records, turns them into *signals* about a model's fit
for a task type, and writes routing adjustments the router applies (the
routing request's ``learned`` field; ``actrouter.learned`` in the engine). It reruns on a schedule, so routing keeps
following the evidence: new signals move it, and old ones fade.

Signals, each about one ``(task type, model)``:

signal         effect  weight  where it comes from
-------------  ------  ------  ------------------------------------------------
correction     bad     1.0     the user's next message said the work was wrong
approval       good    1.0     ... said it was right
declined       bad     0.5     the agent did not hand the work to the routed model
accepted       good    1.0     the user switched to the model routing recommended
override       bad     1.0     the user switched to another model (a vote for it)
failed         bad     1.0     a routed subagent or chat turn failed
interrupted    bad     0.5     ... was interrupted
completed      good    0.25    ... completed (weak: finishing is not succeeding)
escalated      bad     1.0     the same task type was redone on a larger model in
                               the same session soon after (a vote for that model)
agent_choice   bad     0.25    the agent started a subagent on another model than
                               routing would have chosen (a vote for its choice)
unclaimed      bad     0.25    the agent asked route_task and did not follow it

Weights decay with age (``half_life_days``), and only the last
``window_days`` of the ledger are read. Per ``(task type, model)``:

* the **quality estimate** is a Beta posterior over good and bad signals,
  starting from a prior of 0.8 (4 good, 1 bad): the router is trusted until
  evidence says otherwise. Below the prior, the model is penalized. Good
  signals only offset bad ones: a model that is routed often and succeeds is
  not promoted over models never tried, so routing does not lock itself in;
* **votes** for a model (an override, an escalation, an agent's choice that
  went to it) promote it, with diminishing returns;
* nothing is applied until the decayed evidence reaches ``min_evidence``,
  and the result is clamped to ``±max_adjustment`` score points (adjacent
  size tiers usually differ by 0.04-0.09, so the default 0.12 can move a
  task type one tier).

Claude Code runs a subagent on an alias (``haiku``), which runs the newest
model of its line whichever one routing chose. Evidence from Claude Code
about a model its alias stands for is therefore evidence about the alias,
and is applied to every catalog model the alias covers; an agent choosing
``haiku`` where routing chose an older Haiku is not a disagreement.

Modes (``ACTROUTER_LEARNING``): ``off`` learns nothing; ``shadow`` (the
default) learns and writes the adjustments for review, ``actrouter-route
learn`` shows them, and routing ignores them; ``on`` sends them with each
routing request (the API's ``learned`` field), which applies them to that
request only. The adjustments live in ``ACTROUTER_LEARNED`` (default
``~/.actrouter/learned.json``); the ledger never leaves the machine.

Tuning (environment, or the desktop's ``learning`` settings):
``ACTROUTER_LEARN_MIN_EVIDENCE`` (3), ``ACTROUTER_LEARN_MAX_ADJUSTMENT``
(0.12), ``ACTROUTER_LEARN_HALF_LIFE_DAYS`` (30),
``ACTROUTER_LEARN_WINDOW_DAYS`` (120), ``ACTROUTER_LEARN_REFRESH_HOURS``
(6: how stale the adjustments may get before routing relearns them).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from actrouter_client import ledger
from actrouter_client.baseline import to_catalog_id
from actrouter_client.client import env
from actrouter_client.hosts import CLAUDE_ALIASES, get_host

VERSION = 1
MODES = ("off", "shadow", "on")
DEFAULT_MODE = "shadow"

GOOD, BAD = "good", "bad"

#: signal -> (effect, weight)
WEIGHTS: dict[str, tuple[str, float]] = {
    "correction": (BAD, 1.0),
    "approval": (GOOD, 1.0),
    "declined": (BAD, 0.5),
    "accepted": (GOOD, 1.0),
    "override": (BAD, 1.0),
    "failed": (BAD, 1.0),
    "interrupted": (BAD, 0.5),
    "completed": (GOOD, 0.25),
    "escalated": (BAD, 1.0),
    "agent_choice": (BAD, 0.25),
    "unclaimed": (BAD, 0.25),
}

#: signals that are also a vote for another model, and the vote's weight
VOTES: dict[str, float] = {"override": 1.0, "escalated": 1.0, "agent_choice": 0.25}

#: the Beta prior on a routed model's fit: 4 good, 1 bad (0.8)
PRIOR_GOOD, PRIOR_BAD = 4.0, 1.0
PRIOR = PRIOR_GOOD / (PRIOR_GOOD + PRIOR_BAD)
#: score points per unit of quality estimate below the prior
PENALTY_SCALE = 0.4
#: score points at full promotion, and the votes that give half of it
PROMOTION_SCALE = 0.08
PROMOTION_HALF = 4.0

#: work on the same task type redone on a larger model within this is an escalation
ESCALATION_WINDOW_S = 20 * 60

STATUS = {
    "completed": "completed", "success": "completed", "succeeded": "completed",
    "failed": "failed", "error": "failed", "errored": "failed",
    "interrupted": "interrupted", "cancelled": "interrupted", "canceled": "interrupted",
    "killed": "interrupted", "aborted": "interrupted",
}


@dataclass(frozen=True)
class LearnConfig:
    min_evidence: float = 3.0
    max_adjustment: float = 0.12
    half_life_days: float = 30.0
    window_days: float = 120.0
    refresh_hours: float = 6.0

    @classmethod
    def from_env(cls) -> "LearnConfig":
        def number(name: str, default: float) -> float:
            try:
                value = float(env(name) or default)
            except ValueError:
                return default
            return value if value >= 0 else default

        d = cls()
        return cls(
            min_evidence=number("ACTROUTER_LEARN_MIN_EVIDENCE", d.min_evidence),
            max_adjustment=number("ACTROUTER_LEARN_MAX_ADJUSTMENT", d.max_adjustment),
            half_life_days=number("ACTROUTER_LEARN_HALF_LIFE_DAYS", d.half_life_days) or d.half_life_days,
            window_days=number("ACTROUTER_LEARN_WINDOW_DAYS", d.window_days),
            refresh_hours=number("ACTROUTER_LEARN_REFRESH_HOURS", d.refresh_hours),
        )


def mode() -> str:
    value = (env("ACTROUTER_LEARNING") or DEFAULT_MODE).lower()
    return value if value in MODES else DEFAULT_MODE


def learned_path() -> Optional[Path]:
    """Where the adjustments are kept, or None when learning has nowhere to write."""
    value = env("ACTROUTER_LEARNED")
    if value and value.lower() in ledger.OFF:
        return None
    return Path(value).expanduser() if value else Path.home() / ".actrouter" / "learned.json"


# --- signals ------------------------------------------------------------------------


@dataclass
class Signal:
    signal: str
    task_type: str
    model: str
    ts: str
    #: the model the signal votes for instead (override, escalation, agent's choice)
    voted_for: Optional[str] = None
    session_id: Optional[str] = None
    #: the agent host the signal came from (claude-code, codex, ...)
    host: Optional[str] = None


@dataclass
class _Unit:
    """A unit of routed work: a subagent, and the model it ran on."""

    task_type: str
    model: str
    ts: str
    t: float
    session_id: Any
    host: Optional[str] = None


def _when(ts: Any) -> Optional[float]:
    try:
        return dt.datetime.fromisoformat(str(ts)).timestamp()
    except ValueError:
        return None


def _known_task(task_type: Any) -> Optional[str]:
    return str(task_type) if task_type and task_type != "unknown" else None


def alias_lines(catalog: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Catalog id -> every catalog id Claude Code spells with the same alias
    (``haiku``: each Haiku), for ids it spells with an alias."""
    host = get_host("claude-code")
    by_alias: dict[str, list[str]] = defaultdict(list)
    for m in catalog:
        if m.get("id") and m.get("family") == "anthropic" and host.spell(str(m["id"])) in CLAUDE_ALIASES:
            by_alias[str(host.spell(str(m["id"])))].append(str(m["id"]))
    return {model: ids for ids in by_alias.values() for model in ids}


def signals(events: Iterable[dict[str, Any]], catalog: list[dict[str, Any]], *,
            now: Optional[float] = None) -> list[Signal]:
    """The signals ledger `events` hold, oldest first."""
    now = time.time() if now is None else now
    ids = {str(m["id"]) for m in catalog if m.get("id")}
    tiers = {str(m["id"]): m.get("size_tier") for m in catalog if m.get("id")}
    lines = alias_lines(catalog)

    def same(a: Optional[str], b: Optional[str], host: Any) -> bool:
        """Whether `a` and `b` are the same model for the host: one alias in Claude Code."""
        return a == b or (host == "claude-code" and b in lines.get(a or "", ()))

    def catalog_id(model: Any) -> Optional[str]:
        if not model:
            return None
        return str(model) if str(model) in ids else to_catalog_id(str(model), catalog)

    out: list[Signal] = []
    units: dict[tuple[Any, Any], _Unit] = {}           # (session, tool_use_id) -> unit
    by_agent: dict[tuple[Any, Any], _Unit] = {}        # (session, agent_id) -> unit
    judged: set[tuple[Any, Any]] = set()               # units whose status was counted
    sequence: dict[Any, list[_Unit]] = defaultdict(list)
    claimed: set[Any] = set()
    advice: list[dict[str, Any]] = []

    def add(signal: str, task_type: Any, model: Any, ts: Any, **extra: Any) -> None:
        task, m = _known_task(task_type), catalog_id(model)
        if task and m:
            out.append(Signal(signal, task, m, str(ts), **extra))

    def status(unit: _Unit, value: Any, key: tuple[Any, Any], ts: Any) -> None:
        verdict = STATUS.get(str(value or "").lower())
        if verdict and key not in judged:
            judged.add(key)
            add(verdict, unit.task_type, unit.model, ts, session_id=unit.session_id, host=unit.host)

    for e in sorted(events, key=lambda e: str(e.get("ts") or "")):
        kind, ts, session = e.get("kind"), e.get("ts"), e.get("session_id")
        t = _when(ts)
        if kind == "feedback":
            signal = str(e.get("signal") or "")
            if signal in WEIGHTS:
                chosen = catalog_id(e.get("chosen_model") or e.get("chosen_host_model"))
                add(signal, e.get("task_type"), e.get("model"), ts, session_id=session, host=e.get("host"),
                    voted_for=chosen if signal in VOTES and not same(catalog_id(e.get("model")), chosen,
                                                                       e.get("host")) else None)
        elif kind == "route":
            if e.get("claims"):
                claimed.add(e["claims"])
            task, model = _known_task(e.get("task_type")), catalog_id(e.get("routed_model"))
            if e.get("tool_use_id") and task and model and t is not None:
                unit = _Unit(task, model, str(ts), t, session, e.get("host"))
                units[(session, e["tool_use_id"])] = unit
                sequence[session].append(unit)
            elif (e.get("surface") == "mcp" and e.get("tool") in (None, "route_task")
                  and e.get("host") == "claude-code" and e.get("request_id")):
                advice.append(e)  # claimable by a subagent; judged below
        elif kind == "skipped" and e.get("reason") == "explicit_model":
            task, chosen = _known_task(e.get("task_type")), catalog_id(e.get("chosen_model"))
            routed = catalog_id(e.get("routed_model"))
            if task and chosen and t is not None:
                unit = _Unit(task, chosen, str(ts), t, session, e.get("host"))
                if e.get("tool_use_id"):
                    units[(session, e["tool_use_id"])] = unit
                sequence[session].append(unit)
                if routed and not same(routed, chosen, e.get("host")):
                    add("agent_choice", task, routed, ts, session_id=session, host=e.get("host"),
                        voted_for=chosen)
        elif kind == "outcome":
            if e.get("tool_use_id"):
                key = (session, e["tool_use_id"])
                unit = units.get(key)
                if unit is None:
                    continue
                if e.get("status") == "async_launched":
                    if e.get("agent_id"):
                        by_agent[(session, e["agent_id"])] = unit
                    continue
                status(unit, e.get("status"), key, ts)
            elif e.get("agent_id"):
                key = (session, e["agent_id"])
                if (unit := by_agent.get(key)) is not None:
                    status(unit, e.get("status"), key, ts)
        elif kind == "turn":
            verdict = STATUS.get(str(e.get("status") or "").lower())
            if verdict:
                add(verdict, e.get("task_type"), e.get("served_model") or e.get("applied_model"), ts,
                    session_id=session, host=e.get("host"))

    # Work redone on a larger model, on the same task type, soon after.
    for seq in sequence.values():
        seq.sort(key=lambda u: u.t)
        for i, unit in enumerate(seq):
            tier = tiers.get(unit.model)
            if tier is None:
                continue
            for later in seq[i + 1:]:
                if later.t - unit.t > ESCALATION_WINDOW_S:
                    break
                bigger = tiers.get(later.model)
                if later.task_type == unit.task_type and bigger is not None and bigger > tier:
                    add("escalated", unit.task_type, unit.model, later.ts, session_id=unit.session_id,
                        host=unit.host, voted_for=later.model)
                    break

    # route_task advice no subagent claimed once it could no longer be
    for e in advice:
        t = _when(e.get("ts"))
        if e["request_id"] not in claimed and t is not None and now - t > ledger.OFFER_TTL_S:
            add("unclaimed", e.get("task_type"), e.get("routed_model"), e.get("ts"),
                session_id=e.get("session_id"), host=e.get("host"))

    out.sort(key=lambda s: s.ts)
    return out


# --- adjustments --------------------------------------------------------------------


@dataclass
class Evidence:
    good: float = 0.0
    bad: float = 0.0
    votes: float = 0.0
    signals: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.good + self.bad + self.votes

    def quality(self) -> float:
        return (self.good + PRIOR_GOOD) / (self.good + self.bad + PRIOR_GOOD + PRIOR_BAD)

    def bias(self, cfg: LearnConfig) -> float:
        # at the precision the evidence is shown with: three signals recorded
        # a minute ago are three signals, not 2.9999
        if round(self.total, 2) < cfg.min_evidence:
            return 0.0
        penalty = min(0.0, self.quality() - PRIOR)
        promotion = self.votes / (self.votes + PROMOTION_HALF)
        raw = PENALTY_SCALE * penalty + PROMOTION_SCALE * promotion
        return round(max(-cfg.max_adjustment, min(cfg.max_adjustment, raw)), 4)


def _decay(ts: str, now: float, half_life_days: float) -> float:
    t = _when(ts)
    if t is None:
        return 0.0
    age_days = max(0.0, now - t) / 86400
    return 0.5 ** (age_days / half_life_days)


def evidence(found: Iterable[Signal], cfg: LearnConfig, *, now: float,
             lines: Optional[dict[str, list[str]]] = None) -> dict[str, dict[str, Evidence]]:
    """Decayed evidence by task type, then model. With `lines`
    (:func:`alias_lines`), Claude Code evidence about a model counts for
    every model its alias covers."""
    lines = lines or {}
    by: dict[str, dict[str, Evidence]] = defaultdict(lambda: defaultdict(Evidence))

    def models(model: str, host: Optional[str]) -> list[str]:
        return lines.get(model, [model]) if host == "claude-code" else [model]

    for s in found:
        decay = _decay(s.ts, now, cfg.half_life_days)
        effect, weight = WEIGHTS[s.signal]
        for model in models(s.model, s.host):
            e = by[s.task_type][model]
            setattr(e, effect, getattr(e, effect) + weight * decay)
            e.signals[s.signal] = e.signals.get(s.signal, 0) + 1
        for model in models(s.voted_for, s.host) if s.voted_for else []:
            v = by[s.task_type][model]
            v.votes += VOTES[s.signal] * decay
            key = f"{s.signal}_vote"
            v.signals[key] = v.signals.get(key, 0) + 1
    return by


def learn(events: Iterable[dict[str, Any]], catalog: list[dict[str, Any]],
          cfg: Optional[LearnConfig] = None, *, now: Optional[float] = None,
          source: Optional[str] = None) -> dict[str, Any]:
    """The adjustments file's content, from ledger `events`."""
    cfg = cfg or LearnConfig.from_env()
    now = time.time() if now is None else now
    events = list(events)
    found = signals(events, catalog, now=now)
    counts: dict[str, int] = defaultdict(int)
    for s in found:
        counts[s.signal] += 1
    adjustments: dict[str, dict[str, Any]] = {}
    for task, models in sorted(evidence(found, cfg, now=now, lines=alias_lines(catalog)).items()):
        for model, e in sorted(models.items()):
            adjustments.setdefault(task, {})[model] = {
                "bias": e.bias(cfg),
                "evidence": round(e.total, 3),
                "good": round(e.good, 3),
                "bad": round(e.bad, 3),
                "votes": round(e.votes, 3),
                "quality": round(e.quality(), 3),
                "signals": dict(sorted(e.signals.items())),
            }
    return {
        "version": VERSION,
        "generated_at": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(timespec="seconds"),
        "ledger": source,
        "events": len(events),
        "signals": dict(sorted(counts.items())),
        "config": asdict(cfg),
        "adjustments": adjustments,
    }


def _write(target: Path, payload: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def relearn(catalog: list[dict[str, Any]], *, cfg: Optional[LearnConfig] = None,
            source: Optional[Path] = None, target: Optional[Path] = None,
            now: Optional[float] = None, write: bool = True) -> dict[str, Any]:
    """Learn from the ledger now, and (with `write`) keep the result."""
    cfg = cfg or LearnConfig.from_env()
    now = time.time() if now is None else now
    source = source or ledger.path()
    since = (dt.datetime.fromtimestamp(now - cfg.window_days * 86400, dt.timezone.utc)
             if cfg.window_days > 0 else None)
    events = list(ledger.read(source, since)) if source is not None else []
    payload = learn(events, catalog, cfg, now=now, source=str(source) if source else None)
    target = target or learned_path()
    if write and target is not None:
        _write(target, payload)
    return payload


def stale(target: Optional[Path], cfg: LearnConfig, *, now: Optional[float] = None) -> bool:
    """True when the adjustments in `target` are older than the refresh interval."""
    if target is None:
        return False
    now = time.time() if now is None else now
    try:
        return now - target.stat().st_mtime >= cfg.refresh_hours * 3600
    except OSError:
        return True


def refresh(catalog: Callable[[], list[dict[str, Any]]], *, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """Relearn when learning is on or in shadow, there is a ledger, and the
    adjustments are stale. Returns what was learned, or None."""
    if mode() == "off":
        return None
    cfg = LearnConfig.from_env()
    source, target = ledger.path(), learned_path()
    if source is None or target is None or not source.exists() or not stale(target, cfg, now=now):
        return None
    try:
        if target.exists():
            os.utime(target)  # claim this refresh: concurrent hooks need not repeat it
    except OSError:
        pass
    return relearn(catalog(), cfg=cfg, source=source, target=target, now=now)


_activated: Optional[tuple[str, Optional[str]]] = None
_sent: Optional[tuple[str, float, Optional[dict[str, dict[str, float]]]]] = None


def activate(catalog: Callable[[], list[dict[str, Any]]]) -> Optional[dict[str, dict[str, float]]]:
    """Called before a routing request: relearn if stale (once per process
    and settings), then return what the request carries as ``learned`` --
    task type -> model -> bias -- when the mode is ``on``, else None. The
    ActRouter API applies it to that request only (``actrouter.learned``).
    Never raises."""
    global _activated
    current = mode()
    target = learned_path()
    key = (current, str(target) if target else None)
    if _activated != key:
        _activated = key
        try:
            refresh(catalog)
        except Exception as exc:  # noqa: BLE001 - learning must never break routing
            print(f"actrouter: could not relearn routing adjustments: {exc}", file=sys.stderr)
    if current != "on" or target is None:
        return None
    return applied(target)


def applied(target: Path) -> Optional[dict[str, dict[str, float]]]:
    """The non-zero biases in the adjustments file `target`, or None when
    there are none (or it cannot be read). Cached on the file's mtime."""
    global _sent
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return None
    if _sent is not None and _sent[0] == str(target) and _sent[1] == mtime:
        return _sent[2]
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    out: dict[str, dict[str, float]] = {}
    if isinstance(payload, dict) and payload.get("version") == VERSION:
        for task, models in (payload.get("adjustments") or {}).items():
            for model, e in (models or {}).items() if isinstance(models, dict) else ():
                bias = e.get("bias") if isinstance(e, dict) else None
                if isinstance(bias, (int, float)) and bias == bias and bias != 0:
                    out.setdefault(str(task), {})[str(model)] = float(bias)
    result = out or None
    _sent = (str(target), mtime, result)
    return result


def reset() -> None:
    """Forget the activation (tests, and a process that changes its settings)."""
    global _activated, _sent
    _activated, _sent = None, None


# --- presentation -------------------------------------------------------------------


def format_text(payload: dict[str, Any], *, applied: bool, target: Optional[Path]) -> str:
    cfg = payload.get("config") or {}
    lines = [
        f"learning:  {mode()} ({'applied to routing' if applied else 'not applied to routing'})",
        f"file:      {target or '(not written)'}",
        f"ledger:    {payload.get('ledger') or '(none)'}: {payload.get('events', 0)} events, "
        f"last {cfg.get('window_days', 0):g} days, half-life {cfg.get('half_life_days', 0):g} days",
        "signals:   " + (", ".join(f"{k} {v}" for k, v in (payload.get("signals") or {}).items()) or "none yet"),
    ]
    rows = [(task, model, e) for task, models in (payload.get("adjustments") or {}).items()
            for model, e in models.items()]
    if not rows:
        lines.append("\nNo evidence yet: routing is unchanged.")
        return "\n".join(lines)
    rows.sort(key=lambda r: (r[0], -abs(r[2]["bias"]), r[1]))
    lines.append("")
    lines.append(f"{'task type':<16} {'model':<28} {'bias':>7} {'evidence':>8} {'good':>6} {'bad':>6} {'votes':>6}")
    for task, model, e in rows:
        short = round(e["evidence"], 2) < cfg.get("min_evidence", 0)
        mark = " *" if short else ""
        lines.append(f"{task:<16} {model:<28} {e['bias']:>+7.3f} {e['evidence']:>8.2f} "
                     f"{e['good']:>6.2f} {e['bad']:>6.2f} {e['votes']:>6.2f}{mark}")
    if any(round(e["evidence"], 2) < cfg.get("min_evidence", 0) for _, _, e in rows):
        lines.append(f"* less evidence than min_evidence ({cfg.get('min_evidence'):g}): not applied yet")
    return "\n".join(lines)
