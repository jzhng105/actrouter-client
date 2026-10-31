"""Per-session state kept on disk: the hook's pins and the chat wrapper's routing state.

Everything lives under ``ACTROUTER_STATE_DIR`` (default
``~/.actrouter/sessions``), one small JSON file per session and purpose. It
holds models, token counts and decisions only, never prompt text. Writes
never raise: a full disk or an unwritable home directory must not break the
agent.

* :func:`save` / :func:`load` -- a JSON object per ``(session, suffix)``;
* :class:`ChatState` -- what ``actrouter-route chat`` needs to route a
  session's next turn: the model it runs on, each model's cache record, the
  latest request's context size, the last switch and running means of the
  session's turns;
* :class:`Transitions` -- model transitions known to work or fail, per
  agent, from ``actrouter-route chat --test-transitions`` and from switches
  that failed at runtime.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def root() -> Path:
    value = os.environ.get("ACTROUTER_STATE_DIR", "").strip()
    return Path(value).expanduser() if value else Path.home() / ".actrouter" / "sessions"


def state_file(session_id: Any, suffix: str) -> Optional[Path]:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id or ""))
    if not name.strip("._"):
        return None
    return root() / f"{name}.{suffix}.json"


def _write(target: Path, value: Any) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        print(f"actrouter: could not keep session state: {exc}", file=sys.stderr)


def save(session_id: Any, suffix: str, value: Optional[dict[str, Any]]) -> None:
    target = state_file(session_id, suffix)
    if target is None:
        return
    if value is None:
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            print(f"actrouter: could not keep session state: {exc}", file=sys.stderr)
        return
    _write(target, value)


def load(session_id: Any, suffix: str) -> Optional[dict[str, Any]]:
    target = state_file(session_id, suffix)
    try:
        value = json.loads(target.read_text(encoding="utf-8")) if target else None
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# --- the chat wrapper's routing state --------------------------------------------------

#: weight of the newest turn in the running means of a session's workload
OBSERVED_WEIGHT = 0.3


@dataclass
class ChatState:
    session_id: str
    agent: str
    #: catalog id of the model the session runs on, when known
    current_model: Optional[str] = None
    #: the agent's own spelling of it
    current_host_model: Optional[str] = None
    #: prompt tokens of the latest request, after any compaction
    context_tokens: Optional[int] = None
    turn: int = 0
    last_switch_turn: Optional[int] = None
    last_switch_t: Optional[float] = None
    #: catalog id -> {"t": last request, "prefix": its prompt tokens}
    caches: dict[str, Any] = field(default_factory=dict)
    #: catalog id -> context window the agent reported for it
    windows: dict[str, int] = field(default_factory=dict)
    #: running means: turns, rounds, growth, output (see routing_policy.workload)
    observed: dict[str, Any] = field(default_factory=dict)
    pin: Optional[dict[str, Any]] = None
    #: the user chose the model (/model): routing does not switch it
    manual: bool = False
    #: the last exchange, kept in memory only (never written: it is prompt text)
    last_prompt: Optional[str] = field(default=None, repr=False)
    last_reply: Optional[str] = field(default=None, repr=False)

    SUFFIX = "chat"
    _TRANSIENT = ("last_prompt", "last_reply")

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k not in self._TRANSIENT}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ChatState":
        known = {f for f in cls.__dataclass_fields__ if f not in cls._TRANSIENT}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        save(self.session_id, self.SUFFIX, self.to_json())

    @classmethod
    def load(cls, session_id: str, agent: str) -> "ChatState":
        data = load(session_id, cls.SUFFIX)
        if data and data.get("agent") == agent:
            try:
                return cls.from_json(data)
            except TypeError:
                pass
        return cls(session_id=session_id, agent=agent)

    # --- updating from what a turn showed ------------------------------------------

    def switched(self, model: str, host_model: Optional[str], now: float) -> None:
        self.current_model, self.current_host_model = model, host_model
        self.last_switch_turn, self.last_switch_t = self.turn, now

    def observe_turn(self, *, model: Optional[str], prompts: list[int], outputs: list[int],
                     compacted: bool, window: Optional[int], now: float) -> None:
        """Fold one finished turn into the state. `prompts` and `outputs`
        are its requests' prompt and output tokens, in order."""
        self.turn += 1
        if compacted:
            # a compaction rewrites the history: no model's cached prefix matches it
            self.caches = {}
        if model:
            self.current_model = model
            if window:
                self.windows[model] = int(window)
        if prompts:
            self.context_tokens = int(prompts[-1])
            if model:
                self.caches[model] = {"t": now, "prefix": int(prompts[-1])}
            rounds = len(prompts)
            growth = ((prompts[-1] - prompts[0]) / (rounds - 1)) if rounds > 1 else 0.0
            # each round's prompt grows by the previous output too; growth counts input only
            out_mean = (sum(outputs) / len(outputs)) if outputs else 0.0
            growth = max(0.0, growth - out_mean) if rounds > 1 else 0.0
            self._observe(rounds=float(rounds), growth=growth, output=out_mean)

    def _observe(self, **values: float) -> None:
        turns = int(self.observed.get("turns") or 0)
        for key, value in values.items():
            old = self.observed.get(key)
            self.observed[key] = value if old is None or turns == 0 else (
                (1 - OBSERVED_WEIGHT) * float(old) + OBSERVED_WEIGHT * value)
        self.observed["turns"] = turns + 1


# --- transitions known to work -------------------------------------------------------


class Transitions:
    """Which model transitions are known to work for an agent, kept in
    ``transitions.json``: ``{agent: {"a -> b": {"ok": bool, ...}}}``."""

    def __init__(self, agent: str, path: Optional[Path] = None) -> None:
        self.agent = agent
        self.path = path or (root() / "transitions.json")

    def _all(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def key(a: str, b: str) -> str:
        return f"{a} -> {b}"

    def known(self, a: str, b: str) -> Optional[bool]:
        entry = (self._all().get(self.agent) or {}).get(self.key(a, b))
        return bool(entry.get("ok")) if isinstance(entry, dict) and "ok" in entry else None

    def record(self, a: str, b: str, ok: bool, **detail: Any) -> None:
        data = self._all()
        data.setdefault(self.agent, {})[self.key(a, b)] = {"ok": ok, **detail}
        _write(self.path, data)

    def entries(self) -> dict[str, Any]:
        return dict(self._all().get(self.agent) or {})
