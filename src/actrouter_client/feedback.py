"""Track how coding agents and their users respond to routing.

The router learns from this (:mod:`actrouter_client.learning`). Each turn of a
session leaves a small *work record* (``<session>.work.json`` under
``ACTROUTER_STATE_DIR``): the task type the turn was routed as, the model
routing recommended, the model the session ran on, and each unit of work
started in the turn (a subagent and the model it ran on). When the next
message arrives, :func:`settle` judges that work and records what it shows
in the routing ledger as ``feedback`` events:

``correction``  the user's message says the work was wrong ("that's not
                right", "it still fails", "you broke the build");
``approval``    it says the work was right ("thanks, works", "perfect");
``declined``    the prompt hook asked the agent to hand the work to the
                routed model, and nothing in the turn ran on it.

A model switch the user makes soon after routing (:func:`model_switched`)
records

``accepted``    the user switched to the model routing recommended;
``override``    the user switched to another one (``chosen_model``).

Other signals the learner needs are already ledger facts: a subagent the
agent started on a model of its own choosing (``skipped`` with
``reason: explicit_model``, which now carries the task type and the model
routing would have chosen), each subagent's and chat turn's status, and work
redone on a larger model soon after.

Privacy: the user's message is read here, on this machine, and never
written anywhere. Only the label (``correction``, ``approval``) is recorded,
with the task type and models of the work it judged. ``ACTROUTER_FEEDBACK=off``
turns this tracking off; the ledger setting (``ACTROUTER_LEDGER=off``) also
does.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from actrouter_client import ledger, session_state
from actrouter_client.client import env

OFF = frozenset({"0", "off", "false", "no", "none"})

SUFFIX = "work"

#: a message later than this after the work started is not taken as a reaction to it
REACTION_WINDOW_S = 60 * 60

#: a model switch later than this after routing is not taken as a response to it
SWITCH_WINDOW_S = 30 * 60

#: units of work kept per turn (an agent can start many subagents)
MAX_UNITS = 20

#: optional interjections before the reaction proper ("hmm, that's wrong")
_LEAD = r"^\W*(?:(?:hmm+|ugh|well|ok(?:ay)?|wait|sorry|so|but|nope|no)[,.!:;]?\s+)*"

_CORRECTION = re.compile(_LEAD + r"""(?:
      (?:that'?s|that\s+is|this\s+is|it'?s|it\s+is)\s+(?:still\s+)?
        (?:wrong|incorrect|broken|not\s+(?:right|correct|working|what\s+i\s+(?:asked|wanted|meant)))
    | (?:it|that|this)\s+(?:still\s+)?(?:didn'?t|doesn'?t|does\s+not|did\s+not)\s+work
    | (?:it|that|this|the\s+\w+)\s+(?:still\s+)?(?:fails|failed|is\s+failing|broke)\b
    | still\s+(?:broken|failing|fails|wrong|not\s+working|the\s+same|getting|seeing|an?\s+error)
    | (?:that|you|this)\s+broke\b
    | you\s+(?:missed|forgot|broke|ignored|didn'?t\s+(?:fix|do|read|run|follow))\b
    | (?:try|do\s+it)\s+again\b
    | (?:revert|undo)\s+(?:that|this|it|the|your|those|these)\b
    | (?:wrong|incorrect)\b
    | not\s+(?:quite|right|what\s+i\s+(?:asked|wanted|meant))\b
)""", re.I | re.X)

_APPROVAL = re.compile(r"""^\W*(?:
      thanks?\b | thank\s+you | thx\b | ty\b | cheers\b
    | perfect\b | great\b | awesome\b | excellent\b | nice\b | lgtm\b
    | looks\s+(?:good|great|right|correct)
    | (?:that|it|this)\s+(?:works|worked|fixed\s+it|did\s+it)
    | works\b(?:\s+(?:now|great|perfectly|fine))?
    | (?:that'?s|that\s+is|this\s+is)\s+(?:right|correct|it|perfect|great|exactly)
    | good\s+(?:job|work|catch)
    | well\s+done
)""", re.I | re.X)

#: an approval word that opens a correction ("great, now it's broken") is not one
_BUT = re.compile(r"\b(?:but|still|however|except|though)\b", re.I)


def enabled() -> bool:
    value = env("ACTROUTER_FEEDBACK")
    return not (value and value.lower() in OFF) and ledger.path() is not None


def reaction(message: str) -> Optional[str]:
    """``"correction"``, ``"approval"`` or None for a user's message about
    the work just done. Deliberately narrow: an answer to a proposal ("no,
    use the other one") or a new request is neither."""
    text = message.strip()
    if not text or text.startswith(("/", "<")):
        return None
    head = text[:200]
    if _CORRECTION.match(head):
        return "correction"
    if _APPROVAL.match(head):
        first = re.split(r"[.!?\n]", head, maxsplit=1)[0]
        return None if _BUT.search(first) else "approval"
    return None


def _load(session_id: Any) -> Optional[dict[str, Any]]:
    return session_state.load(session_id, SUFFIX)


def begin(
    session_id: Any,
    *,
    host: str,
    surface: str,
    task_type: Optional[str],
    routed_model: Optional[str],
    routed_host_model: Optional[str] = None,
    session_model: Optional[str] = None,
    request_id: Optional[str] = None,
    delegate: bool = False,
    now: Optional[float] = None,
) -> None:
    """Start the work record of a routed turn. `routed_model` is the catalog
    id routing recommended, `session_model` the model the session itself
    runs on (catalog id when known), `delegate` whether the agent was asked
    to hand the work to the routed model."""
    if not enabled():
        return
    session_state.save(session_id, SUFFIX, {
        "t": time.time() if now is None else now,
        "host": host,
        "surface": surface,
        "task_type": task_type,
        "routed_model": routed_model,
        "routed_host_model": routed_host_model,
        "session_model": session_model,
        "request_id": request_id,
        "delegate": delegate,
        "units": [],
    })


def note(
    session_id: Any,
    *,
    task_type: Optional[str],
    model: Optional[str],
    surface: str,
    request_id: Optional[str] = None,
    now: Optional[float] = None,
) -> None:
    """A unit of work started in the current turn (a subagent), on `model`
    (catalog id). A turn the prompt hook did not route still gets a record,
    so the user's reaction can judge the subagents it started."""
    if not enabled() or not model:
        return
    t = time.time() if now is None else now
    work = _load(session_id) or {"t": t, "units": []}
    units = list(work.get("units") or [])
    units.append({"task_type": task_type, "model": model, "surface": surface,
                  "request_id": request_id, "t": t})
    work["units"] = units[-MAX_UNITS:]
    session_state.save(session_id, SUFFIX, work)


def settle(session_id: Any, message: Optional[str], *, host: str, surface: str,
           now: Optional[float] = None) -> Optional[str]:
    """Judge the previous turn's work by the user's new `message`, record
    what it shows, and forget the work. Returns the reaction, if any."""
    if not enabled():
        return None
    work = _load(session_id)
    if not work:
        return None
    session_state.save(session_id, SUFFIX, None)
    t = time.time() if now is None else now
    units = [u for u in work.get("units") or [] if isinstance(u, dict)]
    common = {"surface": surface, "host": host, "session_id": session_id}

    if work.get("delegate") and work.get("routed_model") and not any(
        u.get("model") == work["routed_model"] for u in units
    ):
        # Nothing ran on the routed model: the agent did the work itself. (A
        # subagent started without a model and routed there by the subagent
        # hook counts as following the delegation.)
        ledger.record("feedback", signal="declined", task_type=work.get("task_type"),
                      model=work["routed_model"], about=work.get("request_id"), **common)

    found = reaction(message or "")
    if found is None or t - float(work.get("t") or 0) > REACTION_WINDOW_S:
        return None
    if units:
        # the units of work the turn started: the subagents did it
        judged = [(u.get("task_type") or work.get("task_type"), u.get("model"), u.get("request_id"))
                  for u in units]
    elif work.get("task_type") and work.get("session_model"):
        # nothing was handed off: the session's own model did the work
        judged = [(work["task_type"], work["session_model"], work.get("request_id"))]
    else:
        judged = []
    for task_type, model, about in dict.fromkeys(judged):
        if task_type and model:
            ledger.record("feedback", signal=found, task_type=task_type, model=model, about=about, **common)
    return found


def model_switched(session_id: Any, to_model: Optional[str], *, host: str, surface: str,
                   to_catalog_id: Optional[str] = None, now: Optional[float] = None) -> Optional[str]:
    """The user switched the session to `to_model` (their spelling). Soon
    after routing, that says whether they agreed with it: ``accepted`` when
    it is the model routing recommended, ``override`` when it is another.
    Returns the signal recorded, if any. Each routing is judged once."""
    if not enabled() or not to_model:
        return None
    work = _load(session_id)
    if not work or not work.get("routed_model") or work.get("switch_judged"):
        return None
    t = time.time() if now is None else now
    if t - float(work.get("t") or 0) > SWITCH_WINDOW_S:
        return None
    wanted = {str(work.get("routed_model")).lower(), str(work.get("routed_host_model")).lower()}
    chosen = str(to_model).strip().lower()
    if chosen in wanted or (to_catalog_id and to_catalog_id.lower() in wanted):
        if str(work.get("session_model") or "").lower() in wanted:
            return None  # already on it: nothing was decided
        signal = "accepted"
    else:
        signal = "override"
    ledger.record(
        "feedback", signal=signal, surface=surface, host=host, session_id=session_id,
        task_type=work.get("task_type"), model=work["routed_model"], about=work.get("request_id"),
        chosen_model=to_catalog_id if signal == "override" else None,
        chosen_host_model=to_model if signal == "override" else None,
    )
    work["switch_judged"] = True
    session_state.save(session_id, SUFFIX, work)
    return signal
