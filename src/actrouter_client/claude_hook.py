"""Claude Code hooks: route the user's prompts and subagents, and record the benefit.

``UserPromptSubmit``: every prompt the user submits is routed. Claude Code
gives hooks no way to switch the session's model, so when the routed model
differs from the one the session runs on, the hook says so to the user
(``/model`` switches) and -- with ``ACTROUTER_PROMPT_ROUTING=delegate``, the
default -- tells Claude to do substantial work on the routed model through a
subagent. A follow-up the router cannot classify on its own words ("yes, go
ahead") stays with the routing of the request it continues: each routing
decision pins the conversation until a prompt routes on its own. With nothing
pinned (a new or resumed session), the follow-up is routed on the user's
previous request and a compressed form of the agent's last reply, from the
transcript (:mod:`actrouter_client.transcript`). Slash commands, and
messages that are only tagged blocks Claude Code delivers as prompts
(``<task-notification>`` when a background task finishes), are left alone
and keep the conversation's pin. Work that lives in the conversation itself
(``chat``, by default) is only advised, never delegated: a subagent cannot
see the conversation. ``SessionStart`` and
``PostModelSwitch`` record the session's model, so the first prompt, and the
first after ``/model``, are compared with the right one.

``PreToolUse``: when Claude Code spawns a subagent through the ``Agent`` tool
without choosing a model, this hook asks ActRouter and fills in ``model``
(``haiku`` for a codebase search, ``sonnet``/``opus`` for debugging, ...). A
model the agent or the user chose explicitly is never touched. Before
routing it works out the *baseline* -- the model the subagent would have run
on (:func:`actrouter_client.baseline.claude_code_subagent`) -- and records
both in the routing ledger (:mod:`actrouter_client.ledger`).

``PostToolUse``: when that subagent finishes, it records the model it really
ran on and its token usage, so ``actrouter-route report`` can price the work
on both models. A subagent in the background returns at launch, without
usage, so ``SubagentStop`` records its usage from its own transcript when it
finishes, under the agent id the launch recorded.

The hook fails open: any error, timeout or unexpected input means no output,
and the subagent runs exactly as it would have without the hook.

What is sent: the Agent call's short ``description``, its ``subagent_type``
mapped to a task hint, and -- only when ``ACTROUTER_HOOK_PROMPT_CHARS`` is
set -- the start of the subagent's prompt. The prompt hook sends the first
``ACTROUTER_HOOK_PROMPT_CHARS`` characters of the user's prompt (default 600)
to the ActRouter API; set it to 0 to keep prompts on the machine, which
turns prompt routing off.

Environment:

``ACTROUTER_SUBAGENT_ROUTING``   ``off`` disables subagent routing (default on)
``ACTROUTER_PROMPT_ROUTING``     delegate (default) | advise | off: what to do
                                  when a prompt fits another model
``ACTROUTER_PROMPT_MIN_CONFIDENCE`` classification confidence below which a
                                  prompt is left alone (default 0.5)
``ACTROUTER_PROMPT_ADVISE_TASKS`` task types only advised, never delegated
                                  (comma-separated; default chat)
``ACTROUTER_HOOK_PROMPT_CHARS``  prompt characters to include (prompt hook:
                                  default 600; subagent hook: default 0)
``ACTROUTER_HOOK_PREFERENCE``    quality | balanced | cost (default balanced)
``ACTROUTER_HOOK_DECISION``      permissionDecision sent with the change:
                                  allow (default) | ask | none
``ACTROUTER_HOOK_TIMEOUT``       seconds for the HTTP call (default 4)
``ACTROUTER_BASELINE_MODEL``     the baseline to record, overriding detection
``ACTROUTER_LEDGER``             ledger file, or ``off`` (default
                                  ~/.actrouter/routing-ledger.jsonl)
``ACTROUTER_STATE_DIR``          where each session's latest prompt routing
                                  is kept (default ~/.actrouter/sessions)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, TextIO

from actrouter_client import baseline as baselines
from actrouter_client import ledger
from actrouter_client import transcript
from actrouter_client.advisor import Advisor
from actrouter_client.client import connect

#: the tool Claude Code spawns subagents with ("Task" in older releases)
AGENT_TOOLS = frozenset({"Agent", "Task"})

#: built-in subagent types whose job is a known task type
SUBAGENT_TASKS = {
    "explore": "code_search",
    "plan": "planning",
    "claude-code-guide": "qa",
    "statusline-setup": "code_generation",
}

OFF = frozenset({"0", "off", "false", "no"})

#: ACTROUTER_PROMPT_ROUTING modes
PROMPT_MODES = frozenset({"delegate", "advise"})

#: prompt characters the prompt hook routes on
PROMPT_HOOK_CHARS = 600

#: UserPromptSubmit sources that are a person's request (absent in older releases)
PROMPT_SOURCES = frozenset({"user", "sdk"})

#: one tagged block, as Claude Code delivers notifications and command output
#: in the prompt (``<task-notification>...</task-notification>``)
_TAGGED_BLOCK = re.compile(r"<([A-Za-z][\w-]*)(?:\s[^>]*)?>.*?</\1>|<[A-Za-z][\w-]*(?:\s[^>]*)?/>", re.S)

#: task types that live in the conversation, so delegating them cannot work
ADVISE_TASKS = "chat"


def _is_harness_message(prompt: str) -> bool:
    """True when `prompt` is only tagged blocks: a background task's
    completion notice and the like, not the user's words."""
    return prompt.startswith("<") and not _TAGGED_BLOCK.sub("", prompt).strip()

#: Claude Code passes a plugin's userConfig to its hooks as
#: CLAUDE_PLUGIN_OPTION_<KEY>; explicit ACTROUTER_* settings win
PLUGIN_OPTIONS = {
    "CLAUDE_PLUGIN_OPTION_API_KEY": "ACTROUTER_API_KEY",
    "CLAUDE_PLUGIN_OPTION_ROUTER_URL": "ACTROUTER_URL",
    "CLAUDE_PLUGIN_OPTION_SUBAGENT_ROUTING": "ACTROUTER_SUBAGENT_ROUTING",
    "CLAUDE_PLUGIN_OPTION_PROMPT_ROUTING": "ACTROUTER_PROMPT_ROUTING",
    "CLAUDE_PLUGIN_OPTION_PREFERENCE": "ACTROUTER_HOOK_PREFERENCE",
    "CLAUDE_PLUGIN_OPTION_BASELINE_MODEL": "ACTROUTER_BASELINE_MODEL",
    "CLAUDE_PLUGIN_OPTION_LEDGER": "ACTROUTER_LEDGER",
}


def _apply_plugin_options() -> None:
    for option, setting in PLUGIN_OPTIONS.items():
        value = os.environ.get(option, "").strip()
        if value and not os.environ.get(setting):
            os.environ[setting] = value


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _advisor(advisor: Optional[Advisor]) -> Advisor:
    if advisor is not None:
        return advisor
    timeout = float(os.environ.get("ACTROUTER_HOOK_TIMEOUT", "4"))
    return Advisor(connect(timeout=timeout), host="claude-code")


# --- per-session state ----------------------------------------------------------------
# The session's model, and its latest prompt routing, kept so the PreToolUse
# hook can tell a subagent Claude started on the routed model and record it as
# routed work. Holds models and decisions only, never the prompt.


def _state_file(session_id: Any, suffix: str) -> Optional[Path]:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id or ""))
    if not name.strip("._"):
        return None
    root = os.environ.get("ACTROUTER_STATE_DIR", "").strip()
    return (Path(root).expanduser() if root else Path.home() / ".actrouter" / "sessions") / f"{name}.{suffix}.json"


def _save_state(session_id: Any, suffix: str, value: Optional[dict[str, Any]]) -> None:
    target = _state_file(session_id, suffix)
    if target is None:
        return
    try:
        if value is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value), encoding="utf-8")
    except OSError as exc:
        print(f"actrouter hook: could not keep session state: {exc}", file=sys.stderr)


def _load_state(session_id: Any, suffix: str) -> Optional[dict[str, Any]]:
    target = _state_file(session_id, suffix)
    try:
        value = json.loads(target.read_text(encoding="utf-8")) if target else None
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _save_prompt_route(session_id: Any, route: Optional[dict[str, Any]]) -> None:
    _save_state(session_id, "prompt", route)


def _prompt_route(session_id: Any) -> Optional[dict[str, Any]]:
    return _load_state(session_id, "prompt")


def record_session_model(event: dict[str, Any]) -> None:
    """``SessionStart`` (``model``) or ``PostModelSwitch`` (``to_model``,
    null for the default model): the model the session now runs on."""
    if event.get("agent_id"):
        return
    model = event.get("to_model") if event.get("hook_event_name") == "PostModelSwitch" else event.get("model")
    _save_state(event.get("session_id"), "model", {"model": str(model)} if model else None)


def route_prompt(event: dict[str, Any], advisor: Optional[Advisor] = None) -> Optional[dict[str, Any]]:
    """The hook's JSON answer for one UserPromptSubmit `event`, or None to
    leave the prompt alone."""
    mode = os.environ.get("ACTROUTER_PROMPT_ROUTING", "delegate").strip().lower()
    if mode not in PROMPT_MODES or event.get("agent_id"):
        return None
    session_id = event.get("session_id")
    prompt = str(event.get("prompt") or "").strip()
    source = event.get("source")
    if not prompt or prompt.startswith("/") or (source and source not in PROMPT_SOURCES):
        return None
    if _is_harness_message(prompt):
        return None  # and the conversation keeps its pin: the user asked for nothing

    advisor = _advisor(advisor)
    prompt_chars = _env_int("ACTROUTER_HOOK_PROMPT_CHARS", PROMPT_HOOK_CHARS)
    if prompt_chars <= 0:
        return None

    known = (_load_state(session_id, "model") or {}).get("model")
    current, current_source = baselines.claude_code_session(event, known)
    baseline, baseline_source = baselines.override() or (current, current_source)
    chars = min(prompt_chars, 2000)
    min_confidence = _env_float("ACTROUTER_PROMPT_MIN_CONFIDENCE", 0.5)

    def route(text: str) -> Optional[dict[str, Any]]:
        advice = advisor.for_task(
            text,
            preference=os.environ.get("ACTROUTER_HOOK_PREFERENCE", "balanced"),
            top_k=1,
            baseline=baseline,
        )
        if (not advice.get("host_model") or advice.get("task_type") in (None, "unknown")
                or (advice.get("task_confidence") or 0.0) < min_confidence):
            return None
        return advice

    routed_on = "prompt"
    advice = route(prompt[:chars])
    pinned = _prompt_route(session_id) if advice is None else None
    if pinned:
        # A follow-up ("yes, go ahead") stays on the model the conversation
        # was routed to, as a router pinning the conversation would.
        advice, routed_on = dict(pinned), "pinned"
        if pinned.get("baseline") != baseline:  # /model switched since: the estimate is stale
            for key in ("baseline_estimated_cost_usd", "estimated_savings_usd"):
                advice.pop(key, None)
            advice["baseline_model"] = advisor.baseline_id(baseline)
    elif advice is None:
        # A follow-up with nothing routed before it (a new session, a resumed
        # one): route the work it continues -- what the agent last proposed,
        # and the request that led to it.
        previous, reply = transcript.last_exchange(event.get("transcript_path"), prompt)
        context = " ".join(p for p in (
            prompt[: chars // 4],
            transcript.compress(reply, chars // 2),
            transcript.compress(previous, chars // 4),
        ) if p)
        if not (previous or reply) or not context:
            return None
        advice = route(context)
        routed_on = "previous exchange"
        if advice is None:
            return None
    model = advice["host_model"]
    stays = bool(current) and advisor.baseline_id(current) == advice["model"]
    # Pin the conversation to this routing, also when it keeps the session's
    # model, so follow-ups stay with it until a prompt routes on its own.
    _save_prompt_route(session_id, {
        **{k: advice.get(k) for k in ledger.ROUTE_FIELDS},
        "baseline": baseline,
        "baseline_source": baseline_source,
        "stays": stays,
    })
    if stays:
        return None  # the session already runs on the model that fits

    # work in the conversation itself: tell the user, do not hand it to a subagent
    advise_only = {t.strip() for t in os.environ.get("ACTROUTER_PROMPT_ADVISE_TASKS", ADVISE_TASKS).split(",")}
    if mode == "delegate" and advice.get("task_type") in advise_only:
        mode = "advise"

    ledger.record(
        "prompt", surface="prompt_hook", host="claude-code", session_id=session_id, mode=mode,
        routed_on=routed_on,
        task_type=advice.get("task_type"), task_confidence=advice.get("task_confidence"),
        routed_model=advice["model"], routed_host_model=model,
        baseline_model=advice.get("baseline_model"), baseline_as_given=baseline,
        baseline_source=baseline_source,
    )

    task = str(advice.get("task_type")).replace("_", " ")
    follow_up = routed_on != "prompt"
    how = ("stays on the model the conversation was routed to" if routed_on == "pinned"
           else "routed on the work it continues")
    runs_on = f"this session runs on {current}" if current else "this session's model is unknown"
    output: dict[str, Any] = {
        "systemMessage": f"ActRouter: {task} -> {model} ({runs_on}"
        + (f"; a follow-up, {how})" if follow_up else ")")
        + ("; delegating substantial work to it." if mode == "delegate"
           else f". `/model {model}` switches the session."),
    }
    if mode == "delegate":
        same = "" if current else f" If you already run on {model}, ignore this."
        output["hookSpecificOutput"] = {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                f"ActRouter routed this request as {task} work"
                + (" (a follow-up: it stays with the routing of the request it continues)"
                   if routed_on == "pinned" else
                   " (it continues the previous exchange, so it was routed on what you "
                   "proposed and the request before it)" if follow_up else "")
                + f". The model that fits it is "
                f"{model} ({advice['model']}); {runs_on}.{same} Do not do this work on your "
                f"own model: start a subagent with the Agent tool and model: \"{model}\", give "
                f"it the request and the context it needs, and relay its result. Answer "
                f"directly only when that needs no tool calls (a reply from what is already in "
                f"the conversation), when the request is about this conversation or your own "
                f"work in it (a subagent cannot see the conversation, so it would start cold), "
                f"when the user named a model, or when the routing is clearly wrong for the "
                f"request."
            ),
        }
    return output


def decide(event: dict[str, Any], advisor: Optional[Advisor] = None) -> Optional[dict[str, Any]]:
    """The hook's JSON answer for one PreToolUse `event`, or None to leave
    the call alone."""
    if os.environ.get("ACTROUTER_SUBAGENT_ROUTING", "on").strip().lower() in OFF:
        return None
    if event.get("tool_name") not in AGENT_TOOLS:
        return None
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    context = {
        "session_id": event.get("session_id"),
        "tool_use_id": event.get("tool_use_id"),
        "subagent_type": str(tool_input.get("subagent_type") or "") or None,
    }
    if tool_input.get("model"):
        model = str(tool_input["model"])
        route = _prompt_route(event.get("session_id"))
        if route and not route.get("stays") and model.lower() in (
            str(route.get("host_model")).lower(), str(route.get("model")).lower()
        ):
            # Claude handed the user's request to the model it was routed to
            ledger.route_event(route, surface="prompt_hook", baseline=route.get("baseline"),
                               baseline_source=str(route.get("baseline_source")), **context)
        elif offered := ledger.claim(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd"), model):
            # Claude asked route_task and started the subagent on its answer
            ledger.route_event(offered, surface="mcp", baseline=offered.get("baseline"),
                               baseline_source=str(offered.get("baseline_source")),
                               claims=offered.get("request_id"), **context)
        else:
            ledger.record("skipped", surface="claude_hook", host="claude-code",
                          reason="explicit_model", model=model, **context)
        return None
    description = str(tool_input.get("description") or "").strip()
    prompt = str(tool_input.get("prompt") or "").strip()
    subagent = str(tool_input.get("subagent_type") or "").strip()
    if not description and not prompt:
        return None

    advisor = _advisor(advisor)
    prompt_chars = _env_int("ACTROUTER_HOOK_PROMPT_CHARS", 0)
    text = description or prompt[:120]
    if prompt_chars > 0 and prompt:
        text = f"{text}. {prompt[:prompt_chars]}"

    baseline, baseline_source = baselines.claude_code_subagent(event)
    advice = advisor.for_task(
        text[:2000],
        title=description[:120] or None,
        task_kind=SUBAGENT_TASKS.get(subagent.lower()),
        preference=os.environ.get("ACTROUTER_HOOK_PREFERENCE", "balanced"),
        top_k=1,
        baseline=baseline,
    )
    model = advice.get("host_model")
    if not model:
        return None
    ledger.route_event(
        advice, surface="claude_hook", baseline=baseline, baseline_source=baseline_source, **context
    )

    output: dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "updatedInput": {**tool_input, "model": model},
    }
    decision = os.environ.get("ACTROUTER_HOOK_DECISION", "allow").strip().lower()
    if decision in ("allow", "ask"):
        output["permissionDecision"] = decision
        output["permissionDecisionReason"] = (
            f"ActRouter: {advice.get('task_type')} -> {advice['model']}"
        )
    return {"hookSpecificOutput": output}


def _agent_id(value: Any) -> Optional[str]:
    """An agent id as Claude Code spells it in the Agent result (``a1b2``)
    and in hook events (sometimes ``agent-a1b2``)."""
    text = str(value or "").strip()
    return text.removeprefix("agent-") or None


def record_outcome(event: dict[str, Any]) -> None:
    """``PostToolUse`` on a subagent: what it really ran on, and its usage.

    Claude Code's Agent tool result carries ``resolvedModel``, ``usage``
    (input, output and cache tokens), ``totalTokens`` and
    ``totalDurationMs``. A subagent started in the background reports only
    ``async_launched``; that is recorded too, without usage.
    """
    if os.environ.get("ACTROUTER_SUBAGENT_ROUTING", "on").strip().lower() in OFF:
        return
    if event.get("tool_name") not in AGENT_TOOLS:
        return
    result = event.get("tool_response")
    if not isinstance(result, dict):
        return
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    tool_input = event.get("tool_input") if isinstance(event.get("tool_input"), dict) else {}
    ledger.record(
        "outcome",
        surface="claude_hook",
        host="claude-code",
        session_id=event.get("session_id"),
        tool_use_id=event.get("tool_use_id"),
        agent_id=_agent_id(result.get("agentId")),
        subagent_type=str(tool_input.get("subagent_type") or "") or None,
        requested_model=tool_input.get("model"),
        status=result.get("status"),
        resolved_model=result.get("resolvedModel"),
        models_used=result.get("modelsUsed"),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_input_tokens=usage.get("cache_read_input_tokens"),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
        total_tokens=result.get("totalTokens"),
        duration_ms=result.get("totalDurationMs"),
    )


def record_subagent_stop(event: dict[str, Any]) -> None:
    """``SubagentStop``: the subagent's real usage, read from its own
    transcript (model and usage fields only). This is the only record of a
    background subagent's usage; for a foreground one the report keeps the
    ``PostToolUse`` outcome. A missing transcript (Claude Code can report a
    stop for an agent that never ran) records nothing."""
    if os.environ.get("ACTROUTER_SUBAGENT_ROUTING", "on").strip().lower() in OFF:
        return
    agent_id = _agent_id(event.get("agent_id"))
    path = str(event.get("agent_transcript_path") or "")
    if not agent_id or not path:
        return
    usage = transcript.subagent_usage(str(Path(path).expanduser()))
    if usage is None:
        return
    ledger.record(
        "outcome",
        surface="subagent_stop",
        host="claude-code",
        session_id=event.get("session_id"),
        agent_id=agent_id,
        subagent_type=str(event.get("agent_type") or "") or None,
        status="completed",
        **usage,
    )


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    try:
        _apply_plugin_options()
        event = json.load(stdin)
        if isinstance(event, dict) and event.get("hook_event_name") == "PostToolUse":
            record_outcome(event)
            return 0
        if isinstance(event, dict) and event.get("hook_event_name") == "SubagentStop":
            record_subagent_stop(event)  # never answers: an answer could keep the subagent running
            return 0
        if isinstance(event, dict) and event.get("hook_event_name") in ("SessionStart", "PostModelSwitch"):
            record_session_model(event)
            return 0
        if isinstance(event, dict) and event.get("hook_event_name") == "UserPromptSubmit":
            answer = route_prompt(event)
        else:
            answer = decide(event) if isinstance(event, dict) else None
    except Exception as exc:  # fail open: never block or alter a prompt or subagent on error
        print(f"actrouter hook: {exc}", file=sys.stderr)
        return 0
    if answer is not None:
        json.dump(answer, stdout)
        stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
