import io
import json

import pytest

from actrouter_client import Advisor
from actrouter_client import transcript as conversations
from actrouter_client.claude_hook import decide, main, route_prompt


def event(**tool_input) -> dict:
    return {"hook_event_name": "PreToolUse", "tool_name": "Agent", "tool_input": tool_input}


def test_fills_in_the_model(claude: Advisor) -> None:
    ev = event(description="Find config loader", prompt="Find where config is loaded", subagent_type="Explore")
    out = decide(ev, claude)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "allow"
    assert out["updatedInput"] == {**ev["tool_input"], "model": "haiku"}
    assert "code_search" in out["permissionDecisionReason"]


def test_debugging_goes_to_a_bigger_model(claude: Advisor) -> None:
    out = decide(event(description="Debug flaky test", prompt="The integration test fails 1 in 5 runs"), claude)
    assert out["hookSpecificOutput"]["updatedInput"]["model"] in {"sonnet", "opus"}


@pytest.mark.parametrize(
    "ev",
    [
        event(description="x", prompt="y", model="opus"),  # explicit choice
        {"tool_name": "Bash", "tool_input": {"command": "ls"}},
        {"tool_name": "Agent", "tool_input": {}},
        {"tool_name": "Agent", "tool_input": "nonsense"},
    ],
)
def test_leaves_other_calls_alone(ev: dict, claude: Advisor) -> None:
    assert decide(ev, claude) is None


def test_can_be_switched_off(claude: Advisor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_SUBAGENT_ROUTING", "off")
    assert decide(event(description="Find x", prompt="y"), claude) is None


def test_decision_none_omits_permission(claude: Advisor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_HOOK_DECISION", "none")
    out = decide(event(description="Find x", prompt="y"), claude)["hookSpecificOutput"]
    assert "permissionDecision" not in out
    assert out["updatedInput"]["model"]


def test_prompt_is_not_sent_over_http_by_default() -> None:
    sent = {}

    class Spy:
        where = "https://router.example"

        def list_models(self):
            return [{"id": "claude-haiku-4-5", "family": "anthropic"}]

        def route_task(self, task, **kw):
            sent.update(task)
            return {"chosen_model": "claude-haiku-4-5", "ranking": []}

    decide(event(description="Fix auth", prompt="password=hunter2 fails"), Advisor(Spy(), host="claude-code"))
    assert "hunter2" not in json.dumps(sent)
    assert sent["description"] == "Fix auth"


def test_main_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    assert main(io.StringIO("not json"), out) == 0
    assert out.getvalue() == ""

    monkeypatch.setenv("ACTROUTER_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ACTROUTER_HOOK_TIMEOUT", "1")
    out = io.StringIO()
    assert main(io.StringIO(json.dumps(event(description="Find x", prompt="y"))), out) == 0
    assert out.getvalue() == ""


def test_main_writes_json(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    main(io.StringIO(json.dumps(event(description="Search the repo for usages of foo", prompt="."))), out)
    assert json.loads(out.getvalue())["hookSpecificOutput"]["updatedInput"]["model"] == "haiku"


def test_plugin_options_configure_the_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SUBAGENT_ROUTING", "off")
    out = io.StringIO()
    main(io.StringIO(json.dumps(event(description="Search the repo for usages of foo", prompt="."))), out)
    assert out.getvalue() == ""


# --- UserPromptSubmit: routing the user's own request -------------------------------


def prompt_event(prompt: str, transcript=None, **extra) -> dict:
    return {"hook_event_name": "UserPromptSubmit", "session_id": "s1", "prompt": prompt,
            "transcript_path": str(transcript) if transcript else None, **extra}


def transcript(tmp_path, model: str) -> str:
    """A session transcript whose newest assistant message ran on `model`."""
    path = tmp_path / f"{model}.jsonl"
    path.write_text(json.dumps({"type": "assistant", "message": {"model": model}}) + "\n")
    return str(path)


@pytest.fixture()
def on_opus(tmp_path, claude: Advisor) -> str:
    return transcript(tmp_path, claude.baseline_id("opus"))


def ledger_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_prompt_that_fits_a_smaller_model_is_delegated(claude: Advisor, on_opus: str, ledger_file) -> None:
    out = route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    context = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert 'model: "haiku"' in context and claude.baseline_id("opus") in context
    assert "haiku" in out["systemMessage"]
    [prompt] = ledger_events(ledger_file)
    assert prompt["kind"] == "prompt" and prompt["routed_host_model"] == "haiku"
    assert "retry" not in json.dumps(prompt)  # the decision is recorded, never the prompt


def test_advise_mode_only_tells_the_user(claude: Advisor, on_opus: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_PROMPT_ROUTING", "advise")
    out = route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    assert "hookSpecificOutput" not in out
    assert "/model haiku" in out["systemMessage"]


def test_prompt_already_on_the_right_model_is_left_alone(claude: Advisor, tmp_path) -> None:
    on_haiku = transcript(tmp_path, claude.baseline_id("haiku"))
    assert route_prompt(prompt_event("where is the retry policy configured?", on_haiku), claude) is None


@pytest.mark.parametrize(
    "ev",
    [
        prompt_event("/model opus"),  # a slash command
        prompt_event(""),
        prompt_event("where is the retry policy configured?", source="loop_wakeup"),
        prompt_event("where is the retry policy configured?", agent_id="a1"),
    ],
)
def test_prompts_left_alone(ev: dict, claude: Advisor, on_opus: str) -> None:
    ev["transcript_path"] = on_opus
    assert route_prompt(ev, claude) is None


def test_unclassified_prompt_asks_claude_to_route_it(claude: Advisor, on_opus: str, ledger_file) -> None:
    # cannot be classified, and continues nothing: Claude routes it once it knows the work
    out = route_prompt(prompt_event("yes, go ahead", on_opus), claude)
    assert "systemMessage" not in out  # nothing was routed, so nothing to tell the user
    context = out["hookSpecificOutput"]["additionalContext"]
    assert "route_task" in context and "ToolSearch" in context
    [unrouted] = ledger_events(ledger_file)
    assert unrouted["kind"] == "unrouted" and unrouted["mode"] == "delegate"
    assert "go ahead" not in json.dumps(unrouted)  # the decision is recorded, never the prompt


def test_unclassified_prompt_is_only_recorded_when_advising(claude: Advisor, on_opus: str, ledger_file,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_PROMPT_ROUTING", "advise")
    assert route_prompt(prompt_event("yes, go ahead", on_opus), claude) is None
    assert [e["kind"] for e in ledger_events(ledger_file)] == ["unrouted"]


NOTIFICATIONS = [
    "<task-notification>\n<task-id>b1</task-id>\n<status>completed</status>\n"
    "<summary>Background command \"pytest -q\" completed</summary>\n</task-notification>",
    "<task-notification><task-id>a9</task-id><status>completed</status><summary>Agent \"Test authoring\" "
    "completed</summary><result>Wrote 94 tests</result><usage>total_tokens: 113507</usage></task-notification>",
    "<task-notification><task-id>b2</task-id></task-notification>\n<system-reminder>x</system-reminder>",
]


@pytest.mark.parametrize("notification", NOTIFICATIONS)
def test_notifications_are_not_routed_and_keep_the_pin(claude: Advisor, on_opus: str, ledger_file,
                                                       notification: str) -> None:
    from actrouter_client.claude_hook import _prompt_route

    route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    pinned = _prompt_route("s1")
    # these used to route as summarization -> haiku, delegate, and repin the conversation
    assert route_prompt(prompt_event(notification, on_opus), claude) is None
    assert _prompt_route("s1") == pinned
    assert [e["kind"] for e in ledger_events(ledger_file)] == ["prompt"]


def test_the_users_words_around_a_tag_are_still_routed(claude: Advisor, on_opus: str) -> None:
    prompt = "<ide_selection>http/client.py</ide_selection> where is the retry policy configured?"
    assert route_prompt(prompt_event(prompt, on_opus), claude) is not None


def test_chat_is_advised_not_delegated(claude: Advisor, on_opus: str, ledger_file) -> None:
    # a subagent cannot answer a question about the conversation it cannot see
    out = route_prompt(prompt_event("do you have access to actrouter?", on_opus), claude)
    assert out is not None and "hookSpecificOutput" not in out
    assert "/model haiku" in out["systemMessage"]
    assert ledger_events(ledger_file)[-1]["mode"] == "advise"


def test_advise_only_task_types_are_configurable(claude: Advisor, on_opus: str, monkeypatch) -> None:
    monkeypatch.setenv("ACTROUTER_PROMPT_ADVISE_TASKS", "code_search")
    out = route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    assert "hookSpecificOutput" not in out
    monkeypatch.setenv("ACTROUTER_PROMPT_ADVISE_TASKS", "")
    out = route_prompt(prompt_event("do you have access to actrouter?", on_opus), claude)
    assert "hookSpecificOutput" in out


def test_delegation_leaves_work_about_the_conversation_to_claude(claude: Advisor, on_opus: str) -> None:
    out = route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    assert "about this conversation or your own work in it" in out["hookSpecificOutput"]["additionalContext"]


def test_prompt_routing_can_be_switched_off(claude: Advisor, on_opus: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_PROMPT_ROUTING", "off")
    out = io.StringIO()
    main(io.StringIO(json.dumps(prompt_event("where is the retry policy configured?", on_opus))), out)
    assert out.getvalue() == ""


class RemoteRouter:
    """An HTTP router that routes everything to haiku and keeps what it was sent."""

    where = "https://router.example"

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def list_models(self):
        return [{"id": "claude-haiku-4-5", "family": "anthropic"}]

    def route_task(self, task, **kw):
        self.sent.append(task)
        return {"chosen_model": "claude-haiku-4-5", "ranking": [],
                "classification": {"task_type": "code_search", "confidence": 0.9}}


def test_prompt_is_routed_over_http(claude: Advisor, on_opus: str) -> None:
    remote = RemoteRouter()
    out = route_prompt(prompt_event("where is the retry policy configured? " + "x" * 900, on_opus),
                       Advisor(remote, host="claude-code"))
    assert 'model: "haiku"' in out["hookSpecificOutput"]["additionalContext"]
    [sent] = remote.sent
    assert sent["description"].startswith("where is the retry policy configured?")
    assert len(sent["description"]) <= 600


def test_prompt_stays_local_when_prompt_chars_is_zero(on_opus: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_HOOK_PROMPT_CHARS", "0")
    remote = RemoteRouter()
    assert route_prompt(prompt_event("password=hunter2 fails", on_opus), Advisor(remote, host="claude-code")) is None
    assert remote.sent == []


def test_delegated_subagent_is_recorded_as_routed(claude: Advisor, on_opus: str, ledger_file) -> None:
    route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    ev = {**event(description="Find retry policy", prompt="...", model="haiku"),
          "session_id": "s1", "tool_use_id": "t1"}
    assert decide(ev, claude) is None  # an explicit model is still never changed
    route = ledger_events(ledger_file)[-1]
    assert route["kind"] == "route" and route["surface"] == "prompt_hook"
    assert route["routed_model"] == claude.baseline_id("haiku")
    assert route["baseline_model"] == claude.baseline_id("opus")
    assert route["tool_use_id"] == "t1"

    # a follow-up stays pinned to that routing, so its subagents still count
    route_prompt(prompt_event("yes, go ahead", on_opus), claude)
    decide({**ev, "tool_use_id": "t2"}, claude)
    assert ledger_events(ledger_file)[-1]["surface"] == "prompt_hook"


def test_new_request_on_the_sessions_model_repins(claude: Advisor, tmp_path, ledger_file) -> None:
    debug = "The auth integration test fails 1 in 5 runs, figure out why and fix it"
    fits = claude.for_task(debug, top_k=1)["model"]
    on_it = transcript(tmp_path, fits)
    assert route_prompt(prompt_event("where is the retry policy configured?", on_it), claude)
    assert route_prompt(prompt_event(debug, on_it), claude) is None  # already on the model that fits
    assert route_prompt(prompt_event("yes, go ahead", on_it), claude) is None  # pinned to staying
    ev = {**event(description="Fix it", prompt="...", model="haiku"), "session_id": "s1", "tool_use_id": "t3"}
    decide(ev, claude)
    assert ledger_events(ledger_file)[-1]["kind"] == "skipped"  # not a delegation


def test_follow_up_stays_pinned_rather_than_rerouting(claude: Advisor, tmp_path, ledger_file) -> None:
    on_opus = transcript(tmp_path, claude.baseline_id("opus"))
    route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    path = conversation(
        tmp_path, "where is the retry policy configured?",
        "It is in `http/client.py`. Want me to write release notes for the last 10 commits as well?",
        claude.baseline_id("opus"),
    )
    out = route_prompt(prompt_event("sure", path), claude)
    assert "stays on the model the conversation was routed to" in out["systemMessage"]
    assert "code search" in out["hookSpecificOutput"]["additionalContext"]
    assert ledger_events(ledger_file)[-1]["routed_on"] == "pinned"


def test_main_routes_prompts(on_opus: str, monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    main(io.StringIO(json.dumps(prompt_event("where is the retry policy configured?", on_opus))), out)
    assert 'model: "haiku"' in json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]


def test_session_model_comes_from_session_start_and_model_switches(claude: Advisor) -> None:
    opus, haiku = claude.baseline_id("opus"), claude.baseline_id("haiku")
    ask = prompt_event("where is the retry policy configured?")  # no transcript yet: the first prompt
    assert "model is unknown" in route_prompt(ask, claude)["systemMessage"]

    main(io.StringIO(json.dumps({"hook_event_name": "SessionStart", "session_id": "s1", "model": opus})), io.StringIO())
    assert f"runs on {opus}" in route_prompt(ask, claude)["systemMessage"]

    switch = {"hook_event_name": "PostModelSwitch", "session_id": "s1", "to_model": haiku}
    main(io.StringIO(json.dumps(switch)), io.StringIO())
    assert route_prompt(ask, claude) is None  # now on the model that fits


def test_baseline_override_is_not_taken_for_the_session_model(
    claude: Advisor, tmp_path, monkeypatch: pytest.MonkeyPatch, ledger_file
) -> None:
    monkeypatch.setenv("ACTROUTER_BASELINE_MODEL", "opus")
    on_haiku = transcript(tmp_path, claude.baseline_id("haiku"))
    assert route_prompt(prompt_event("where is the retry policy configured?", on_haiku), claude) is None
    on_sonnet = transcript(tmp_path, claude.baseline_id("sonnet"))
    out = route_prompt(prompt_event("where is the retry policy configured?", on_sonnet), claude)
    assert claude.baseline_id("sonnet") in out["systemMessage"]
    assert ledger_events(ledger_file)[-1]["baseline_as_given"] == "opus"


# --- follow-ups: routed on the exchange they continue -------------------------------


def conversation(tmp_path, previous: str, reply: str, model: str) -> str:
    rows = [
        {"type": "user", "message": {"role": "user", "content": previous}},
        {"type": "assistant", "message": {"model": model, "content": [{"type": "tool_use", "name": "Bash"}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "out"}]}},
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "<system-reminder>x"}},
        {"type": "user", "message": {"role": "user", "content": "<local-command-stdout>ok</local-command-stdout>"}},
        {"type": "assistant", "message": {"model": model, "content": [{"type": "text", "text": reply}]}},
    ]
    path = tmp_path / "conversation.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_follow_up_is_routed_on_the_agents_proposal(claude: Advisor, tmp_path, ledger_file) -> None:
    path = conversation(
        tmp_path, "where is the retry policy configured?",
        "It is set in `http/client.py`. Do you want me to search the repo for every caller that overrides it?",
        claude.baseline_id("opus"),
    )
    out = route_prompt(prompt_event("yes please", path), claude)
    assert "follow-up" in out["systemMessage"]
    assert 'model: "haiku"' in out["hookSpecificOutput"]["additionalContext"]
    assert ledger_events(ledger_file)[-1]["routed_on"] == "previous exchange"


def test_unpinned_follow_up_routes_what_the_agent_proposed(claude: Advisor, tmp_path) -> None:
    path = conversation(
        tmp_path, "explain what this repo does",
        "It is a model router. Want me to write release notes for the last 10 commits as well?",
        claude.baseline_id("opus"),
    )
    out = route_prompt(prompt_event("sure", path), claude)
    assert "documentation" in out["hookSpecificOutput"]["additionalContext"]


def test_a_new_request_the_router_cannot_read_does_not_inherit_the_pin(claude: Advisor, on_opus: str,
                                                                      ledger_file) -> None:
    from actrouter_client.claude_hook import _prompt_route

    route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    pinned = _prompt_route("s1")
    # used to be delegated as the pinned code search -> haiku
    out = route_prompt(prompt_event("Commit the working tree changes with a sensible message.", on_opus), claude)
    assert "systemMessage" not in out and "route_task" in out["hookSpecificOutput"]["additionalContext"]
    assert ledger_events(ledger_file)[-1]["kind"] == "unrouted"
    assert _prompt_route("s1") == pinned  # a real follow-up still finds it


@pytest.mark.parametrize("question", ["did you use actrouter in the work?", "Why did you skip the tests?",
                                      "have you committed it yet?"])
def test_questions_about_the_agents_work_are_not_routed(claude: Advisor, on_opus: str, ledger_file,
                                                        question: str) -> None:
    from actrouter_client.claude_hook import _prompt_route

    route_prompt(prompt_event("where is the retry policy configured?", on_opus), claude)
    pinned = _prompt_route("s1")
    assert route_prompt(prompt_event(question, on_opus), claude) is None
    assert ledger_events(ledger_file)[-1]["kind"] == "conversation"
    assert _prompt_route("s1") == pinned


def test_fork_subagents_are_recorded_not_routed(claude: Advisor, ledger_file) -> None:
    # a fork runs on its parent's model whatever `model` says
    ev = {**event(description="Build the web UI", prompt="Build the UI in frontend/", subagent_type="fork"),
          "session_id": "s1", "tool_use_id": "t9"}
    assert decide(ev, claude) is None
    skipped = ledger_events(ledger_file)[-1]
    assert skipped["kind"] == "skipped" and skipped["reason"] == "inherits_model"


def test_a_subagent_the_router_can_only_guess_at_keeps_its_model(claude: Advisor, ledger_file) -> None:
    ev = {**event(description="Handle it", prompt="Take care of the remaining items"),
          "session_id": "s1", "tool_use_id": "t10"}
    assert decide(ev, claude) is None
    skipped = ledger_events(ledger_file)[-1]
    assert skipped["kind"] == "skipped" and skipped["reason"] == "low_confidence"


def test_small_talk_follow_up_is_not_routed(claude: Advisor, tmp_path) -> None:
    path = conversation(tmp_path, "hi", "Hello! How can I help?", claude.baseline_id("opus"))
    out = route_prompt(prompt_event("thanks", path), claude)
    assert "systemMessage" not in out  # no model is named or delegated to
    assert "route_task" in out["hookSpecificOutput"]["additionalContext"]


def test_last_exchange_skips_tool_results_tags_and_the_current_prompt(tmp_path) -> None:
    path = conversation(tmp_path, "fix the flaky test", "Found it. Shall I fix it?", "m")
    with open(path, "a") as f:  # Claude Code may already have written the prompt being routed
        f.write(json.dumps({"type": "user", "message": {"role": "user", "content": "go ahead"}}) + "\n")
    assert conversations.last_exchange(path, "go ahead") == ("fix the flaky test", "Found it. Shall I fix it?")
    assert conversations.last_exchange(None) == (None, None)
    assert conversations.last_exchange(str(tmp_path / "missing.jsonl")) == (None, None)


def test_compress_drops_code_and_keeps_the_start_and_the_end() -> None:
    reply = "I found the race. " + "```python\nsecret = 1\n```\n" + "Detail. " * 100 + "Should I fix the cache reset?"
    short = conversations.compress(reply, 120)
    assert len(short) <= 120
    assert short.startswith("I found the race.") and short.endswith("Should I fix the cache reset?")
    assert "secret" not in short
    assert conversations.compress("Short reply.", 120) == "Short reply."
    assert conversations.compress(None, 120) == ""


def test_plugin_api_key_reaches_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from actrouter_client import connect
    from actrouter_client.claude_hook import _apply_plugin_options

    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_KEY", "ar_live_test")
    _apply_plugin_options()
    assert connect().headers["x-api-key"] == "ar_live_test"


def test_used_up_plan_is_told_to_the_user_once(
    on_opus: str, http_stub, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detail = "ActRouter free tier used up: 30 of 30 routing calls this month. Upgrade: run `actrouter-route upgrade`."
    monkeypatch.setenv("ACTROUTER_URL", http_stub(lambda m, p, b: (402, {"detail": detail})))
    monkeypatch.setenv("ACTROUTER_STATE_DIR", str(tmp_path / "state"))
    ev = prompt_event("where is the retry policy configured?", on_opus)
    out = io.StringIO()
    assert main(io.StringIO(json.dumps(ev)), out) == 0
    message = json.loads(out.getvalue())["systemMessage"]
    assert "30 of 30" in message and "actrouter-route upgrade" in message
    assert set(json.loads(out.getvalue())) == {"systemMessage"}  # the prompt is not touched
    again = io.StringIO()
    main(io.StringIO(json.dumps(ev)), again)
    assert again.getvalue() == ""


def test_other_errors_still_fail_open_silently(on_opus: str, http_stub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTROUTER_URL", http_stub(lambda m, p, b: (500, {"detail": "boom"})))
    out = io.StringIO()
    main(io.StringIO(json.dumps(prompt_event("where is the retry policy configured?", on_opus))), out)
    assert out.getvalue() == ""
