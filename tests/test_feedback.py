"""Tracking how agents and users respond to routing (actrouter_client.feedback)."""

import asyncio
import io
import json

import pytest

from actrouter_client import Advisor, HttpRouter, feedback
from actrouter_client.claude_hook import decide, main, route_prompt


def _events(path, kind="feedback"):
    if not path.exists():
        return []
    return [e for e in map(json.loads, path.read_text().splitlines()) if kind in (None, e["kind"])]


@pytest.mark.parametrize("message", [
    "that's wrong, the loader is in config/base.py",
    "No, that's not what I asked for",
    "it still fails with the same error",
    "the test still fails",
    "That didn't work",
    "you broke the build",
    "You missed the second call site",
    "try again",
    "revert that",
    "wrong file",
    "hmm, still broken",
    "not quite",
])
def test_corrections(message: str) -> None:
    assert feedback.reaction(message) == "correction"


@pytest.mark.parametrize("message", [
    "thanks!", "Thank you, that works", "perfect", "LGTM", "looks good, now add a test",
    "that works", "works now", "great job", "that's exactly it",
])
def test_approvals(message: str) -> None:
    assert feedback.reaction(message) == "approval"


@pytest.mark.parametrize("message", [
    "yes, go ahead",                         # assent to a proposal, not a verdict on work
    "no, use the other approach",            # an answer to a proposal
    "Fix the failing auth test",             # a new request
    "great, but it still fails",             # not an approval
    "/model haiku",
    "<task-notification>done</task-notification>",
    "why did you change the retry count?",
    "",
])
def test_neither(message: str) -> None:
    assert feedback.reaction(message) is None


def test_a_correction_judges_the_subagents_the_turn_started(ledger_file) -> None:
    feedback.begin("s1", host="claude-code", surface="prompt_hook", task_type="debugging",
                   routed_model="claude-sonnet-5-5", session_model="claude-opus-5-5", request_id="r1",
                   delegate=True, now=100)
    feedback.note("s1", task_type="debugging", model="claude-sonnet-5-5", surface="prompt_hook",
                  request_id="r1", now=110)
    feedback.note("s1", task_type="code_search", model="claude-haiku-5-5", surface="claude_hook", now=120)
    assert feedback.settle("s1", "that's wrong, it still fails", host="claude-code", surface="prompt_hook",
                           now=200) == "correction"
    got = {(e["signal"], e["task_type"], e["model"]) for e in _events(ledger_file)}
    assert got == {("correction", "debugging", "claude-sonnet-5-5"),
                   ("correction", "code_search", "claude-haiku-5-5")}
    assert "still fails" not in ledger_file.read_text()  # the label, never the words
    # the work is judged once
    assert feedback.settle("s1", "that's wrong", host="claude-code", surface="prompt_hook", now=210) is None


def test_with_nothing_handed_off_the_session_model_is_judged(ledger_file) -> None:
    feedback.begin("s1", host="claude-code", surface="prompt_hook", task_type="code_search",
                   routed_model="claude-opus-5-5", session_model="claude-opus-5-5", now=100)
    feedback.settle("s1", "thanks, that works", host="claude-code", surface="prompt_hook", now=200)
    [e] = _events(ledger_file)
    assert (e["signal"], e["task_type"], e["model"]) == ("approval", "code_search", "claude-opus-5-5")


def test_a_delegation_the_agent_did_not_follow_is_declined(ledger_file) -> None:
    feedback.begin("s1", host="claude-code", surface="prompt_hook", task_type="code_search",
                   routed_model="claude-haiku-5-5", session_model="claude-opus-5-5", request_id="r1",
                   delegate=True, now=100)
    feedback.settle("s1", "Now update the docs", host="claude-code", surface="prompt_hook", now=200)
    [e] = _events(ledger_file)
    assert (e["signal"], e["model"], e["about"]) == ("declined", "claude-haiku-5-5", "r1")


def test_a_late_message_is_not_a_reaction(ledger_file) -> None:
    feedback.begin("s1", host="claude-code", surface="prompt_hook", task_type="code_search",
                   routed_model="claude-opus-5-5", session_model="claude-opus-5-5", now=100)
    assert feedback.settle("s1", "that's wrong", host="claude-code", surface="prompt_hook",
                           now=100 + feedback.REACTION_WINDOW_S + 1) is None
    assert _events(ledger_file) == []


def test_a_model_switch_after_routing_agrees_or_overrides(ledger_file) -> None:
    def routed():
        feedback.begin("s1", host="claude-code", surface="prompt_hook", task_type="debugging",
                       routed_model="claude-haiku-5-5", routed_host_model="haiku",
                       session_model="claude-opus-5-5", now=100)

    routed()
    assert feedback.model_switched("s1", "haiku", host="claude-code", surface="prompt_hook", now=150) == "accepted"
    assert feedback.model_switched("s1", "sonnet", host="claude-code", surface="prompt_hook", now=160) is None
    routed()
    assert feedback.model_switched("s1", "sonnet", host="claude-code", surface="prompt_hook",
                                   to_catalog_id="claude-sonnet-5-5", now=150) == "override"
    accepted, override = _events(ledger_file)
    assert (accepted["signal"], accepted["model"]) == ("accepted", "claude-haiku-5-5")
    assert (override["model"], override["chosen_model"], override["chosen_host_model"]) == (
        "claude-haiku-5-5", "claude-sonnet-5-5", "sonnet")
    routed()
    assert feedback.model_switched("s1", "sonnet", host="claude-code", surface="prompt_hook",
                                   now=100 + feedback.SWITCH_WINDOW_S + 1) is None


def test_feedback_can_be_switched_off(ledger_file, monkeypatch) -> None:
    monkeypatch.setenv("ACTROUTER_FEEDBACK", "off")
    feedback.begin("s1", host="claude-code", surface="prompt_hook", task_type="debugging",
                   routed_model="claude-haiku-5-5", session_model="claude-haiku-5-5", now=100)
    assert feedback.settle("s1", "that's wrong", host="claude-code", surface="prompt_hook", now=150) is None
    assert not ledger_file.exists()


# --- through the Claude Code hooks ---------------------------------------------------


def _prompt(text: str, session="s1") -> dict:
    return {"hook_event_name": "UserPromptSubmit", "session_id": session, "prompt": text, "model": "opus"}


def test_the_hook_judges_a_routed_subagent_by_the_next_prompt(claude: Advisor, ledger_file) -> None:
    ev = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
          "tool_input": {"description": "Find config loader", "prompt": "Find where config is loaded",
                         "subagent_type": "Explore"}}
    assert decide(ev, claude)["hookSpecificOutput"]["updatedInput"]["model"] == "haiku"
    route_prompt(_prompt("that's wrong, it's not in that file"), claude)
    [e] = _events(ledger_file)
    assert (e["signal"], e["task_type"], e["surface"]) == ("correction", "code_search", "prompt_hook")
    assert e["model"].startswith("claude-haiku")


def test_the_hook_records_a_delegation_claude_did_not_follow(claude: Advisor, ledger_file) -> None:
    out = route_prompt(_prompt("Find where the retry policy is configured in the codebase"), claude)
    assert "Agent tool" in out["hookSpecificOutput"]["additionalContext"]
    route_prompt(_prompt("Debug the flaky integration test in the auth module"), claude)
    [e] = _events(ledger_file)
    assert (e["signal"], e["task_type"]) == ("declined", "code_search")


def test_an_explicit_model_records_the_agents_choice(claude: Advisor, ledger_file, monkeypatch) -> None:
    # classifying the brief is one more (metered) routing call: opt-in
    monkeypatch.setenv("ACTROUTER_LEARN_AGENT_CHOICE", "on")
    ev = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
          "tool_input": {"description": "Find config loader", "prompt": "Find where config is loaded",
                         "subagent_type": "Explore", "model": "opus"}}
    assert decide(ev, claude) is None
    [e] = _events(ledger_file, "skipped")
    assert (e["reason"], e["model"], e["task_type"]) == ("explicit_model", "opus", "code_search")
    assert e["routed_model"].startswith("claude-haiku") and e["chosen_model"].startswith("claude-opus")
    assert "config" not in json.dumps(e)


def test_an_explicit_model_is_not_classified_by_default(claude: Advisor, ledger_file, http_stub) -> None:
    def handler(method, path, body):
        handler.posts += method == "POST"
        return 200, {}

    handler.posts = 0
    ev = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
          "tool_input": {"description": "Find config loader", "prompt": "Find where config is loaded",
                         "subagent_type": "Explore", "model": "opus"}}
    assert decide(ev, Advisor(HttpRouter(http_stub(handler)), host="claude-code")) is None
    [e] = _events(ledger_file, "skipped")
    assert e["reason"] == "explicit_model" and "task_type" not in e and handler.posts == 0


def test_post_model_switch_records_an_override(claude: Advisor, ledger_file, monkeypatch) -> None:
    route_prompt(_prompt("Find where the retry policy is configured in the codebase"), claude)
    main(io.StringIO(json.dumps({"hook_event_name": "PostModelSwitch", "session_id": "s1",
                                 "to_model": "sonnet"})), io.StringIO())
    [e] = [e for e in _events(ledger_file) if e["signal"] == "override"]
    assert e["task_type"] == "code_search" and e["chosen_model"].startswith("claude-sonnet")


# --- through a chat session ----------------------------------------------------------


def test_a_chat_session_records_reactions_and_model_choices(ledger_file, tmp_path, monkeypatch) -> None:
    from test_chat_session import session

    monkeypatch.setenv("ACTROUTER_STATE_DIR", str(tmp_path / "state"))
    s, agent = session(mode="shadow")

    async def go():
        await s.start()
        await s.turn("search the repo for the retry policy")
        await s.turn("that's wrong, look again")
        await s.set_manual_model("haiku")

    asyncio.run(go())
    correction, accepted = _events(ledger_file)
    assert (correction["signal"], correction["surface"], correction["task_type"], correction["model"]) == (
        "correction", "chat", "code_search", "claude-opus-5-5")
    # routing recommended haiku; shadow mode kept opus; the person then chose haiku
    assert (accepted["signal"], accepted["model"]) == ("accepted", "claude-haiku-5-5")


def test_a_subagent_routed_to_the_delegated_model_follows_the_delegation(claude: Advisor, ledger_file) -> None:
    route_prompt(_prompt("Find where the retry policy is configured in the codebase"), claude)
    ev = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
          "tool_input": {"description": "Find retry policy", "prompt": "Find where the retry policy is configured",
                         "subagent_type": "Explore"}}
    assert decide(ev, claude)["hookSpecificOutput"]["updatedInput"]["model"] == "haiku"
    route_prompt(_prompt("Debug the flaky integration test in the auth module"), claude)
    assert _events(ledger_file) == []

