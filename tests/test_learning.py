"""Learning routing adjustments from the ledger (actrouter_client.learning)."""

import datetime as dt
import io
import json
import os
from contextlib import redirect_stdout

import pytest

from actrouter_client import HttpRouter, learning, ledger
from actrouter_client.cli import main as cli

from api_replay import RECORDING

#: the catalog as the API serves it (GET /v1/models, recorded)
CATALOG = json.loads(RECORDING.read_text())["GET /v1/models null"]["body"]["data"]
NOW = dt.datetime(2026, 11, 3, 12, tzinfo=dt.timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def fresh_activation():
    learning.reset()
    yield
    learning.reset()


def ts(minutes_ago: float = 0, days_ago: float = 0) -> str:
    t = NOW - minutes_ago * 60 - days_ago * 86400
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds")


def fb(signal, task="debugging", model="claude-haiku-5-5", days_ago=0.0, **extra):
    return {"kind": "feedback", "ts": ts(days_ago=days_ago), "signal": signal, "task_type": task,
            "model": model, "session_id": "s1", **extra}


def found(events):
    return [(s.signal, s.task_type, s.model, s.voted_for) for s in learning.signals(events, CATALOG, now=NOW)]


def test_subagent_outcomes_are_joined_to_their_routes() -> None:
    events = [
        {"kind": "route", "ts": ts(30), "session_id": "s1", "tool_use_id": "t1", "task_type": "code_search",
         "routed_model": "claude-haiku-5-5", "surface": "claude_hook"},
        {"kind": "outcome", "ts": ts(29), "session_id": "s1", "tool_use_id": "t1", "status": "completed"},
        # a background subagent: launched, then finished under its agent id
        {"kind": "route", "ts": ts(20), "session_id": "s1", "tool_use_id": "t2", "task_type": "debugging",
         "routed_model": "claude-sonnet-5-5", "surface": "claude_hook"},
        {"kind": "outcome", "ts": ts(20), "session_id": "s1", "tool_use_id": "t2", "status": "async_launched",
         "agent_id": "a2"},
        {"kind": "outcome", "ts": ts(10), "session_id": "s1", "agent_id": "a2", "status": "completed"},
        {"kind": "route", "ts": ts(9), "session_id": "s1", "tool_use_id": "t3", "task_type": "debugging",
         "routed_model": "claude-sonnet-5-5", "surface": "claude_hook"},
        {"kind": "outcome", "ts": ts(8), "session_id": "s1", "tool_use_id": "t3", "status": "failed"},
        # an outcome for a subagent that was never routed teaches nothing
        {"kind": "outcome", "ts": ts(5), "session_id": "s1", "tool_use_id": "t9", "status": "failed"},
    ]
    assert found(events) == [
        ("completed", "code_search", "claude-haiku-5-5", None),
        ("completed", "debugging", "claude-sonnet-5-5", None),
        ("failed", "debugging", "claude-sonnet-5-5", None),
    ]


def test_work_redone_on_a_larger_model_is_an_escalation() -> None:
    events = [
        {"kind": "route", "ts": ts(30), "session_id": "s1", "tool_use_id": "t1", "task_type": "debugging",
         "routed_model": "claude-haiku-5-5"},
        {"kind": "skipped", "ts": ts(25), "session_id": "s1", "tool_use_id": "t2", "reason": "explicit_model",
         "model": "opus", "task_type": "debugging", "routed_model": "claude-haiku-5-5",
         "chosen_model": "claude-opus-5-5"},
        # another session, and a later unrelated task: no escalation
        {"kind": "route", "ts": ts(20), "session_id": "s2", "tool_use_id": "t3", "task_type": "debugging",
         "routed_model": "claude-opus-5-5"},
    ]
    assert found(events) == [
        ("agent_choice", "debugging", "claude-haiku-5-5", "claude-opus-5-5"),
        ("escalated", "debugging", "claude-haiku-5-5", "claude-opus-5-5"),
    ]


def test_feedback_turns_and_unclaimed_advice() -> None:
    events = [
        fb("correction"),
        fb("override", chosen_host_model="sonnet"),
        {"kind": "turn", "ts": ts(5), "session_id": "c1", "task_type": "code_search",
         "served_model": "claude-opus-5-5", "status": "interrupted"},
        {"kind": "turn", "ts": ts(4), "session_id": "c1", "status": "completed"},  # not routed
        {"kind": "route", "ts": ts(60), "surface": "mcp", "tool": "route_task", "host": "claude-code",
         "request_id": "r1", "task_type": "debugging", "routed_model": "claude-sonnet-5-5"},
        {"kind": "route", "ts": ts(60), "surface": "mcp", "tool": "route_task", "host": "claude-code",
         "request_id": "r2", "task_type": "debugging", "routed_model": "claude-sonnet-5-5"},
        {"kind": "route", "ts": ts(55), "surface": "mcp", "session_id": "s1", "tool_use_id": "t1",
         "claims": "r2", "task_type": "debugging", "routed_model": "claude-sonnet-5-5"},
        # Codex cannot claim advice, so its advice is never "unclaimed"
        {"kind": "route", "ts": ts(60), "surface": "mcp", "tool": "route_task", "host": "codex",
         "request_id": "r3", "task_type": "debugging", "routed_model": "gpt-5.5"},
    ]
    got = found(events)
    assert ("correction", "debugging", "claude-haiku-5-5", None) in got
    assert ("override", "debugging", "claude-haiku-5-5", "claude-sonnet-5-5") in got
    assert ("interrupted", "code_search", "claude-opus-5-5", None) in got
    assert ("unclaimed", "debugging", "claude-sonnet-5-5", None) in got
    assert len(got) == 4


def adjustment(events, cfg=learning.LearnConfig(), task="debugging", model="claude-haiku-5-5"):
    return learning.learn(events, CATALOG, cfg, now=NOW)["adjustments"][task][model]


def test_repeated_corrections_lower_a_model_up_to_the_cap() -> None:
    e = adjustment([fb("correction")] * 3)
    assert e["bias"] == -0.12 and e["bad"] == 3.0
    assert adjustment([fb("correction")] * 2)["bias"] == 0.0  # less than min_evidence


def test_good_signals_offset_bad_ones_but_never_promote() -> None:
    mixed = adjustment([fb("correction")] + [fb("approval")] * 3)
    assert -0.12 < mixed["bias"] < 0
    assert adjustment([fb("approval")] * 10)["bias"] == 0.0


def test_votes_promote_the_model_chosen_instead() -> None:
    events = [fb("override", chosen_model="claude-sonnet-5-5")] * 3
    assert adjustment(events)["bias"] == -0.12
    promoted = adjustment(events, model="claude-sonnet-5-5")
    assert 0 < promoted["bias"] < 0.12 and promoted["votes"] == 3.0


def test_old_evidence_fades() -> None:
    old = [fb("correction", days_ago=90)] * 3  # three half-lives
    e = adjustment(old)
    assert e["bias"] == 0.0 and e["evidence"] == pytest.approx(0.375)


def test_learned_adjustments_change_what_the_router_chooses(router, tmp_path, monkeypatch) -> None:
    candidates = ["claude-haiku-5-5", "claude-haiku-4-5", "claude-sonnet-5-5"]

    def chosen():
        return router.route_task({"description": "Find where the config is loaded"},
                                 candidate_models=candidates)["chosen_model"]

    assert chosen() == "claude-haiku-5-5"
    path = tmp_path / "learned.json"
    payload = learning.learn([fb("correction", task="code_search")] * 4, CATALOG, now=NOW)
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("ACTROUTER_LEARNED", str(path))
    assert chosen() == "claude-haiku-5-5"  # shadow, the default: not sent
    monkeypatch.setenv("ACTROUTER_LEARNING", "on")
    resp = router.route_task({"description": "Find where the config is loaded"}, candidate_models=candidates)
    assert resp["chosen_model"] != "claude-haiku-5-5"
    reasons = {m["model_id"]: m["reasons"] for m in resp["ranking"]}
    assert any(r.startswith("learned=-0.11") for r in reasons["claude-haiku-5-5"])


def _write_ledger(path, events) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def test_activate_relearns_and_applies_only_when_on(ledger_file, tmp_path, monkeypatch) -> None:
    # recorded just now: activate() learns at the real time, so NOW's events would have decayed
    just_now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _write_ledger(ledger_file, [{**fb("correction", task="code_search"), "ts": just_now}] * 3)
    target = tmp_path / "learned.json"
    monkeypatch.setenv("ACTROUTER_LEARNED", str(target))

    # shadow, the default: learned, not sent
    assert learning.activate(lambda: CATALOG) is None
    assert json.loads(target.read_text())["adjustments"]["code_search"]

    learning.reset()
    monkeypatch.setenv("ACTROUTER_LEARNING", "on")
    sent = learning.activate(lambda: CATALOG)
    assert sent and all(b < 0 for b in sent["code_search"].values())
    assert all(b != 0 for models in sent.values() for b in models.values())

    learning.reset()
    monkeypatch.setenv("ACTROUTER_LEARNING", "off")
    target.unlink()
    assert learning.activate(lambda: CATALOG) is None
    assert not target.exists()


def test_fresh_adjustments_are_not_relearned(ledger_file, tmp_path, monkeypatch) -> None:
    _write_ledger(ledger_file, [fb("correction")])
    target = tmp_path / "learned.json"
    monkeypatch.setenv("ACTROUTER_LEARNED", str(target))
    assert learning.refresh(lambda: CATALOG) is not None
    assert learning.refresh(lambda: CATALOG) is None  # within refresh_hours
    monkeypatch.setenv("ACTROUTER_LEARN_REFRESH_HOURS", "0")
    assert learning.refresh(lambda: CATALOG) is not None


def test_no_ledger_learns_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ACTROUTER_LEARNED", str(tmp_path / "learned.json"))
    assert learning.refresh(lambda: CATALOG) is None
    assert not (tmp_path / "learned.json").exists()


def test_the_learn_command(ledger_file, tmp_path, monkeypatch) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _write_ledger(ledger_file, [{**fb("correction", task="code_search"), "ts": now}] * 3)
    target = tmp_path / "learned.json"
    monkeypatch.setenv("ACTROUTER_LEARNED", str(target))

    out = io.StringIO()
    with redirect_stdout(out):
        assert cli(["learn"]) == 0
    text = out.getvalue()
    assert "code_search" in text and "-0.120" in text and "Shadow mode" in text
    assert target.exists()

    target.unlink()
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli(["learn", "--json", "--dry-run"]) == 0
    payload = json.loads(out.getvalue())
    assert payload["signals"] == {"correction": 3} and payload["file"] is None
    assert not target.exists()


def test_ledger_reads_the_feedback_the_hooks_write(ledger_file) -> None:
    ledger.record("feedback", signal="approval", task_type="qa", model="claude-haiku-5-5")
    [e] = list(ledger.read())
    assert found([e]) == [("approval", "qa", "claude-haiku-5-5", None)]


def test_claude_code_evidence_about_an_alias_covers_its_whole_line() -> None:
    events = [fb("correction", task="code_search", host="claude-code")] * 3
    adjustments = learning.learn(events, CATALOG, now=NOW)["adjustments"]["code_search"]
    # `haiku` runs the newest Haiku whichever Haiku was routed: penalize the alias
    assert adjustments["claude-haiku-5-5"]["bias"] == adjustments["claude-haiku-4-5"]["bias"] == -0.12
    assert "claude-sonnet-5-5" not in adjustments
    # the same evidence from a host that runs exact ids stays with that id
    exact = learning.learn([fb("correction", task="code_search", host="codex")] * 3, CATALOG, now=NOW)
    assert list(exact["adjustments"]["code_search"]) == ["claude-haiku-5-5"]


def test_choosing_the_alias_routing_chose_is_not_a_disagreement() -> None:
    events = [{"kind": "skipped", "ts": ts(5), "session_id": "s1", "tool_use_id": "t1", "host": "claude-code",
               "reason": "explicit_model", "model": "haiku", "task_type": "code_search",
               "routed_model": "claude-haiku-4-5", "chosen_model": "claude-haiku-5-5"}]
    assert found(events) == []


def test_with_line_wide_evidence_claude_code_moves_up_a_tier(claude, tmp_path, monkeypatch) -> None:
    advisor = claude
    assert advisor.for_task("Find where the config is loaded")["host_model"] == "haiku"
    path = tmp_path / "learned.json"
    events = [fb("correction", task="code_search", host="claude-code")] * 3
    path.write_text(json.dumps(learning.learn(events, CATALOG, now=NOW)))
    monkeypatch.setenv("ACTROUTER_LEARNED", str(path))
    monkeypatch.setenv("ACTROUTER_LEARNING", "on")
    assert advisor.for_task("Find where the config is loaded")["host_model"] == "sonnet"


# --- what the client sends ------------------------------------------------------------


def _stub(http_stub, *, rejects_learned=False):
    def handler(method, path, body):
        handler.requests.append((method, path, body))
        if method == "GET":
            return 200, {"data": CATALOG}
        if rejects_learned and "learned" in (body or {}):
            return 422, {"detail": [{"loc": ["body", "learned"], "msg": "Extra inputs are not permitted"}]}
        return 200, {"chosen_model": "claude-haiku-5-5", "ranking": [], "request_id": "r1"}

    handler.requests = []
    return HttpRouter(http_stub(handler)), handler


def test_requests_carry_learned_only_when_on(http_stub, tmp_path, monkeypatch) -> None:
    path = tmp_path / "learned.json"
    path.write_text(json.dumps(learning.learn([fb("correction", task="code_search")] * 4, CATALOG, now=NOW)))
    monkeypatch.setenv("ACTROUTER_LEARNED", str(path))
    router, handler = _stub(http_stub)

    router.route_task({"description": "Find where the config is loaded"})
    assert "learned" not in handler.requests[-1][2]  # shadow

    learning.reset()
    monkeypatch.setenv("ACTROUTER_LEARNING", "on")
    router.route_task({"description": "Find where the config is loaded"})
    sent = handler.requests[-1][2]["learned"]
    assert set(sent) == {"code_search"} and sent["code_search"]["claude-haiku-5-5"] < 0
    router.route_function({"name": "find_config"})
    assert handler.requests[-1][2]["learned"] == sent


def test_nothing_learned_sends_nothing(http_stub, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ACTROUTER_LEARNED", str(tmp_path / "learned.json"))
    monkeypatch.setenv("ACTROUTER_LEARNING", "on")
    router, handler = _stub(http_stub)
    router.route_task({"description": "Find where the config is loaded"})
    assert "learned" not in handler.requests[-1][2]


def test_a_deployment_without_learned_still_routes(http_stub, tmp_path, monkeypatch) -> None:
    path = tmp_path / "learned.json"
    path.write_text(json.dumps(learning.learn([fb("correction", task="code_search")] * 4, CATALOG, now=NOW)))
    monkeypatch.setenv("ACTROUTER_LEARNED", str(path))
    monkeypatch.setenv("ACTROUTER_LEARNING", "on")
    router, handler = _stub(http_stub, rejects_learned=True)
    resp = router.route_task({"description": "Find where the config is loaded"})
    assert resp["chosen_model"] == "claude-haiku-5-5" and "compat_learned" in resp
    assert "learned" not in handler.requests[-1][2]
