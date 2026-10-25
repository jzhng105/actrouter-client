"""Recording the baseline model, the routed model and the outcome; reporting savings."""

import io
import json
from pathlib import Path

import pytest

from actrouter_client import Advisor, HttpRouter
from actrouter_client import baseline as baselines
from actrouter_client import ledger
from actrouter_client.claude_hook import decide, main
from actrouter_client.cli import main as cli_main
from actrouter_client.report import _usd, cost_of, format_html, format_text, summarize

CATALOG = [
    {"id": "claude-haiku-4-5", "family": "anthropic", "input_price_per_mtok": 1.0, "output_price_per_mtok": 5.0,
     "cache_read_price_per_mtok": 0.1},
    {"id": "claude-haiku-5-5", "family": "anthropic", "input_price_per_mtok": 0.1, "output_price_per_mtok": 0.5,
     "cache_read_price_per_mtok": 0.01,
     "context_tiers": [{"above_tokens": 100000, "input_price_per_mtok": 0.5, "output_price_per_mtok": 2.5,
                        "cache_read_price_per_mtok": 0.05}]},
    {"id": "claude-sonnet-4-6", "family": "anthropic", "input_price_per_mtok": 3.0, "output_price_per_mtok": 15.0,
     "cache_read_price_per_mtok": 0.3},
    {"id": "claude-sonnet-5-5", "family": "anthropic", "input_price_per_mtok": 2.0, "output_price_per_mtok": 10.0},
    {"id": "claude-opus-4-7", "family": "anthropic", "input_price_per_mtok": 5.0, "output_price_per_mtok": 25.0},
    {"id": "gpt-5.5", "family": "openai", "input_price_per_mtok": 5.0, "output_price_per_mtok": 30.0},
]


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _transcript(tmp_path: Path, *models: str) -> str:
    path = tmp_path / "transcript.jsonl"
    lines = [json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}})]
    lines += [json.dumps({"type": "assistant", "message": {"model": m, "content": []}}) for m in models]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# --- baseline detection ------------------------------------------------------------


def _subagent_event(tmp_path: Path, subagent_type: str = "Explore", **extra) -> dict:
    return {"tool_input": {"subagent_type": subagent_type, "description": "x"}, "cwd": str(tmp_path), **extra}


def test_baseline_is_the_session_model_for_inheriting_agents(tmp_path):
    ev = _subagent_event(tmp_path, transcript_path=_transcript(tmp_path, "claude-opus-4-7", "claude-sonnet-4-6", "<synthetic>"))
    assert baselines.claude_code_subagent(ev) == ("claude-sonnet-4-6", "session model (inherit)")


def test_baseline_precedence(tmp_path, monkeypatch):
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "reviewer.md").write_text("---\nname: code-reviewer\nmodel: opus\n---\nReview code.\n")
    (agents / "helper.md").write_text("---\nname: helper\nmodel: inherit\n---\n")
    ev = lambda t: _subagent_event(tmp_path, t, transcript_path=_transcript(tmp_path, "claude-sonnet-4-6"))

    assert baselines.claude_code_subagent(ev("code-reviewer")) == ("opus", "subagent definition reviewer.md")
    assert baselines.claude_code_subagent(ev("helper"))[1] == "session model (inherit)"
    assert baselines.claude_code_subagent(ev("statusline-setup")) == ("sonnet", "built-in statusline-setup")
    monkeypatch.setenv("CLAUDE_CODE_SUBAGENT_MODEL", "haiku")
    assert baselines.claude_code_subagent(ev("code-reviewer")) == ("haiku", "CLAUDE_CODE_SUBAGENT_MODEL")
    monkeypatch.setenv("ACTROUTER_BASELINE_MODEL", "claude-opus-4-7")
    assert baselines.claude_code_subagent(ev("code-reviewer")) == ("claude-opus-4-7", "ACTROUTER_BASELINE_MODEL")


def test_baseline_falls_back_to_settings_then_unknown(tmp_path, monkeypatch):
    ev = _subagent_event(tmp_path)  # no transcript
    assert baselines.claude_code_subagent(ev) == baselines.UNKNOWN
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text('{"model": "opus"}')
    assert baselines.claude_code_subagent(ev) == ("opus", "settings settings.json")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    assert baselines.claude_code_subagent(ev) == ("claude-sonnet-4-6", "ANTHROPIC_MODEL")


def test_codex_baseline_reads_the_top_level_model(tmp_path, monkeypatch):
    home = tmp_path / "codex"
    home.mkdir()
    (home / "config.toml").write_text('approval = "never"\nmodel = "gpt-5.5"\n\n[profiles.fast]\nmodel = "gpt-5-mini"\n')
    monkeypatch.setenv("CODEX_HOME", str(home))
    assert baselines.agent_cli("codex") == ("gpt-5.5", "codex config.toml")
    (home / "config.toml").write_text('[profiles.fast]\nmodel = "gpt-5-mini"\n')
    assert baselines.agent_cli("codex") == baselines.UNKNOWN


@pytest.mark.parametrize(
    "spelled, catalog_id",
    [
        ("sonnet", "claude-sonnet-5-5"),          # alias: newest of the line, as Claude Code resolves it
        ("haiku", "claude-haiku-5-5"),
        ("claude-sonnet-4-6-20260205", "claude-sonnet-4-6"),  # dated id
        ("anthropic/claude-opus-4-7", "claude-opus-4-7"),     # provider prefix
        ("GPT-5.5", "gpt-5.5"),
        ("gemini-9", None),
        (None, None),
    ],
)
def test_to_catalog_id(spelled, catalog_id):
    assert baselines.to_catalog_id(spelled, CATALOG) == catalog_id


# --- ledger ------------------------------------------------------------------------


def test_ledger_appends_events_and_can_be_switched_off(ledger_file, monkeypatch):
    ledger.record("route", routed_model="m", baseline_model=None)
    [e] = _events(ledger_file)
    assert e["kind"] == "route" and e["routed_model"] == "m" and "baseline_model" not in e
    assert e["v"] == 1 and e["ts"].endswith("+00:00")
    monkeypatch.setenv("ACTROUTER_LEDGER", "off")
    ledger.record("route", routed_model="n")
    assert len(_events(ledger_file)) == 1


def test_ledger_never_raises(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("ACTROUTER_LEDGER", str(blocker / "ledger.jsonl"))  # parent is a file
    ledger.record("route", routed_model="m")
    assert "could not record" in capsys.readouterr().err


def test_ledger_read_skips_malformed_lines_and_filters_by_time(ledger_file):
    ledger_file.write_text(
        '{"kind":"route","ts":"2026-01-01T00:00:00+00:00"}\nnot json\n[1]\n'
        '{"kind":"route","ts":"2026-10-01T00:00:00+00:00"}\n'
    )
    import datetime as dt
    assert len(list(ledger.read(ledger_file))) == 2
    since = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    assert [e["ts"][:7] for e in ledger.read(ledger_file, since=since)] == ["2026-10"]


# --- hooks record what happened -----------------------------------------------------


def test_pre_tool_use_records_the_route_against_the_baseline(claude: Advisor, ledger_file, tmp_path):
    ev = {
        "hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
        "cwd": str(tmp_path), "transcript_path": _transcript(tmp_path, "claude-sonnet-4-6"),
        "tool_input": {"description": "Find config loader", "prompt": "Find where config is loaded",
                       "subagent_type": "Explore"},
    }
    assert decide(ev, claude)["hookSpecificOutput"]["updatedInput"]["model"] == "haiku"
    [e] = _events(ledger_file)
    assert (e["kind"], e["surface"], e["session_id"], e["tool_use_id"]) == ("route", "claude_hook", "s1", "t1")
    assert (e["baseline_model"], e["baseline_source"]) == ("claude-sonnet-4-6", "session model (inherit)")
    assert e["routed_model"].startswith("claude-haiku") and e["task_type"] == "code_search"
    assert e["estimated_savings_usd"] > 0
    assert "Find where config is loaded" not in ledger_file.read_text()  # no task text recorded


def test_explicit_model_is_recorded_as_skipped(claude: Advisor, ledger_file):
    ev = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t2",
          "tool_input": {"description": "x", "prompt": "y", "model": "opus"}}
    assert decide(ev, claude) is None
    [e] = _events(ledger_file)
    assert (e["kind"], e["reason"], e["model"]) == ("skipped", "explicit_model", "opus")


def _agent_call(tool_use_id: str, model: str, cwd: str) -> dict:
    return {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": tool_use_id,
            "cwd": cwd, "tool_input": {"description": "Fix the type errors", "prompt": "...", "model": model}}


def test_a_subagent_started_on_route_task_advice_is_measured(claude: Advisor, ledger_file, tmp_path, monkeypatch):
    # route_task, then an Agent call with the advised model, then its outcome:
    # the flow in which every call used to end up "skipped" and unmeasured.
    from actrouter_client import mcp_server

    project = tmp_path / "proj"
    monkeypatch.setenv("ACTROUTER_PROJECT_DIR", str(project))
    advice = mcp_server._recorded(claude.for_task("Fix the failing test in the auth module", baseline="opus"),
                                  ("opus", "current_model (agent)"), "route_task")
    assert decide(_agent_call("t1", advice["host_model"], str(project)), claude) is None
    main(stdin=io.StringIO(json.dumps({
        "hook_event_name": "PostToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
        "tool_input": {"model": advice["host_model"]},
        "tool_response": {**AGENT_RESULT, "resolvedModel": advice["model"]}})), stdout=io.StringIO())

    advised, claimed, outcome = _events(ledger_file)
    assert (advised["surface"], advised.get("tool_use_id")) == ("mcp", None)
    assert (claimed["kind"], claimed["surface"], claimed["tool_use_id"]) == ("route", "mcp", "t1")
    assert claimed["claims"] == advised["request_id"] and claimed["baseline_as_given"] == "opus"
    assert outcome["kind"] == "outcome"
    s = summarize(_events(ledger_file), claude.catalog())
    assert s["calls"]["routed"] == 1  # the advice and its subagent count once
    assert s["measured"]["calls"] == 1


def test_route_task_advice_is_claimable_when_the_project_dir_is_not_passed(
        claude: Advisor, ledger_file, tmp_path, monkeypatch):
    # Claude Code gives MCP servers no CLAUDE_PROJECT_DIR and can leave the
    # plugin's placeholder unexpanded; the server runs in the project instead.
    from actrouter_client import mcp_server

    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("ACTROUTER_PROJECT_DIR", "${CLAUDE_PROJECT_DIR}")
    monkeypatch.chdir(project)
    advice = mcp_server._recorded(claude.for_task("Fix the failing test in the auth module", baseline="opus"),
                                  ("opus", "current_model (agent)"), "route_task")
    monkeypatch.delenv("ACTROUTER_PROJECT_DIR")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
    assert ledger.claim(str(project), advice["host_model"])["request_id"] == advice["request_id"]


def test_advice_is_claimed_once_by_the_matching_model_only(claude: Advisor, ledger_file, tmp_path, monkeypatch):
    project = str(tmp_path / "proj")
    ledger.offer({"host_model": "sonnet", "model": "claude-sonnet-5-5", "request_id": "r1"},
                 project=project, baseline="opus", baseline_source="x")
    assert ledger.claim(project, "haiku") is None
    assert ledger.claim(str(tmp_path / "other"), "sonnet") is None
    assert ledger.claim(project, "SONNET")["request_id"] == "r1"
    assert ledger.claim(project, "sonnet") is None


def test_stale_advice_is_not_claimed(ledger_file, tmp_path, monkeypatch):
    project = str(tmp_path / "proj")
    ledger.offer({"host_model": "sonnet", "model": "m", "request_id": "r1"},
                 project=project, baseline=None, baseline_source="unknown")
    later = ledger.time.time() + ledger.OFFER_TTL_S + 1
    monkeypatch.setattr(ledger.time, "time", lambda: later)
    assert ledger.claim(project, "sonnet") is None


def test_offers_hold_no_task_text(ledger_file, tmp_path):
    project = str(tmp_path / "proj")
    ledger.offer({"host_model": "sonnet", "model": "m", "request_id": "r1",
                  "reasons": ["docstring:'SECRET'"]}, project=project, baseline=None, baseline_source="unknown")
    assert "SECRET" not in "".join(f.read_text() for f in (tmp_path / "home").rglob("offers-*.json"))


AGENT_RESULT = {
    "status": "completed", "agentId": "a1", "resolvedModel": "claude-haiku-5-5", "modelsUsed": ["claude-haiku-5-5"],
    "totalTokens": 52000, "totalDurationMs": 8100, "totalToolUseCount": 6,
    "usage": {"input_tokens": 2000, "output_tokens": 1000, "cache_read_input_tokens": 48000,
              "cache_creation_input_tokens": 1000, "server_tool_use": None},
    "content": [{"type": "text", "text": "SECRET findings"}], "prompt": "SECRET prompt",
}


def test_post_tool_use_records_the_outcome(ledger_file):
    ev = {"hook_event_name": "PostToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
          "tool_input": {"subagent_type": "Explore", "model": "haiku"}, "tool_response": AGENT_RESULT}
    out = io.StringIO()
    assert main(stdin=io.StringIO(json.dumps(ev)), stdout=out) == 0
    assert out.getvalue() == ""  # PostToolUse never answers
    [e] = _events(ledger_file)
    assert (e["kind"], e["resolved_model"], e["input_tokens"], e["cache_read_input_tokens"]) == (
        "outcome", "claude-haiku-5-5", 2000, 48000)
    assert (e["total_tokens"], e["duration_ms"], e["status"]) == (52000, 8100, "completed")
    assert "SECRET" not in ledger_file.read_text()


def _subagent_transcript(tmp_path: Path) -> Path:
    """Two API requests, the second written twice (once per content block,
    as Claude Code does), plus lines that carry no usage."""
    def assistant(msg_id, model, inp, cached, created, out, text="SECRET reply"):
        return {"type": "assistant", "message": {"id": msg_id, "model": model, "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": inp, "cache_read_input_tokens": cached,
                          "cache_creation_input_tokens": created, "output_tokens": out}}}
    lines = [
        {"type": "user", "message": {"role": "user", "content": "SECRET task"}},
        assistant("m1", "claude-haiku-5-5", 10, 0, 4_000, 300),
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "SECRET"}]}},
        assistant("m2", "claude-haiku-5-5", 5, 4_000, 60_000, 900),
        assistant("m2", "claude-haiku-5-5", 5, 4_000, 60_000, 900),
        assistant("s1", "<synthetic>", 0, 0, 0, 0),
    ]
    path = tmp_path / "subagents" / "agent-abc.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def test_subagent_usage_counts_each_request_once(tmp_path):
    from actrouter_client.transcript import subagent_usage

    usage = subagent_usage(str(_subagent_transcript(tmp_path)))
    assert usage == {"input_tokens": 15, "cache_read_input_tokens": 4_000, "cache_creation_input_tokens": 64_000,
                     "output_tokens": 1_200, "requests": 2, "max_request_prompt_tokens": 64_005,
                     "resolved_model": "claude-haiku-5-5", "models_used": ["claude-haiku-5-5"]}
    assert subagent_usage(str(tmp_path / "missing.jsonl")) is None


def _launch_events(tmp_path: Path, agent_id_in_stop: str) -> list[dict]:
    pre = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
           "cwd": str(tmp_path), "tool_input": {"description": "Find where config is loaded", "prompt": "..."}}
    launched = {"hook_event_name": "PostToolUse", "tool_name": "Agent", "session_id": "s1", "tool_use_id": "t1",
                "tool_input": {"model": "haiku"},
                "tool_response": {"status": "async_launched", "agentId": "abc", "resolvedModel": "claude-haiku-5-5",
                                  "description": "x", "prompt": "SECRET prompt", "outputFile": "/tmp/out"}}
    stop = {"hook_event_name": "SubagentStop", "session_id": "s1", "agent_id": agent_id_in_stop, "agent_type": "Explore",
            "agent_transcript_path": str(_subagent_transcript(tmp_path)), "stop_hook_active": False,
            "last_assistant_message": "SECRET findings"}
    return [pre, launched, stop]


@pytest.mark.parametrize("agent_id_in_stop", ["abc", "agent-abc"])
def test_a_background_subagent_is_measured_when_it_finishes(claude, ledger_file, tmp_path, monkeypatch,
                                                            agent_id_in_stop):
    monkeypatch.setenv("ACTROUTER_BASELINE_MODEL", "sonnet")
    pre, launched, stop = _launch_events(tmp_path, agent_id_in_stop)
    assert decide(pre, claude) is not None
    for ev in (launched, stop):
        out = io.StringIO()
        assert main(stdin=io.StringIO(json.dumps(ev)), stdout=out) == 0
        assert out.getvalue() == ""  # an answer to SubagentStop could keep the subagent running
    route, launch, finish = _events(ledger_file)
    assert (launch["status"], launch["agent_id"]) == ("async_launched", "abc")
    assert (finish["surface"], finish["agent_id"], finish["requests"]) == ("subagent_stop", "abc", 2)
    assert "SECRET" not in ledger_file.read_text()
    m = summarize(_events(ledger_file), CATALOG + claude.catalog())["measured"]
    assert m["calls"] == 1 and m["tokens"] == 15 + 4_000 + 64_000 + 1_200


def test_subagent_stop_without_a_transcript_records_nothing(ledger_file, tmp_path):
    ev = {"hook_event_name": "SubagentStop", "session_id": "s1", "agent_id": "ghost",
          "agent_transcript_path": str(tmp_path / "never-written.jsonl")}
    assert main(stdin=io.StringIO(json.dumps(ev)), stdout=io.StringIO()) == 0
    assert not ledger_file.exists()


def test_a_foreground_outcome_is_kept_over_the_subagent_stop_copy():
    route = _route("t1", "code_search", "claude-haiku-4-5", "claude-sonnet-4-6")
    foreground = {**_outcome("t1", "claude-haiku-4-5", 10_000, 1_000), "agent_id": "abc"}
    stop = {"kind": "outcome", "ts": "2026-10-20T10:01:00+00:00", "session_id": "s", "agent_id": "abc",
            "surface": "subagent_stop", "input_tokens": 99_000, "output_tokens": 9_000}
    m = summarize([route, foreground, stop], CATALOG)["measured"]
    assert (m["calls"], m["tokens"]) == (1, 11_000)


def test_the_long_context_tier_applies_per_request_not_to_the_sum():
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 150_000}  # e.g. three 50k requests
    haiku = CATALOG[1]  # 0.1/M, 0.5/M above 100k
    assert cost_of(usage, haiku) == pytest.approx(150_000 * 0.5 / 1e6)
    assert cost_of(usage, haiku, prompt_tokens=50_000) == pytest.approx(150_000 * 0.1 / 1e6)


def test_post_tool_use_ignores_other_tools(ledger_file):
    ev = {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_response": {"stdout": "x"}}
    main(stdin=io.StringIO(json.dumps(ev)), stdout=io.StringIO())
    assert not ledger_file.exists()


# --- report ------------------------------------------------------------------------


def test_cost_of_prices_cache_reads_and_long_context_tiers():
    usage = {"input_tokens": 2000, "output_tokens": 1000, "cache_read_input_tokens": 48000,
             "cache_creation_input_tokens": 1000}
    haiku = CATALOG[1]
    # 51k prompt tokens: base tier. fresh 3000*0.1 + cached 48000*0.01 + out 1000*0.5
    assert cost_of(usage, haiku) == pytest.approx((300 + 480 + 500) / 1e6)
    big = {**usage, "cache_read_input_tokens": 148000}  # 151k prompt: long-context tier
    assert cost_of(big, haiku) == pytest.approx((3000 * 0.5 + 148000 * 0.05 + 1000 * 2.5) / 1e6)
    no_cache_price = CATALOG[3]
    assert cost_of(usage, no_cache_price) == pytest.approx((51000 * 2.0 + 1000 * 10.0) / 1e6)
    assert cost_of(usage, None) is None


def _route(tool_use_id, task, routed, baseline, est=0.001, base_est=0.01, day="2026-10-20"):
    return {"kind": "route", "ts": f"{day}T10:00:00+00:00", "surface": "claude_hook", "session_id": "s",
            "tool_use_id": tool_use_id, "task_type": task, "routed_model": routed, "baseline_model": baseline,
            "baseline_source": "session model (inherit)", "estimated_cost_usd": est,
            "baseline_estimated_cost_usd": base_est}


def _outcome(tool_use_id, model, inp, out, cached=0, day="2026-10-20"):
    return {"kind": "outcome", "ts": f"{day}T10:01:00+00:00", "session_id": "s", "tool_use_id": tool_use_id,
            "resolved_model": model, "input_tokens": inp, "output_tokens": out,
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}


EVENTS = [
    _route("t1", "code_search", "claude-haiku-4-5", "claude-sonnet-4-6"),
    _outcome("t1", "claude-haiku-4-5-20251001", 100_000, 10_000),
    _route("t2", "debugging", "claude-sonnet-4-6", "claude-sonnet-4-6", est=0.02, base_est=0.02, day="2026-10-21"),
    _outcome("t2", "claude-sonnet-4-6", 10_000, 1_000, day="2026-10-21"),
    _route("t3", "code_search", "claude-haiku-4-5", None),                    # no baseline
    _route("t4", "planning", "claude-opus-4-7", "claude-sonnet-4-6", est=0.05, base_est=0.03),  # no outcome
    {"kind": "skipped", "ts": "2026-10-21T00:00:00+00:00", "reason": "explicit_model", "model": "opus"},
]


def test_summary_measures_and_estimates_savings():
    s = summarize(EVENTS, CATALOG, source="ledger.jsonl")
    m = s["measured"]
    # t1: haiku 100k*1 + 10k*5 = 0.15 vs sonnet 100k*3 + 10k*15 = 0.45; t2: same model both sides.
    t2 = (10_000 * 3 + 1_000 * 15) / 1e6
    assert m["calls"] == 2
    assert m["actual_usd"] == pytest.approx(0.15 + t2)
    assert m["baseline_usd"] == pytest.approx(0.45 + t2)
    assert m["saved_usd"] == pytest.approx(0.30)
    assert m["saved_pct"] == pytest.approx(round(100 * 0.30 / (0.45 + t2), 1))
    e = s["estimated"]  # t1, t2, t4 (t3 has no baseline)
    assert (e["calls"], e["saved_usd"]) == (3, pytest.approx(0.009 + 0 - 0.02))
    c = s["calls"]
    assert (c["routed"], c["skipped_explicit_model"], c["changed_model"], c["with_outcome"], c["baseline_unknown"]) == (4, 1, 2, 2, 1)
    assert c["coverage_pct"] == 80.0
    search = next(r for r in s["by_task_type"] if r["task_type"] == "code_search")
    assert (search["calls"], search["measured_calls"], search["saved_usd"]) == (2, 1, pytest.approx(0.30))
    switch = next(r for r in s["by_switch"] if r["routed_model"] == "claude-haiku-4-5")
    assert switch["baseline_model"] == "claude-sonnet-4-6" and switch["saved_usd"] == pytest.approx(0.30)
    assert [d["day"] for d in s["by_day"]] == ["2026-10-20", "2026-10-21"]
    assert any("no detectable baseline" in n for n in s["notes"])
    assert c["unmeasured_by_cause"] == {"no_outcome": 1}  # t4
    assert any("1 routed calls are subagents with no PostToolUse outcome" in n for n in s["notes"])


def test_unmeasured_calls_are_projected_from_their_task_types_measured_median():
    events = [
        _route("m1", "debugging", "claude-sonnet-4-6", "claude-opus-4-7"),
        _outcome("m1", "claude-sonnet-4-6", 20_000, 2_000),
        _route("m2", "debugging", "claude-sonnet-4-6", "claude-opus-4-7"),
        _outcome("m2", "claude-sonnet-4-6", 40_000, 4_000),
        _route("m3", "debugging", "claude-sonnet-4-6", "claude-opus-4-7"),
        _outcome("m3", "claude-sonnet-4-6", 90_000, 9_000),
        # MCP advice and a background launch: no usage of their own
        {**_route(None, "debugging", "claude-sonnet-4-6", "claude-opus-4-7"), "surface": "mcp"},
        _route("bg", "debugging", "claude-sonnet-4-6", "claude-opus-4-7"),
        {"kind": "outcome", "ts": "2026-10-20T10:01:00+00:00", "session_id": "s", "tool_use_id": "bg",
         "status": "async_launched"},
        # one measured code_search call is too few to project from
        _route("c1", "code_search", "claude-haiku-4-5", "claude-sonnet-4-6"),
        _outcome("c1", "claude-haiku-4-5", 10_000, 1_000),
        _route("c2", "code_search", "claude-haiku-4-5", "claude-sonnet-4-6"),
    ]
    p = summarize(events, CATALOG)["projected"]
    # the median measured debugging call: 40k in, 4k out, on each side
    sonnet, opus = (40_000 * 3 + 4_000 * 15) / 1e6, (40_000 * 5 + 4_000 * 25) / 1e6
    assert p["calls"] == 2
    assert p["routed_usd"] == pytest.approx(2 * sonnet)
    assert p["baseline_usd"] == pytest.approx(2 * opus)
    assert "projected (2 unmeasured calls" in format_text(summarize(events, CATALOG))


def test_estimates_from_before_token_priors_are_flagged():
    events = [_route("t1", "debugging", "claude-sonnet-4-6", "claude-opus-4-7"),
              {**_route("t2", "debugging", "claude-sonnet-4-6", "claude-opus-4-7"), "estimate_basis": "prior"}]
    s = summarize(events, CATALOG)
    assert s["estimated"]["basis"] == {"prior": 1, "unrecorded": 1}
    assert any("1 estimates predate token priors" in n for n in s["notes"])
    assert "basis prior 1, unrecorded 1" in format_text(s)


def test_unmeasured_calls_are_explained_by_cause():
    events = [
        {**_route(None, "debugging", "claude-sonnet-4-6", "claude-opus-4-7"), "surface": "mcp", "tool": "route_task"},
        {**_route(None, "summarization", "claude-haiku-4-5", "claude-sonnet-4-6"), "surface": "mcp",
         "tool": "route_function"},
        {**_route(None, "debugging", "claude-sonnet-4-6", "claude-opus-4-7"), "surface": "launch"},
        _route("bg", "code_search", "claude-haiku-4-5", "claude-sonnet-4-6"),
        {"kind": "outcome", "ts": "2026-10-20T10:01:00+00:00", "session_id": "s", "tool_use_id": "bg",
         "agent_id": "abc", "status": "async_launched"},
        _route("t9", "code_search", "claude-haiku-4-5", "claude-sonnet-4-6"),
    ]
    s = summarize(events, CATALOG)
    assert s["calls"]["unmeasured_by_cause"] == {
        "unclaimed_advice": 1, "code_advice": 1, "launch": 1, "background_pending": 1, "no_outcome": 1}
    notes = " ".join(s["notes"])
    assert "no subagent claimed" in notes and "SubagentStop" in notes
    assert "PostToolUse hook not installed" not in notes  # the old catch-all guess


def test_summary_counts_prompt_routing():
    prompt = {"v": 1, "ts": "2026-10-21T10:00:00+00:00", "kind": "prompt", "surface": "prompt_hook",
              "routed_model": "claude-haiku-4-5", "baseline_model": "claude-sonnet-4-6"}
    delegated = {**_route("t9", "code_search", "claude-haiku-4-5", "claude-sonnet-4-6"), "surface": "prompt_hook"}
    c = summarize([prompt, delegated], CATALOG)["calls"]
    assert (c["prompts_routed"], c["prompts_delegated"], c["routed"]) == (1, 1, 1)
    assert "prompts routed      1" in format_text(summarize([prompt, delegated], CATALOG))


def test_text_and_html_reports(tmp_path):
    s = summarize(EVENTS + [_route("t9", "<script>x</script>", "claude-haiku-4-5", "claude-sonnet-4-6")], CATALOG,
                  source="ledger.jsonl")
    text = format_text(s)
    assert "saved             $0.30" in text and "claude-sonnet-4-6 -> claude-haiku-4-5" in text
    page = format_html(s)
    assert page.startswith("<!doctype html>") and "<title>ActRouter Savings</title>" in page
    assert "<script>" not in page and "&lt;script&gt;" in page  # ledger content is escaped
    assert "prefers-color-scheme: dark" in page and "<svg" in page
    empty = format_html(summarize([], CATALOG))
    assert "No measured calls yet" in empty and "No routed calls yet" in empty


def test_cli_report_writes_the_dashboard(ledger_file, tmp_path, capsys):
    ledger_file.write_text("\n".join(json.dumps(e) for e in EVENTS) + "\n")
    out = tmp_path / "savings.html"
    assert cli_main(["report", "--days", "0", "--html", str(out)]) == 0
    assert "routed calls        4" in capsys.readouterr().out
    assert out.read_text().startswith("<!doctype html>")
    assert cli_main(["report", "--days", "0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["calls"]["routed"] == 4


def test_cli_task_records_with_baseline(ledger_file, capsys):
    assert cli_main(["task", "Search the repo for usages of parse_args", "--host", "claude-code",
                     "--baseline", "sonnet", "--json"]) == 0
    advice = json.loads(capsys.readouterr().out)
    assert advice["baseline_model"].startswith("claude-sonnet") and advice["estimated_savings_usd"] is not None
    [e] = _events(ledger_file)
    assert (e["surface"], e["baseline_source"]) == ("cli", "--baseline")


# --- HTTP: deployments that predate baseline_model -------------------------------------


def test_http_retries_without_baseline_on_an_old_deployment(http_stub):
    seen = []

    def handler(method, path, body):
        seen.append(body)
        if body and "baseline_model" in body:
            return 422, {"detail": [{"loc": ["body", "baseline_model"], "msg": "Extra inputs are not permitted"}]}
        return 200, {"chosen_model": "claude-haiku-4-5", "router": "heuristic", "ranking": []}

    resp = HttpRouter(http_stub(handler)).route_task({"description": "x"}, baseline_model="claude-sonnet-4-6")
    assert resp["chosen_model"] == "claude-haiku-4-5" and "compat_baseline" in resp
    assert "baseline_model" in seen[0] and "baseline_model" not in seen[1]


def test_http_other_422s_still_raise(http_stub):
    from actrouter_client import ActRouterError

    url = http_stub(lambda m, p, b: (422, {"detail": "task.description: field required"}))
    with pytest.raises(ActRouterError):
        HttpRouter(url).route_task({"description": "x"}, baseline_model="m")


@pytest.mark.parametrize(
    "value, text",
    [(None, "-"), (0.0, "$0.00"), (0.0042, "$0.0042"), (0.3921, "$0.39"), (-2.67, "-$2.67"), (1234.5, "$1,234.50")],
)
def test_money_formatting(value, text):
    assert _usd(value) == text


def test_negative_savings_are_marked():
    events = [_route("t1", "planning", "claude-opus-4-7", "claude-sonnet-4-6"),
              _outcome("t1", "claude-opus-4-7", 10_000, 1_000)]
    page = format_html(summarize(events, CATALOG))
    assert 'class="num neg">-$' in page
