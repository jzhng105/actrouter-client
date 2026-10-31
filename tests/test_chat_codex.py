import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from actrouter_client.chat.base import APPROVE_SESSION, DENY
from actrouter_client.chat.codex import DEVELOPER_INSTRUCTIONS, CodexAgent
from actrouter_client.chat.ui import ScriptedUI

FAKE = Path(__file__).parent / "fake_codex.py"


@pytest.fixture()
def codex_log(tmp_path, monkeypatch):
    log = tmp_path / "codex.jsonl"
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    return log


@pytest.fixture()
def fake_bin(tmp_path):
    # `<bin> app-server` must start the fake: a tiny wrapper script
    exe = tmp_path / "codex"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} -I {FAKE} \"$@\"\n")
    exe.chmod(0o755)
    return str(exe)


def sent(log: Path, method: str) -> list[dict]:
    return [m for m in map(json.loads, log.read_text().splitlines()) if m.get("method") == method]


def agent(fake_bin, **kw) -> CodexAgent:
    return CodexAgent(codex_bin=fake_bin, cwd=os.getcwd(), **kw)


def test_one_thread_with_the_model_set_per_turn(fake_bin, codex_log) -> None:
    async def go():
        a = agent(fake_bin, effort="high")
        ui = ScriptedUI()
        await a.start(model="gpt-5.5")
        first = await a.run_turn("hello", ui)
        await a.set_model("gpt-5.4-mini")
        second = await a.run_turn("again", ui)
        third = await a.run_turn("and again", ui)
        await a.close()
        return a, ui, first, second, third

    a, ui, first, second, third = asyncio.run(go())
    assert a.session_id == "thr-1" and a.model == "gpt-5.4-mini"
    start = sent(codex_log, "thread/start")[0]["params"]
    assert start["model"] == "gpt-5.5" and start["developerInstructions"] == DEVELOPER_INSTRUCTIONS
    turns = [m["params"] for m in sent(codex_log, "turn/start")]
    assert [t["threadId"] for t in turns] == ["thr-1"] * 3
    assert "model" not in turns[0] and turns[1]["model"] == "gpt-5.4-mini" and "model" not in turns[2]
    assert turns[0]["effort"] == "high" and "effort" not in turns[1]  # persists like the model
    assert turns[0]["input"] == [{"type": "text", "text": "hello", "text_elements": []}]
    assert sent(codex_log, "initialized")

    assert first.status == "completed" and first.models == ["gpt-5.5"]
    assert second.models == ["gpt-5.4-mini"]
    assert "".join(ui.texts).count("on gpt-5.4-mini") == 2
    # two requests: the repeated usage report is not a third
    assert [(r.prompt, r.cached, r.output) for r in first.requests] == [(30000, 20000, 400), (33000, 30000, 300)]
    assert first.context_window == 258400 and first.reply == "done"
    assert "sub-agent" not in " ".join(ui.activities)


def test_approvals_go_to_the_person(fake_bin, codex_log) -> None:
    async def go(answer):
        a = agent(fake_bin)
        ui = ScriptedUI()

        async def approve(kind, summary, **kw):
            ui.approvals.append((kind, summary, answer))
            return answer

        ui.approve = approve
        await a.start(model=None)
        result = await a.run_turn("please run it", ui)
        await a.close()
        return ui, result

    ui, result = asyncio.run(go(APPROVE_SESSION))
    assert ui.approvals == [("command", "ls -la", APPROVE_SESSION)]
    assert "command acceptForSession" in "".join(ui.texts)
    assert "$ ls -la" in ui.activities
    ui, _ = asyncio.run(go(DENY))
    assert "command decline" in "".join(ui.texts)


def test_unsupported_server_requests_are_refused(fake_bin, codex_log) -> None:
    async def go():
        a = agent(fake_bin)
        ui = ScriptedUI()
        await a.start()
        result = await a.run_turn("ask me", ui)
        await a.close()
        return ui, result

    ui, result = asyncio.run(go())
    assert result.status == "completed"
    reply = [m for m in map(json.loads, codex_log.read_text().splitlines()) if m.get("id") == "srv-2"][0]
    assert "not supported" in reply["error"]["message"]
    assert any("item/tool/requestUserInput" in n for n in ui.notices)


def test_interrupt(fake_bin, codex_log) -> None:
    async def go():
        a = agent(fake_bin)
        ui = ScriptedUI()
        await a.start()
        task = asyncio.create_task(a.run_turn("slow one", ui))
        await asyncio.sleep(0.3)
        await a.interrupt()
        result = await task
        await a.close()
        return result

    result = asyncio.run(go())
    assert result.status == "interrupted"
    assert sent(codex_log, "turn/interrupt")[0]["params"] == {"threadId": "thr-1", "turnId": "turn-1"}


def test_a_lost_server_is_recovered_on_the_same_thread(fake_bin, codex_log) -> None:
    async def go():
        a = agent(fake_bin)
        ui = ScriptedUI()
        await a.start(model="gpt-5.5")
        await a.set_model("gpt-5-mini")
        lost = await a.run_turn("crash now", ui)
        await a.recover()
        after = await a.run_turn("hello", ui)
        await a.close()
        return lost, after

    lost, after = asyncio.run(go())
    assert lost.status == "failed" and lost.transport_lost
    resume = sent(codex_log, "thread/resume")[0]["params"]
    assert resume["threadId"] == "thr-1" and resume["model"] == "gpt-5-mini"
    assert after.status == "completed" and after.models == ["gpt-5-mini"]


def test_a_reroute_is_what_ran(fake_bin, codex_log) -> None:
    async def go():
        a = agent(fake_bin)
        ui = ScriptedUI()
        await a.start(model="gpt-5.5")
        result = await a.run_turn("reroute please", ui)
        await a.close()
        return a, ui, result

    a, ui, result = asyncio.run(go())
    assert result.models == ["gpt-5.5", "gpt-5.4-mini"] and a.model == "gpt-5.4-mini"
    assert any("rerouted" in n for n in ui.notices)


def test_prompt_routing_is_off_for_the_agent(fake_bin) -> None:
    assert agent(fake_bin).env["ACTROUTER_PROMPT_ROUTING"] == "off"
