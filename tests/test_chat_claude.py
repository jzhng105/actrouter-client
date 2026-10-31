import asyncio
from types import SimpleNamespace

import pytest

sdk = pytest.importorskip("claude_agent_sdk")

from actrouter_client.chat.base import APPROVE, APPROVE_SESSION, CANCEL, DENY  # noqa: E402
from actrouter_client.chat.claude import ClaudeAgent  # noqa: E402
from actrouter_client.chat.ui import ScriptedUI  # noqa: E402


def usage(inp, read, write, out):
    return {"input_tokens": inp, "cache_read_input_tokens": read, "cache_creation_input_tokens": write,
            "output_tokens": out}


class FakeClient:
    """ClaudeSDKClient's surface, answering every query with `script(model, prompt)`."""

    instances: list["FakeClient"] = []

    def __init__(self, options):
        self.options = options
        self.calls: list[tuple] = []
        self.model = options.model
        FakeClient.instances.append(self)

    async def connect(self):
        self.calls.append(("connect",))

    async def set_model(self, model):
        self.calls.append(("set_model", model))
        self.model = model

    async def query(self, prompt):
        self.calls.append(("query", prompt))
        self.prompt = prompt

    async def interrupt(self):
        self.calls.append(("interrupt",))

    async def disconnect(self):
        self.calls.append(("disconnect",))

    async def receive_response(self):
        model = {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-5-5"}.get(self.model, self.model or "claude-opus-5-5")
        if "tool" in self.prompt:
            decision = await self.options.can_use_tool(
                "Bash", {"command": "rm -rf build"},
                SimpleNamespace(suggestions=[{"type": "addRules"}], tool_use_id="t1"))
            self.decision = decision
        yield sdk.SystemMessage("init", {"session_id": "sess-1", "model": model})
        yield sdk.StreamEvent("u1", "sess-1", {"type": "content_block_delta",
                                               "delta": {"type": "text_delta", "text": "hi "}})
        yield sdk.StreamEvent("u2", "sess-1", {"type": "content_block_delta",
                                               "delta": {"type": "text_delta", "text": "sub"}}, "toolu_1")
        # a response repeated once per content block, the later copy with more output
        yield sdk.AssistantMessage([sdk.ToolUseBlock("t1", "Read", {"file_path": "a.py"})], model,
                                   usage=usage(10, 20000, 3000, 5), message_id="m1")
        yield sdk.AssistantMessage([sdk.TextBlock("reading")], model, usage=usage(10, 20000, 3000, 80),
                                   message_id="m1")
        yield sdk.AssistantMessage([sdk.TextBlock("subagent text")], "claude-haiku-5-5",
                                   parent_tool_use_id="toolu_1", usage=usage(1, 1, 1, 1), message_id="s1")
        if "compact" in self.prompt:
            yield sdk.SystemMessage("compact_boundary", {})
        yield sdk.AssistantMessage([sdk.TextBlock("all done")], model, usage=usage(5, 23000, 500, 120),
                                   message_id="m2")
        yield sdk.StreamEvent("u3", "sess-1", {"type": "message_stop"})
        self.cost = getattr(self, "cost", 0.0) + 0.0421  # a running total, as Claude Code reports it
        yield sdk.ResultMessage(
            "success", 10, 9, False, 2, "sess-1", total_cost_usd=self.cost,
            model_usage={model: {"inputTokens": 15, "outputTokens": 200, "cacheReadInputTokens": 43000,
                                 "cacheCreationInputTokens": 3500, "webSearchRequests": 0, "costUSD": 0.04,
                                 "contextWindow": 200000, "maxOutputTokens": 64000}},
            terminal_reason="aborted_streaming" if "stop" in self.prompt else "completed",
        )


def agent(**kw) -> ClaudeAgent:
    FakeClient.instances.clear()
    return ClaudeAgent(client_factory=FakeClient, **kw)


def test_set_model_before_the_prompt_on_one_client() -> None:
    async def go():
        a = agent(effort="high")
        ui = ScriptedUI()
        await a.start(model="opus")
        first = await a.run_turn("hello", ui)
        await a.set_model("sonnet")
        await a.set_model("sonnet")  # already on it: no second call
        second = await a.run_turn("again", ui)
        await a.close()
        return a, ui, first, second

    a, ui, first, second = asyncio.run(go())
    (client,) = FakeClient.instances
    assert client.calls == [("connect",), ("query", "hello"), ("set_model", "sonnet"), ("query", "again"),
                            ("disconnect",)]
    opts = client.options
    assert opts.model == "opus" and opts.effort == "high" and opts.include_partial_messages
    assert opts.env["ACTROUTER_PROMPT_ROUTING"] == "off"
    assert a.session_id == "sess-1"
    assert first.status == "completed" and second.models == ["claude-sonnet-5"]
    # main-session requests only, each message id once, with its largest output
    assert [(r.prompt, r.cached, r.cache_write, r.output) for r in first.requests] == [
        (23010, 20000, 3000, 80), (23505, 23000, 500, 120)]
    assert first.cost_reported_usd == 0.0421 and first.context_window == 200000
    assert second.cost_reported_usd == pytest.approx(0.0421)  # the turn's share of the running total
    assert first.reply == "all done"
    assert "".join(ui.texts) == "hi \n" * 2  # the subagent's stream is not shown
    assert ui.activities == ["Read: a.py"] * 2


@pytest.mark.parametrize("answer,behavior", [(APPROVE, "allow"), (APPROVE_SESSION, "allow"),
                                             (DENY, "deny"), (CANCEL, "deny")])
def test_tool_permissions_go_to_the_person(answer, behavior) -> None:
    async def go():
        a = agent()
        ui = ScriptedUI()

        async def approve(kind, summary, **kw):
            ui.approvals.append((kind, summary))
            return answer

        ui.approve = approve
        await a.start()
        await a.run_turn("use a tool", ui)
        return ui

    ui = asyncio.run(go())
    decision = FakeClient.instances[0].decision
    assert ui.approvals == [("tool", "Bash: rm -rf build")]
    assert decision.behavior == behavior
    if answer == APPROVE_SESSION:
        assert decision.updated_permissions == [{"type": "addRules"}]
    if answer == CANCEL:
        assert decision.interrupt


def test_interrupted_and_compacted_turns() -> None:
    async def go():
        a = agent()
        await a.start()
        stopped = await a.run_turn("stop", ScriptedUI())
        compacted = await a.run_turn("compact", ScriptedUI())
        await a.interrupt()
        return stopped, compacted

    stopped, compacted = asyncio.run(go())
    assert stopped.status == "interrupted"
    assert compacted.compacted
    assert ("interrupt",) in FakeClient.instances[0].calls


def test_a_dead_cli_is_recovered_by_resuming_the_session() -> None:
    class Dying(FakeClient):
        async def receive_response(self):
            raise ConnectionError("CLI exited")
            yield  # pragma: no cover

    async def go():
        a = ClaudeAgent(client_factory=Dying)
        await a.start(model="sonnet")
        a.session_id = "sess-9"
        lost = await a.run_turn("hello", ScriptedUI())
        a._factory = FakeClient
        await a.recover()
        return lost

    FakeClient.instances.clear()
    lost = asyncio.run(go())
    assert lost.status == "failed" and lost.transport_lost and "CLI exited" in lost.error
    resumed = FakeClient.instances[-1].options
    assert resumed.resume == "sess-9" and resumed.model == "sonnet"
