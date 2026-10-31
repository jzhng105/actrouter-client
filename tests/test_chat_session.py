import asyncio
import json
from pathlib import Path

import pytest

from actrouter_client import routing_policy as policy
from actrouter_client.advisor import Advisor
from actrouter_client.chat import repl
from actrouter_client.chat.base import RequestUsage, TurnResult
from actrouter_client.chat.session import FRESH_CONTEXT_TOKENS, ChatConfig, RoutedSession
from actrouter_client.chat.ui import ScriptedUI
from actrouter_client.client import ActRouterError
from actrouter_client.session_state import ChatState, Transitions

CATALOG = json.loads((Path(__file__).parent / "fixtures" / "api_replay.json").read_text())[
    "GET /v1/models null"]["body"]["data"]

#: words in a prompt -> (chosen model, task type, confidence)
ROUTES = {
    "search": ("claude-haiku-5-5", "code_search", 0.95),
    "debug": ("claude-opus-5-5", "debugging", 0.95),
    "hello": ("claude-haiku-5-5", "chat", 0.9),
}


class StubRouter:
    where = "stub"

    def __init__(self, fail=False):
        self.fail = fail
        self.tasks: list[str] = []

    def list_models(self):
        return CATALOG

    def route_task(self, task, *, preference="balanced", candidate_models=None, top_k=3, baseline_model=None, **kw):
        if self.fail:
            raise ActRouterError("router down", 503)
        text = task["description"]
        self.tasks.append(text)
        chosen, task_type, conf = next(
            (v for k, v in ROUTES.items() if k in text.lower()), ("claude-haiku-4-5", "unknown", 0.0))
        return {"chosen_model": chosen, "classification": {"task_type": task_type, "confidence": conf},
                "ranking": [{"model_id": chosen, "reasons": []}], "request_id": f"r{len(self.tasks)}"}

    def route_function(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError

    def list_task_types(self):  # pragma: no cover
        return {}


SPELLED = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}


class FakeAgent:
    """Runs turns on whatever model it was last set to. Each turn makes
    `rounds` requests that grow the context by 2000 tokens each."""

    name, host = "claude", "claude-code"

    def __init__(self, model="opus", rounds=3, fail_on=None):
        self.model, self.rounds, self.fail_on = model, rounds, fail_on
        self.session_id = None
        self.set_calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []
        self.context = 30_000
        self.recovered = 0

    async def start(self, *, model=None, resume=None):
        self.session_id = resume or "sess-1"
        self.model = model or self.model

    async def set_model(self, model):
        self.set_calls.append(model)
        self.model = model

    async def run_turn(self, prompt, ui):
        served = SPELLED.get(self.model, self.model)
        self.prompts.append((prompt, served))
        if self.fail_on and self.model == self.fail_on:
            return TurnResult(status="failed", error="thinking blocks rejected", models=[served])
        reqs = []
        for _ in range(self.rounds):
            reqs.append(RequestUsage(prompt=self.context, cached=self.context - 2000, cache_write=2000, output=300))
            self.context += 2000
        return TurnResult(status="completed", models=[served], requests=reqs, context_window=200_000,
                          reply="Shall I search the codebase for it?")

    async def interrupt(self):
        pass

    async def recover(self):
        self.recovered += 1

    async def close(self):
        pass


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTROUTER_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def turns(ledger_file):
    if not ledger_file.exists():
        return []
    return [e for e in map(json.loads, ledger_file.read_text().splitlines()) if e["kind"] == "turn"]


def session(mode="on", agent=None, router=None, tested=True, **cfg):
    agent = agent or FakeAgent()
    advisor = Advisor(router or StubRouter(), host="claude-code")
    transitions = Transitions("claude")
    if tested:
        for a in ("claude-opus-5-5", "claude-haiku-5-5", "claude-sonnet-5-5"):
            for b in ("claude-opus-5-5", "claude-haiku-5-5", "claude-sonnet-5-5"):
                if a != b:
                    transitions.record(a, b, True, source="test")
    clock = iter(range(1_000, 1_000_000, 200))  # 200 s between calls: no cooldown by time
    s = RoutedSession(agent, advisor, ScriptedUI(), ChatConfig(mode=mode, **cfg), transitions=transitions,
                      clock=lambda: float(next(clock)))
    return s, agent


def test_shadow_decides_and_records_but_never_switches(state_dir, ledger_file) -> None:
    s, agent = session(mode="shadow")

    async def go():
        await s.start()
        return await s.turn("search the repo for the retry policy")

    result = asyncio.run(go())
    assert result.status == "completed" and agent.set_calls == []
    (rec,) = turns(ledger_file)
    assert rec["reason"] == "switch" and rec["shadow"] is True and rec["switched"] is False
    assert rec["recommended_model"] == "claude-haiku-5-5" and rec["applied_model"] == "claude-opus-5-5"
    assert rec["gate"]["context_tokens"] == FRESH_CONTEXT_TOKENS
    assert rec["served_model"] == "claude-opus-5-5" and rec["requests"] == 3
    assert rec["cost_computed_usd"] > 0 and "cost_reported_usd" not in rec
    assert "retry" not in json.dumps(rec)  # no prompt text in the ledger
    assert any("shadow: would switch to haiku" in n for n in s.ui.notices)


def test_on_switches_the_main_model_before_the_turn(state_dir, ledger_file) -> None:
    s, agent = session()

    async def go():
        await s.start()
        await s.turn("search the repo for the retry policy")
        await s.turn("debug why the auth test fails")  # cooldown: one turn since the switch
        await s.turn("yes, go ahead")  # a follow-up: stays on the model that runs
        await s.turn("debug it again")  # two turns on haiku: the cooldown is over

    asyncio.run(go())
    assert agent.set_calls == ["haiku", "opus"]
    assert [m for _, m in agent.prompts] == ["claude-haiku-5-5"] * 3 + ["claude-opus-5-5"]
    recs = turns(ledger_file)
    assert [r["reason"] for r in recs] == ["switch", "gate_cooldown", "follow_up", "switch"]
    assert recs[0]["switched"] and recs[0]["applied_host_model"] == "haiku"
    assert recs[1]["recommended_model"] == "claude-opus-5-5" and recs[1]["applied_model"] == "claude-haiku-5-5"
    assert recs[2]["routed_on"] == "pinned" and recs[2]["applied_model"] == "claude-haiku-5-5"
    assert recs[3]["gate"]["direction"] == "up"
    st = ChatState.load("sess-1", "claude")
    assert st.current_model == "claude-opus-5-5" and st.last_switch_turn == 3 and st.turn == 4
    # haiku's cache stays on record: switching back soon can reuse it
    assert set(st.caches) == {"claude-haiku-5-5", "claude-opus-5-5"}
    assert st.caches["claude-opus-5-5"]["prefix"] == agent.context - 2000
    assert st.observed["turns"] == 4 and st.observed["rounds"] == pytest.approx(3)
    assert st.pin["model"] == "claude-opus-5-5" and st.pin["applied_model"] == "claude-opus-5-5"
    assert Transitions("claude").entries()["claude-opus-5-5 -> claude-haiku-5-5"]["source"] == "runtime"
    assert st.to_json().get("last_prompt") is None  # prompt text is never written


def test_a_rejected_recommendation_pins_the_model_that_runs(state_dir, ledger_file) -> None:
    s, agent = session(tested=False)

    async def go():
        await s.start()
        await s.turn("search the repo for the retry policy")
        await s.turn("yes please")

    asyncio.run(go())
    assert agent.set_calls == []
    first, second = turns(ledger_file)
    assert first["reason"] == "transition_untested" and first["recommended_model"] == "claude-haiku-5-5"
    assert second["reason"] == "follow_up" and second["applied_model"] == "claude-opus-5-5"
    assert ChatState.load("sess-1", "claude").pin["applied_model"] == "claude-opus-5-5"


def test_a_failed_switch_goes_back_and_marks_the_transition(state_dir, ledger_file) -> None:
    s, agent = session(agent=FakeAgent(fail_on="haiku"))

    async def go():
        await s.start()
        return await s.turn("search the repo for the retry policy")

    result = asyncio.run(go())
    assert result.status == "failed"
    assert agent.set_calls == ["haiku", "opus"]
    assert Transitions("claude").known("claude-opus-5-5", "claude-haiku-5-5") is False
    st = ChatState.load("sess-1", "claude")
    assert st.current_model == "claude-opus-5-5" and st.last_switch_turn is None
    assert turns(ledger_file)[0]["switched"] is False


def test_routing_failure_keeps_the_model(state_dir, ledger_file) -> None:
    s, agent = session(router=StubRouter(fail=True))

    async def go():
        await s.start()
        await s.turn("search the repo")
        await s.turn("search again")

    asyncio.run(go())
    assert agent.set_calls == [] and [r["reason"] for r in turns(ledger_file)] == ["no_route"] * 2
    assert sum("routing unavailable" in n for n in s.ui.notices) == 1


def test_off_does_not_route(state_dir, ledger_file) -> None:
    router = StubRouter()
    s, agent = session(mode="off", router=router)

    async def go():
        await s.start()
        await s.turn("search the repo")

    asyncio.run(go())
    assert router.tasks == [] and turns(ledger_file)[0]["reason"] == "no_route"


def test_a_follow_up_with_nothing_pinned_routes_on_the_exchange(state_dir, ledger_file) -> None:
    router = StubRouter()
    s, agent = session(router=router)

    async def go():
        await s.start()
        s.state.last_prompt, s.state.last_reply = "where is it set?", "Shall I search the codebase for it?"
        await s.turn("yes please")

    asyncio.run(go())
    assert router.tasks[-1].startswith("yes please Shall I search")
    rec = turns(ledger_file)[0]
    assert rec["reason"] == "follow_up" and rec["recommended_model"] == "claude-haiku-5-5"
    assert agent.set_calls == []


def test_resume_restores_the_routing_state(state_dir, ledger_file) -> None:
    s, agent = session()

    async def go():
        await s.start()
        await s.turn("search the repo for the retry policy")

    asyncio.run(go())
    before = ChatState.load("sess-1", "claude")
    s2, agent2 = session(agent=FakeAgent(model="haiku"))

    async def again():
        await s2.start(resume="sess-1")
        await s2.turn("search for the loader")  # cooldown still counts from the earlier switch

    asyncio.run(again())
    after = ChatState.load("sess-1", "claude")
    assert after.turn == before.turn + 1 and after.last_switch_turn == before.last_switch_turn
    assert turns(ledger_file)[-1]["reason"] == "already_on"


def test_the_persons_model_choice_wins(state_dir, ledger_file) -> None:
    s, agent = session()

    async def go():
        await s.start()
        await s.set_manual_model("sonnet")
        await s.turn("search the repo for the retry policy")
        await s.set_manual_model(None)
        s.state.last_switch_turn = -10
        await s.turn("search the repo for the loader")

    asyncio.run(go())
    assert agent.set_calls == ["sonnet", "haiku"]
    assert [r["reason"] for r in turns(ledger_file)] == ["manual_model", "switch"]


def test_an_explicit_start_model_is_manual(state_dir) -> None:
    s, agent = session()
    asyncio.run(s.start(model="sonnet"))
    assert s.state.manual and agent.model == "sonnet"


def test_on_mode_starts_a_new_session_on_the_first_prompts_model(state_dir) -> None:
    s, agent = session()
    asyncio.run(s.start(first_prompt="search the repo for the retry policy"))
    assert agent.model == "haiku" and not s.state.manual


def test_a_lost_connection_is_recovered(state_dir) -> None:
    class Losing(FakeAgent):
        async def run_turn(self, prompt, ui):
            return TurnResult(status="failed", error="gone", transport_lost=True)

    s, agent = session(agent=Losing())

    async def go():
        await s.start()
        await s.turn("hello there")

    asyncio.run(go())
    assert agent.recovered == 1


def test_repl_commands(state_dir, ledger_file) -> None:
    s, agent = session(mode="shadow")
    s.ui.inputs = ["/route on", "/status", "/route nonsense", "search the repo for X", "/model haiku",
                   "/model auto", "/help", "/quit", "never read"]

    async def go():
        await s.start()
        return await repl.run(s)

    assert asyncio.run(go()) == 0
    assert s.cfg.mode == "on" and s.ui.inputs == ["never read"]
    assert any(n.startswith("session  sess-1") for n in s.ui.notices)
    assert any("usage: /route" in n for n in s.ui.notices)
    assert any("resume with: actrouter-route chat claude --resume sess-1" in n for n in s.ui.notices)
    assert agent.prompts[0][0] == "search the repo for X" and not s.state.manual


def test_decision_reasons_are_known() -> None:
    assert {"switch", "follow_up", "gate_savings", "gate_context", "gate_cooldown",
            "transition_untested"} <= policy.REASONS


def test_the_report_summarizes_chat_turns(state_dir, ledger_file) -> None:
    from actrouter_client.report import format_text, load

    s, agent = session(mode="shadow")

    async def go():
        await s.start()
        await s.turn("search the repo for the retry policy")
        await s.turn("hello there")

    asyncio.run(go())
    summary = load(0, ledger_file, CATALOG)
    chat = summary["chat"]
    assert chat["turns"] == 2 and chat["switched"] == 0 and chat["would_switch_in_shadow"] >= 1
    assert chat["cost_computed_usd"] > 0 and chat["estimated_savings_of_switches_usd"] > 0
    assert "chat turns          2" in format_text(summary)


def test_an_unreachable_router_fails_open(state_dir, ledger_file) -> None:
    class Down(StubRouter):
        def list_models(self):
            raise ActRouterError("could not reach ActRouter", None)

    s, agent = session(router=Down(fail=True))

    async def go():
        await s.start()
        return await s.turn("search the repo")

    assert asyncio.run(go()).status == "completed"
    rec = turns(ledger_file)[0]
    assert rec["reason"] == "no_route" and "cost_computed_usd" not in rec and agent.set_calls == []


def test_the_policy_prices_what_the_host_really_runs(state_dir, ledger_file) -> None:
    # Claude Code spells claude-haiku-4-5 `haiku`, which runs the newest Haiku
    s, agent = session()
    ROUTES["older"] = ("claude-haiku-4-5", "code_search", 0.95)
    try:
        async def go():
            await s.start()
            await s.turn("search the older modules")

        asyncio.run(go())
    finally:
        del ROUTES["older"]
    rec = turns(ledger_file)[0]
    assert rec["recommended_model"] == "claude-haiku-5-5" and rec["recommended_host_model"] == "haiku"


def test_the_first_prompt_is_routed_once(state_dir) -> None:
    router = StubRouter()
    s, agent = session(router=router)

    async def go():
        await s.start(first_prompt="search the repo for the retry policy")
        await s.turn("search the repo for the retry policy")

    asyncio.run(go())
    assert router.tasks == ["search the repo for the retry policy"]
    assert s.last_decision.reason == "already_on"


def test_a_refused_switch_keeps_the_session_and_runs_the_prompt(state_dir, ledger_file) -> None:
    class Refusing(FakeAgent):
        async def set_model(self, model):
            self.set_calls.append(model)
            raise RuntimeError(f"model {model} is not available")

    s, agent = session(agent=Refusing())

    async def go():
        await s.start()
        return await s.turn("search the repo for the retry policy")

    result = asyncio.run(go())
    assert result.status == "completed" and agent.prompts == [("search the repo for the retry policy",
                                                               "claude-opus-5-5")]
    assert Transitions("claude").known("claude-opus-5-5", "claude-haiku-5-5") is False
    rec = turns(ledger_file)[0]
    assert rec["switched"] is False and "not available" in rec["switch_error"]
    assert ChatState.load("sess-1", "claude").current_model == "claude-opus-5-5"


def test_a_switch_lost_with_the_connection_is_undone_after_reconnecting(state_dir, ledger_file) -> None:
    class Dropping(FakeAgent):
        async def run_turn(self, prompt, ui):
            if self.model == "haiku":
                return TurnResult(status="failed", error="gone", transport_lost=True)
            return await super().run_turn(prompt, ui)

        async def recover(self):
            self.recovered += 1
            self.set_calls.append(f"recover on {self.model}")  # rejoins on the model last selected

    s, agent = session(agent=Dropping())

    async def go():
        await s.start()
        await s.turn("search the repo for the retry policy")
        await s.turn("yes, go ahead")

    asyncio.run(go())
    assert agent.set_calls == ["haiku", "recover on haiku", "opus"]
    assert agent.model == "opus" and agent.prompts[-1] == ("yes, go ahead", "claude-opus-5-5")
    assert ChatState.load("sess-1", "claude").current_model == "claude-opus-5-5"
    assert Transitions("claude").known("claude-opus-5-5", "claude-haiku-5-5") is False
