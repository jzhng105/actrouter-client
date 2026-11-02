"""One routed agent session: route each prompt, decide, switch, run, record.

:class:`RoutedSession` sits between the person and an
:class:`~actrouter_client.chat.base.Agent`. For every prompt it

1. routes the prompt with ActRouter (failing open: no answer keeps the model);
2. picks the routing the prompt follows (its own, the pinned one, or the
   exchange it continues: :func:`~actrouter_client.routing_policy.choose_route`).
   A longer prompt the router cannot read is a request of its own and is not
   routed; a question about the agent's own work is not routed and keeps the pin;
3. decides whether to switch (:func:`~actrouter_client.routing_policy.decide`);
4. switches through the agent's own call when the decision says so;
5. runs the turn, then folds what it used into the session's state
   (:class:`~actrouter_client.session_state.ChatState`) and records one
   ``turn`` event in the routing ledger, which holds both the decision and the
   outcome.

Routing modes: ``shadow`` (the default) decides and records but never
switches; ``on`` switches; ``off`` does not route at all.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from actrouter_client import ledger
from actrouter_client import routing_policy as policy
from actrouter_client import transcript
from actrouter_client.advisor import Advisor
from actrouter_client.chat.base import Agent, TurnResult, UI
from actrouter_client.session_state import ChatState, Transitions

MODES = ("shadow", "on", "off")

#: windows assumed for a model the agent has not reported one for: agents run
#: models on less than their API window unless configured otherwise
HOST_WINDOW_CAPS = {"claude-code": 200_000, "codex": 272_000}

#: a fresh session's first request: the agent's system prompt and tool definitions
FRESH_CONTEXT_TOKENS = 25_000


def estimate_tokens(text: str) -> int:
    """A deliberately high token estimate for text (about 3 characters per
    token; tokenizers differ)."""
    return math.ceil(len(text) / 3)


@dataclass
class ChatConfig:
    mode: str = "shadow"
    preference: str = "balanced"
    #: prompt characters sent to ActRouter (0 keeps prompts on the machine,
    #: which turns routing off)
    prompt_chars: int = 600
    min_confidence: float = 0.5
    policy: policy.PolicyConfig = field(default_factory=policy.PolicyConfig)
    window_cap: Optional[int] = None


class RoutedSession:
    def __init__(
        self,
        agent: Agent,
        advisor: Advisor,
        ui: UI,
        cfg: Optional[ChatConfig] = None,
        *,
        transitions: Optional[Transitions] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.agent = agent
        self.advisor = advisor
        self.ui = ui
        self.cfg = cfg or ChatConfig()
        self.transitions = transitions or Transitions(agent.name)
        self.clock = clock
        self.state: Optional[ChatState] = None
        self.last_decision: Optional[policy.Decision] = None
        self._routing_failed = False
        #: the first prompt's routing, made at start: not asked for twice
        self._prerouted: Optional[tuple[str, Optional[dict[str, Any]]]] = None

    # --- lifecycle -------------------------------------------------------------------

    async def start(self, *, model: Optional[str] = None, resume: Optional[str] = None,
                    first_prompt: Optional[str] = None) -> None:
        """Start (or resume) the agent's session. A model given explicitly
        is the person's choice and stays until ``/model auto``. A new
        session with `first_prompt` in ``on`` mode starts on the model that
        prompt routes to: a new session has no cache to lose."""
        initial = model
        if not model and not resume and first_prompt and self.cfg.mode == "on":
            advice = self._route(first_prompt[: self._chars()], baseline=None)
            self._prerouted = (first_prompt, advice)
            if advice:
                initial = advice["host_model"]
                self.ui.notice(f"{advice.get('task_type')} -> starting on {initial}")
        await self.agent.start(model=initial, resume=resume)
        sid = self.agent.session_id or f"pending-{int(self.clock())}"
        self.state = ChatState.load(sid, self.agent.name) if resume else ChatState(sid, self.agent.name)
        if not resume:
            self.state.context_tokens = FRESH_CONTEXT_TOKENS
        self.state.manual = bool(model) or self.state.manual
        self._set_current(self.agent.model)

    async def close(self) -> None:
        await self.agent.close()

    def _set_current(self, host_model: Optional[str]) -> None:
        assert self.state is not None
        if host_model:
            self.state.current_host_model = host_model
            self.state.current_model = self.advisor.baseline_id(host_model) or host_model

    def _catalog(self) -> dict[str, Any]:
        """The model catalog by id; empty when ActRouter cannot be reached,
        which makes every price unknown, so the model stays."""
        try:
            return {str(m.get("id")): m for m in self.advisor.catalog()}
        except Exception:  # noqa: BLE001 - routing is advisory: never end the session
            return {}

    def _chars(self) -> int:
        return max(0, min(self.cfg.prompt_chars, 2000))

    # --- routing ---------------------------------------------------------------------

    def _route(self, text: str, baseline: Optional[str]) -> Optional[dict[str, Any]]:
        advice = self._ask(text, baseline)
        if advice is None:
            return None
        # What the host really runs for this spelling: Claude Code's alias
        # `haiku` runs the newest Haiku, whichever Haiku was chosen
        runs = self.advisor.baseline_id(advice.get("host_model"))
        if runs and runs != advice.get("model"):
            advice = {**advice, "routed_model": advice.get("model"), "model": runs}
        return advice

    def _ask(self, text: str, baseline: Optional[str]) -> Optional[dict[str, Any]]:
        try:
            advice = self.advisor.for_task(text, preference=self.cfg.preference, top_k=1, baseline=baseline)
        except Exception as exc:  # noqa: BLE001 - routing is advisory: whatever it raises, keep the model
            if not self._routing_failed:
                self.ui.notice(f"routing unavailable, keeping the model: {exc}")
                self._routing_failed = True
            return None
        self._routing_failed = False
        return advice if policy.usable(advice, self.cfg.min_confidence) else None

    def decide(self, prompt: str) -> policy.Decision:
        assert self.state is not None
        st = self.state
        current_host, current = st.current_host_model, st.current_model
        chars = self._chars()
        if self.cfg.mode == "off" or chars == 0:
            return policy.Decision(policy.NO_ROUTE, current_model=current, recommended_model=None,
                                   recommended_host_model=None, applied_model=current,
                                   applied_host_model=current_host)
        if policy.about_own_work(prompt):
            # A question about the agent's own work ("did you run the
            # tests?"): only this model can answer, so it stays, and the
            # conversation keeps its pin.
            self._prerouted = None
            return policy.Decision(policy.CONVERSATION, current_model=current, recommended_model=None,
                                   recommended_host_model=None, applied_model=current,
                                   applied_host_model=current_host)
        if self._prerouted and self._prerouted[0] == prompt:
            own = self._prerouted[1]
        else:
            own = self._route(prompt[:chars], current_host)
        self._prerouted = None
        advice, routed_on = policy.choose_route(
            own, st.pin if own is None else None,
            baseline=current_host, baseline_id=current, prompt=prompt,
        )
        if advice is None and routed_on == policy.ROUTED_ON_EXCHANGE:
            text = policy.continuation_text(prompt, st.last_prompt, st.last_reply, chars, transcript.compress)
            advice = self._route(text, current_host) if text else None
        catalog = self._catalog()
        view = policy.SessionView(
            current_model=current,
            context_tokens=st.context_tokens,
            now=self.clock(),
            caches=dict(st.caches),
            windows=dict(st.windows),
            turn=st.turn,
            last_switch_turn=st.last_switch_turn,
            last_switch_t=st.last_switch_t,
            observed=dict(st.observed) if st.observed else None,
        )
        decision = policy.decide(
            advice, routed_on if advice or routed_on == policy.ROUTED_ON_NONE else None, view, catalog,
            new_prompt_tokens=estimate_tokens(prompt),
            cfg=self.cfg.policy,
            transition_ok=self.transitions.known,
            window_cap=self.cfg.window_cap or HOST_WINDOW_CAPS.get(self.agent.host),
            shadow=self.cfg.mode == "shadow",
            manual=st.manual,
        )
        if decision.applied_model == current:
            decision.applied_host_model = current_host
        if advice is not None and routed_on != policy.ROUTED_ON_PIN:
            st.pin = policy.pin(
                advice, baseline=current_host, baseline_source="session model",
                stays=advice.get("model") == current, applied_model=decision.applied_model,
            )
        return decision

    # --- a turn ----------------------------------------------------------------------

    async def turn(self, prompt: str) -> TurnResult:
        assert self.state is not None
        st = self.state
        decision = self.decide(prompt)
        self.last_decision = decision
        before, before_host = st.current_model, st.current_host_model
        switched = ok = False
        switch_error: Optional[str] = None
        if decision.switches and decision.applied_host_model:
            try:
                await self.agent.set_model(decision.applied_host_model)
            except Exception as exc:  # noqa: BLE001 - the agent refused the model: keep the session
                switch_error = str(exc) or type(exc).__name__
                if before and decision.applied_model:
                    self.transitions.record(before, decision.applied_model, False, source="runtime",
                                            error=switch_error, at=int(self.clock()))
                self.ui.notice(f"the agent refused {decision.applied_host_model} ({switch_error}); "
                               f"staying on {before_host}, and this transition will not be used again")
            else:
                switched = True
                self.ui.notice(f"{decision.task_type} -> {decision.applied_host_model} (was {before_host})")
        elif decision.reason == policy.SWITCH and decision.shadow:
            self.ui.notice(f"shadow: would switch to {decision.recommended_host_model} for "
                           f"{decision.task_type}; staying on {before_host}")

        result = await self.agent.run_turn(prompt, self.ui)
        now = self.clock()

        if switched:
            ok = result.status != "failed" or bool(result.requests)
            if before and decision.applied_model:
                self.transitions.record(before, decision.applied_model, ok, source="runtime",
                                        error=None if ok else result.error, at=int(now))
            if ok:
                st.switched(decision.applied_model or "", decision.applied_host_model, now)

        if result.transport_lost:
            self.ui.notice("lost the connection to the agent; reconnecting to the same session")
            try:
                await self.agent.recover()
            except Exception as exc:  # noqa: BLE001
                self.ui.notice(f"could not reconnect: {exc}")

        reverted = False
        if switched and not ok:
            # after any reconnection: recover() rejoins on the model last selected
            if before_host:
                try:
                    await self.agent.set_model(before_host)
                    reverted = True
                except Exception as exc:  # noqa: BLE001
                    self.ui.notice(f"could not switch back to {before_host}: {exc}")
            self.ui.notice(f"the switch to {decision.applied_host_model} failed ({result.error}); "
                           + (f"back on {before_host}" if reverted else f"still on {self.agent.model}")
                           + ", and this transition will not be used again")

        served = result.models[-1] if result.models else self.agent.model
        if switched and not ok:
            # the state follows the model the agent now runs on
            served, switched = (before_host if reverted else self.agent.model), False
        served_id = self.advisor.baseline_id(served) if served else None
        st.observe_turn(
            model=served_id or st.current_model,
            prompts=[r.prompt for r in result.requests],
            outputs=[r.output for r in result.requests],
            compacted=result.compacted,
            window=result.context_window,
            now=now,
        )
        if served:
            st.current_host_model = served if not switched else (decision.applied_host_model or served)
        if self.agent.session_id and self.agent.session_id != st.session_id:
            st.session_id = self.agent.session_id  # known after the first answer (Claude)
        st.last_prompt, st.last_reply = prompt, result.reply
        st.save()
        self._record(decision, result, switched, served_id, switch_error)
        return result

    def _record(self, decision: policy.Decision, result: TurnResult, switched: bool,
                served_id: Optional[str], switch_error: Optional[str] = None) -> None:
        assert self.state is not None
        total = result.total()
        catalog = self._catalog()
        computed = None
        pricing = policy.Pricing.from_catalog(catalog.get(served_id or ""))
        behaviour = policy.CACHE_BEHAVIOUR.get((pricing.family or "") if pricing else "")
        if pricing and behaviour and result.requests:
            computed = round(sum(
                policy.request_cost(pricing, behaviour, prompt=r.prompt, cached=r.cached,
                                    cache_write=r.cache_write, output=r.output)
                for r in result.requests
            ), 6)
        ledger.record(
            "turn",
            surface="chat",
            host=self.agent.host,
            agent=self.agent.name,
            session_id=self.state.session_id,
            turn=self.state.turn,
            mode=self.cfg.mode,
            switched=switched,
            switch_error=switch_error,
            **decision.record(),
            status=result.status,
            error=result.error,
            served_model=served_id,
            models_used=result.models or None,
            requests=len(result.requests),
            prompt_tokens=total.prompt,
            cached_tokens=total.cached,
            cache_write_tokens=total.cache_write,
            output_tokens=total.output,
            reasoning_tokens=total.reasoning or None,
            context_tokens_after=self.state.context_tokens,
            compacted=result.compacted or None,
            #: from tokens and catalog prices, main session only
            cost_computed_usd=computed,
            #: what the agent itself reported (Claude Code: includes subagents)
            cost_reported_usd=result.cost_reported_usd,
        )

    # --- the person's own choices ----------------------------------------------------

    async def set_manual_model(self, host_model: Optional[str]) -> None:
        """``/model <m>``: the person's choice, which routing then leaves
        alone; ``/model auto`` (None) hands the model back to routing."""
        assert self.state is not None
        if host_model is None:
            self.state.manual = False
            self.state.save()
            return
        await self.agent.set_model(host_model)
        self.state.manual = True
        self.state.switched(self.advisor.baseline_id(host_model) or host_model, host_model, self.clock())
        self.state.save()
        ledger.record("model_set", surface="chat", host=self.agent.host, agent=self.agent.name,
                      session_id=self.state.session_id, turn=self.state.turn,
                      applied_model=self.state.current_model, applied_host_model=host_model)
