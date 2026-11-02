import pytest

from actrouter_client import routing_policy as policy
from actrouter_client.routing_policy import Decision, PolicyConfig, Pricing, SessionView, decide


def model(mid, family="anthropic", tier=3, inp=5.0, out=25.0, read=0.5, window=1_000_000, tiers=()):
    return {
        "id": mid, "family": family, "size_tier": tier, "context_window": window,
        "input_price_per_mtok": inp, "output_price_per_mtok": out, "cache_read_price_per_mtok": read,
        "context_tiers": list(tiers), "threshold_inclusive": False,
    }


CATALOG = {m["id"]: m for m in [
    model("big"),
    model("mid", tier=2, inp=3.0, out=15.0, read=0.3),
    model("small", tier=1, inp=1.0, out=5.0, read=0.1, window=200_000),
    model("cheap-long", tier=1, inp=0.1, out=0.5, read=0.01,
          tiers=[{"above_tokens": 100_000, "input_price_per_mtok": 0.5,
                  "output_price_per_mtok": 2.5, "cache_read_price_per_mtok": 0.05}]),
    model("gpt-big", family="openai", inp=5.0, out=30.0, read=0.5),
    model("gpt-small", family="openai", tier=1, inp=0.25, out=2.0, read=0.025, window=272_000),
    model("unpriced", family="anthropic", tier=1, inp=None),
    model("other-vendor", family="nobody", tier=1, inp=0.1, out=0.1, read=0.01),
]}

NOW = 10_000.0
OK = lambda a, b: True  # noqa: E731 - every transition tested


def advice(target, task="code_search", conf=0.9, host=None):
    return {"model": target, "host_model": host or target, "task_type": task, "task_confidence": conf}


def view(current="big", context=50_000, **kw):
    kw.setdefault("caches", {current: {"t": NOW - 30, "prefix": context}} if context else {})
    return SessionView(current_model=current, context_tokens=context, now=NOW, **kw)


def run(adv, v=None, routed_on="prompt", **kw):
    kw.setdefault("transition_ok", OK)
    kw.setdefault("new_prompt_tokens", 200)
    return decide(adv, routed_on, v or view(), CATALOG, **kw)


# --- prices --------------------------------------------------------------------------


def test_long_context_tier_prices_the_whole_request() -> None:
    p = Pricing.from_catalog(CATALOG["cheap-long"])
    assert p.at(50_000) == (0.1, 0.5, 0.01)
    assert p.at(150_000) == (0.5, 2.5, 0.05)


def test_missing_cache_read_price_is_priced_as_input() -> None:
    p = Pricing.from_catalog({**CATALOG["small"], "cache_read_price_per_mtok": None})
    assert p.at(10)[2] == 1.0


def test_request_cost_separates_fresh_cached_and_written_tokens() -> None:
    p, b = Pricing.from_catalog(CATALOG["big"]), policy.CACHE_BEHAVIOUR["anthropic"]
    cost = policy.request_cost(p, b, prompt=1_000_000, cached=500_000, cache_write=250_000, output=0)
    assert cost == pytest.approx(0.25 * 5 + 0.5 * 0.5 + 0.25 * 5 * 1.25)


def test_a_cache_past_its_ttl_offers_nothing() -> None:
    b = policy.CACHE_BEHAVIOUR["anthropic"]
    assert policy.warm_prefix({"t": NOW - 30, "prefix": 9000}, b, NOW) == (9000, b.hit_rate)
    assert policy.warm_prefix({"t": NOW - 301, "prefix": 9000}, b, NOW) == (0, 0)
    assert policy.warm_prefix(None, b, NOW) == (0, 0)


# --- the decision ---------------------------------------------------------------------


def test_a_much_cheaper_model_is_worth_its_cold_cache() -> None:
    d = run(advice("cheap-long"))
    assert d.reason == policy.SWITCH and d.switches
    assert d.applied_model == "cheap-long"
    assert d.gate["switch_cost_usd"] < d.gate["stay_cost_usd"]
    assert d.gate["workload"]["basis"] == "prior"


def test_a_warm_cache_on_a_big_context_beats_a_cold_cheaper_model() -> None:
    # 120k tokens cached on mid; small must write them all at 1.25x input
    d = run(advice("small", task="chat"), view("mid", 120_000))
    assert d.reason == policy.GATE_SAVINGS and not d.switches
    assert d.recommended_model == "small" and d.applied_model == "mid"
    assert d.gate["stay_cost_usd"] < d.gate["switch_cost_usd"]


def test_the_current_models_cold_cache_is_counted_too() -> None:
    warm = run(advice("small", task="chat"), view("mid", 120_000))
    cold = run(advice("small", task="chat"), view("mid", 120_000, caches={}))
    assert cold.gate["stay_cost_usd"] > warm.gate["stay_cost_usd"]
    assert cold.gate["current_cache_prefix"] == 0


def test_switching_back_can_reuse_a_cache_that_has_not_expired() -> None:
    caches = {"big": {"t": NOW - 10, "prefix": 80_000}, "small": {"t": NOW - 60, "prefix": 79_000}}
    back = run(advice("small", task="qa"), view("big", 80_000, caches=caches))
    gone = run(advice("small", task="qa"), view("big", 80_000, caches={"big": caches["big"]}))
    assert back.gate["target_cache_prefix"] == 79_000
    assert back.gate["switch_cost_usd"] < gone.gate["switch_cost_usd"]


def test_openai_has_no_write_premium_but_a_lower_hit_rate() -> None:
    oa, an = policy.CACHE_BEHAVIOUR["openai"], policy.CACHE_BEHAVIOUR["anthropic"]
    assert oa.write_multiplier == 1.0 < an.write_multiplier
    assert oa.hit_rate < an.hit_rate
    d = run(advice("gpt-small"), view("gpt-big", 60_000))
    assert d.reason == policy.SWITCH


def test_priors_need_a_wider_margin_than_observations() -> None:
    cfg = PolicyConfig(min_savings_usd=0, rel_margin=0.2, prior_margin_extra=0.5)
    observed = {"turns": 3, "rounds": 6, "growth": 3000, "output": 300}
    # both caches cold: big -> mid saves ~40%, enough against 20%, not against 70%
    prior = run(advice("mid"), view(caches={}), cfg=cfg)
    seen = run(advice("mid"), view(caches={}, observed=observed), cfg=cfg)
    assert prior.gate["workload"]["basis"] == "prior" and prior.reason == policy.GATE_SAVINGS
    assert seen.gate["workload"]["basis"] == "observed" and seen.reason == policy.SWITCH


def test_a_minimum_absolute_saving_applies() -> None:
    d = run(advice("cheap-long", task="chat"), view(context=1000), cfg=PolicyConfig(min_savings_usd=5.0))
    assert d.reason == policy.GATE_SAVINGS
    assert d.gate["savings_needed_usd"] == 5.0


def test_upgrades_are_for_quality_not_savings() -> None:
    d = run(advice("big", task="debugging"), view("small", 20_000))
    assert d.reason == policy.SWITCH and d.gate["direction"] == "up"
    assert d.gate["switch_cost_usd"] > d.gate["stay_cost_usd"]
    strict = run(advice("big", task="debugging"), view("small", 20_000), cfg=PolicyConfig(upgrades="cost"))
    assert strict.reason == policy.GATE_SAVINGS


# --- hard gates and unknowns -----------------------------------------------------------


def test_the_context_window_is_a_hard_gate_with_reserves() -> None:
    # 150k * 1.2 + reserves (48k) > 200k
    d = run(advice("small"), view("big", 150_000))
    assert d.reason == policy.GATE_CONTEXT
    assert d.gate["context_required"] == 150_000 * 1.2 + 200 + 32_000 + 16_000
    assert run(advice("small"), view("big", 100_000)).reason != policy.GATE_CONTEXT


def test_the_agents_own_window_wins_over_the_catalogs() -> None:
    d = run(advice("cheap-long"), view("big", 150_000, windows={"cheap-long": 200_000}))
    assert d.reason == policy.GATE_CONTEXT and d.gate["window_source"] == "agent"


def test_the_window_cap_bounds_catalog_windows() -> None:
    d = run(advice("cheap-long"), view("big", 150_000), window_cap=200_000)
    assert d.reason == policy.GATE_CONTEXT and d.gate["window_source"] == "catalog, capped"


@pytest.mark.parametrize("target,current", [("unpriced", "big"), ("other-vendor", "big"), ("small", "nope")])
def test_unknown_pricing_keeps_the_model(target, current) -> None:
    d = run(advice(target), view(current))
    assert d.reason == policy.UNKNOWN_PRICING and d.applied_model == current


def test_unknown_context_keeps_the_model() -> None:
    assert run(advice("small"), view(context=None)).reason == policy.UNKNOWN_CONTEXT


def test_untested_and_failed_transitions() -> None:
    assert run(advice("small"), transition_ok=lambda a, b: None).reason == policy.TRANSITION_UNTESTED
    assert run(advice("small"), transition_ok=lambda a, b: False).reason == policy.TRANSITION_UNSUPPORTED
    allowed = run(advice("cheap-long"), transition_ok=lambda a, b: None, cfg=PolicyConfig(allow_untested=True))
    assert allowed.reason == policy.SWITCH and allowed.gate["transition"] == "untested"


def test_cooldown_after_a_switch() -> None:
    recent = view(turn=5, last_switch_turn=4, last_switch_t=NOW - 600)
    assert run(advice("cheap-long"), recent).reason == policy.GATE_COOLDOWN
    soon = view(turn=9, last_switch_turn=4, last_switch_t=NOW - 30)
    assert run(advice("cheap-long"), soon).reason == policy.GATE_COOLDOWN
    later = view(turn=9, last_switch_turn=4, last_switch_t=NOW - 600)
    assert run(advice("cheap-long"), later).reason == policy.SWITCH


def test_follow_ups_manual_models_and_no_route_never_switch() -> None:
    assert run(advice("cheap-long"), routed_on="pinned").reason == policy.FOLLOW_UP
    assert run(advice("cheap-long"), routed_on="previous exchange").reason == policy.FOLLOW_UP
    assert run(advice("cheap-long"), manual=True).reason == policy.MANUAL_MODEL
    assert run(None, routed_on=None).reason == policy.NO_ROUTE
    assert run(advice("big")).reason == policy.ALREADY_ON


def test_shadow_decides_but_keeps_the_model() -> None:
    d = run(advice("cheap-long"), shadow=True)
    assert d.reason == policy.SWITCH and d.shadow and not d.switches
    assert d.recommended_model == "cheap-long" and d.applied_model == "big"


def test_decision_record_is_flat_and_textless() -> None:
    rec = run(advice("cheap-long")).record()
    assert rec["reason"] == "switch" and rec["recommended_model"] == "cheap-long"
    assert set(rec) <= {f for f in Decision.__dataclass_fields__}


# --- which routing a prompt follows ----------------------------------------------------


def test_choose_route_prefers_the_prompts_own_routing() -> None:
    own = advice("small")
    assert policy.choose_route(own, {"model": "big"}, baseline="x", baseline_id="x") == (own, "prompt")


def test_a_follow_up_takes_the_pin_and_a_stale_estimate_is_dropped() -> None:
    pinned = {"model": "small", "baseline": "big", "estimated_savings_usd": 1.0,
              "baseline_estimated_cost_usd": 2.0, "baseline_model": "big"}
    same, how = policy.choose_route(None, pinned, baseline="big", baseline_id="big")
    assert how == "pinned" and same["estimated_savings_usd"] == 1.0
    moved, _ = policy.choose_route(None, pinned, baseline="mid", baseline_id="mid")
    assert "estimated_savings_usd" not in moved and moved["baseline_model"] == "mid"


def test_nothing_pinned_asks_for_the_exchange() -> None:
    assert policy.choose_route(None, None, baseline=None, baseline_id=None) == (None, "previous exchange")
    assert policy.continuation_text("yes", None, None, 600, lambda t, n: t or "") == ""
    text = policy.continuation_text("yes", "find the loader", "Shall I search?", 600, lambda t, n: t or "")
    assert text == "yes Shall I search? find the loader"


def test_a_longer_prompt_the_router_cannot_read_is_a_request_of_its_own() -> None:
    pinned = {"model": "small", "baseline": "big"}
    prompt = "Commit the working tree changes with a sensible message."
    assert not policy.continues(prompt)
    assert policy.choose_route(None, pinned, baseline="big", baseline_id="big", prompt=prompt) == (None, "unrouted")
    # short, or opening with an assent: a follow-up that takes the pin
    for follow_up in ("yes, go ahead", "ok", "Sure, and also update the changelog and the docs index please"):
        assert policy.continues(follow_up)
        _, how = policy.choose_route(None, pinned, baseline="big", baseline_id="big", prompt=follow_up)
        assert how == "pinned"


@pytest.mark.parametrize("question", ["did you use actrouter in the work?", "Why did you skip the tests?",
                                      "have you committed it yet?"])
def test_questions_about_the_agents_own_work(question: str) -> None:
    assert policy.about_own_work(question)


@pytest.mark.parametrize("request_", ["Did the deploy fail last night? Check the logs",
                                      "you should fix the failing test", "what does this function do?"])
def test_requests_are_not_questions_about_own_work(request_: str) -> None:
    assert not policy.about_own_work(request_)


def test_pin_keeps_the_applied_model_apart() -> None:
    p = policy.pin(advice("small"), baseline="big", baseline_source="session model", stays=False,
                   applied_model="big")
    assert p["model"] == "small" and p["applied_model"] == "big"
    assert "applied_model" not in policy.pin(advice("small"), baseline=None, baseline_source="x", stays=False)


def test_usable() -> None:
    assert policy.usable(advice("small"), 0.5)
    assert not policy.usable(advice("small", conf=0.4), 0.5)
    assert not policy.usable(advice("small", task="unknown"), 0.5)
    assert not policy.usable({**advice("small"), "host_model": None}, 0.5)
    assert not policy.usable(None, 0.5)
