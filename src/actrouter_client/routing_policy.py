"""When to switch a session's main model: the routing policy, without I/O.

Two parts, shared by the Claude Code prompt hook and ``actrouter-route chat``:

**Which routing a prompt follows** (:func:`choose_route`). A prompt the router
classifies on its own words routes on them. A follow-up it cannot classify
("yes, go ahead") follows the routing the conversation is pinned to, and with
nothing pinned it is routed on the exchange it continues
(:func:`continuation_text`). Each routing decision pins the conversation
(:func:`pin`) until a prompt routes on its own. Only a short prompt, or one
that opens with an assent, counts as a follow-up (:func:`continues`): a longer
one the router cannot read is a request of its own and stays unrouted
(``ROUTED_ON_NONE``) rather than inheriting the pin. A question about the
agent's own work in the conversation ("did you use ...?") is never routed and
keeps the pin (:func:`about_own_work`).

**Whether the session switches** (:func:`decide`). A recommendation is not a
switch. ``decide`` compares the expected cost of finishing the task on the
current model with the cost of finishing it on the recommended one. Its
inputs are the request rounds the task is expected to take, how warm each
model's prompt cache is, the target's cold-prefix and cache-write cost, and
long-context price tiers. It also applies the hard gates: the context
window, transitions known to work, and a cooldown. The answer is a
:class:`Decision` that keeps the *recommended* model apart from the
*applied* one: when a gate keeps the current model, follow-ups stay on the
model actually running.

The policy is conservative by design. When the context size, a price or the
workload is unknown, it keeps the current model and records why. The
workload priors (:data:`ROUND_PRIORS`) are only a starting estimate:
decisions based on them need a wider margin than decisions based on rounds
observed in the session.

Nothing here reads files, the environment or the network; callers pass
everything in, so every decision can be replayed from its record.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional

from actrouter_client.ledger import ROUTE_FIELDS

# --- reasons ------------------------------------------------------------------------

SWITCH = "switch"
ALREADY_ON = "already_on"
NO_ROUTE = "no_route"
CONVERSATION = "conversation"
FOLLOW_UP = "follow_up"
MANUAL_MODEL = "manual_model"
UNKNOWN_PRICING = "unknown_pricing"
UNKNOWN_CONTEXT = "unknown_context"
GATE_CONTEXT = "gate_context"
TRANSITION_UNSUPPORTED = "transition_unsupported"
TRANSITION_UNTESTED = "transition_untested"
GATE_COOLDOWN = "gate_cooldown"
GATE_SAVINGS = "gate_savings"

#: every reason a Decision can carry
REASONS = frozenset({
    SWITCH, ALREADY_ON, NO_ROUTE, CONVERSATION, FOLLOW_UP, MANUAL_MODEL, UNKNOWN_PRICING, UNKNOWN_CONTEXT,
    GATE_CONTEXT, TRANSITION_UNSUPPORTED, TRANSITION_UNTESTED, GATE_COOLDOWN, GATE_SAVINGS,
})


# --- choosing the routing a prompt follows -------------------------------------------

#: advice fields a pin keeps (no prompt text: reasons and evidence can quote it)
PIN_FIELDS = ROUTE_FIELDS

ROUTED_ON_PROMPT = "prompt"
ROUTED_ON_PIN = "pinned"
ROUTED_ON_EXCHANGE = "previous exchange"
#: the prompt is a request of its own the router cannot read: nothing applies
ROUTED_ON_NONE = "unrouted"

#: a prompt the router cannot read continues the work before it only when it
#: is this short or opens with an assent ("yes, go ahead"); a longer one
#: ("Commit the working tree changes with a sensible message") is a request of
#: its own, and inheriting the previous routing misroutes it
FOLLOW_UP_WORDS = 6
_ASSENT = re.compile(
    r"^(yes|yeah|yep|ok|okay|sure|please|go ahead|do it|do that|proceed|continue|sounds good|lgtm|"
    r"let's|option|the (first|second|third|last) one)\b",
    re.I,
)

#: a question about what the agent did in this conversation ("did you use
#: actrouter in the work?"): only the agent that did it can answer
_ABOUT_OWN_WORK = re.compile(r"^(why |what |how |where |when |which )?(did|have|had|were) you\b", re.I)


def continues(prompt: str) -> bool:
    """True when `prompt` reads as a follow-up to the work before it."""
    return len(re.findall(r"[\w'-]+", prompt)) <= FOLLOW_UP_WORDS or bool(_ASSENT.match(prompt.strip()))


def about_own_work(prompt: str) -> bool:
    """True when `prompt` asks about the agent's own work in the
    conversation: not new work, so it is not routed and keeps the pin."""
    return bool(_ABOUT_OWN_WORK.match(prompt.strip()))


def usable(advice: Optional[Mapping[str, Any]], min_confidence: float) -> bool:
    """True when `advice` classified the text well enough to act on."""
    return bool(
        advice
        and advice.get("host_model")
        and advice.get("task_type") not in (None, "unknown")
        and (advice.get("task_confidence") or 0.0) >= min_confidence
    )


def choose_route(
    own: Optional[Mapping[str, Any]],
    pinned: Optional[Mapping[str, Any]],
    *,
    baseline: Optional[str],
    baseline_id: Optional[str],
    prompt: Optional[str] = None,
) -> tuple[Optional[dict[str, Any]], str]:
    """The routing a prompt follows: ``(advice, routed_on)``.

    `own` is the prompt's own advice when it was usable (see :func:`usable`),
    else None. Given the `prompt`, one that does not read as a follow-up
    (:func:`continues`) returns ``(None, ROUTED_ON_NONE)``: it is a request
    the router cannot read, and nothing applies. A follow-up takes `pinned`. A pin made against another baseline
    (the user switched model since) keeps its model but loses its savings
    estimate, which was against the old baseline. ``(None,
    ROUTED_ON_EXCHANGE)`` means nothing applies yet: route the exchange the
    prompt continues (:func:`continuation_text`), if there is one.
    """
    if own is not None:
        return dict(own), ROUTED_ON_PROMPT
    if prompt is not None and not continues(prompt):
        return None, ROUTED_ON_NONE
    if pinned:
        advice = dict(pinned)
        if pinned.get("baseline") != baseline:
            for key in ("baseline_estimated_cost_usd", "estimated_savings_usd"):
                advice.pop(key, None)
            advice["baseline_model"] = baseline_id
        return advice, ROUTED_ON_PIN
    return None, ROUTED_ON_EXCHANGE


def continuation_text(prompt: str, previous: Optional[str], reply: Optional[str], chars: int,
                      compress: Callable[[Optional[str], int], str]) -> str:
    """What to route a follow-up on when nothing is pinned. It is made of the
    follow-up, the agent's last reply (which usually ends with the proposal
    being approved) and the request before it. Returns an empty string when
    there is no exchange to continue."""
    if not (previous or reply):
        return ""
    return " ".join(p for p in (
        prompt[: chars // 4],
        compress(reply, chars // 2),
        compress(previous, chars // 4),
    ) if p)


def pin(advice: Mapping[str, Any], *, baseline: Optional[str], baseline_source: str,
        stays: bool, applied_model: Optional[str] = None) -> dict[str, Any]:
    """The pin a routing decision leaves on the conversation. `applied_model`
    is the model the session actually runs the work on, when a switch policy
    decided it (it may differ from the advice's model when a gate kept the
    current model)."""
    out = {k: advice.get(k) for k in PIN_FIELDS}
    out.update(baseline=baseline, baseline_source=baseline_source, stays=stays)
    if applied_model is not None:
        out["applied_model"] = applied_model
    return out


# --- prices and caches ---------------------------------------------------------------


@dataclass(frozen=True)
class CacheBehaviour:
    """How a provider's prompt cache behaves, for the cost model.

    `write_multiplier` scales the input price for tokens written to the cache
    (Anthropic: 1.25x for the 5-minute cache; OpenAI caches automatically,
    with no premium). `ttl_s` is how long an unused prefix is assumed to stay
    cached, and `hit_rate` the chance a prefix within its TTL is really
    served from cache. OpenAI documents no retention guarantee, so its hit
    rate is lower.
    """

    write_multiplier: float
    ttl_s: float
    hit_rate: float


#: catalog family -> cache behaviour; a family missing here is unknown pricing
CACHE_BEHAVIOUR: dict[str, CacheBehaviour] = {
    "anthropic": CacheBehaviour(write_multiplier=1.25, ttl_s=300.0, hit_rate=0.95),
    "openai": CacheBehaviour(write_multiplier=1.0, ttl_s=300.0, hit_rate=0.8),
}


@dataclass(frozen=True)
class Tier:
    above_tokens: int
    input: float
    output: float
    cache_read: Optional[float]


@dataclass(frozen=True)
class Pricing:
    """One model's prices ($ per million tokens) and limits, from a ``GET
    /v1/models`` entry."""

    model: str
    family: Optional[str]
    input: float
    output: float
    cache_read: Optional[float]
    tiers: tuple[Tier, ...] = ()
    threshold_inclusive: bool = False
    context_window: Optional[int] = None
    size_tier: Optional[int] = None

    @classmethod
    def from_catalog(cls, entry: Optional[Mapping[str, Any]]) -> Optional["Pricing"]:
        """None when the entry lacks an input or output price."""
        if not entry:
            return None
        try:
            inp = float(entry["input_price_per_mtok"])
            out = float(entry["output_price_per_mtok"])
        except (KeyError, TypeError, ValueError):
            return None
        read = entry.get("cache_read_price_per_mtok")
        tiers = tuple(sorted(
            (
                Tier(int(t["above_tokens"]), float(t["input_price_per_mtok"]),
                     float(t["output_price_per_mtok"]),
                     None if t.get("cache_read_price_per_mtok") is None
                     else float(t["cache_read_price_per_mtok"]))
                for t in entry.get("context_tiers") or []
                if isinstance(t, Mapping) and t.get("above_tokens") is not None
            ),
            key=lambda t: t.above_tokens,
        ))
        window = entry.get("context_window")
        tier = entry.get("size_tier")
        return cls(
            model=str(entry.get("id") or entry.get("model_id") or ""),
            family=entry.get("family"),
            input=inp,
            output=out,
            cache_read=None if read is None else float(read),
            tiers=tiers,
            threshold_inclusive=bool(entry.get("threshold_inclusive")),
            context_window=int(window) if window else None,
            size_tier=int(tier) if tier is not None else None,
        )

    def at(self, prompt_tokens: float) -> tuple[float, float, float]:
        """``(input, output, cache_read)`` prices for a request whose prompt
        is `prompt_tokens` long: a long-context tier prices the whole request.
        Without a cache-read price, cached tokens are priced as input."""
        inp, out, read = self.input, self.output, self.cache_read
        for t in self.tiers:
            over = prompt_tokens >= t.above_tokens if self.threshold_inclusive else prompt_tokens > t.above_tokens
            if over:
                inp, out = t.input, t.output
                read = t.cache_read if t.cache_read is not None else read
        return inp, out, inp if read is None else read


def request_cost(pricing: Pricing, behaviour: CacheBehaviour, *, prompt: float, cached: float,
                 cache_write: float, output: float) -> float:
    """What one request costs ($), from its token counts. `prompt` is the
    whole prompt; `cached` and `cache_write` are parts of it."""
    inp, out, read = pricing.at(prompt)
    fresh = max(0.0, prompt - cached - cache_write)
    return (fresh * inp + cached * read + cache_write * inp * behaviour.write_multiplier
            + output * out) / 1e6


# --- the workload a turn is expected to be -------------------------------------------


@dataclass(frozen=True)
class Workload:
    """The request rounds a turn takes: how many, how much new input each
    adds (tool results, file reads) and how much each writes."""

    rounds: float
    growth_per_round: float
    output_per_round: float
    #: "observed" in this session, or "prior" (the task type's default)
    basis: str


#: task type -> (request rounds, new input tokens per round, output tokens per
#: round): a starting estimate for an agent turn, replaced by what the session
#: shows (:func:`workload`) once it has turns to go on
ROUND_PRIORS: dict[str, tuple[float, float, float]] = {
    "default": (5, 3000, 700),
    "chat": (1, 0, 500),
    "qa": (3, 2500, 500),
    "summarization": (2, 4000, 800),
    "code_search": (6, 3000, 300),
    "planning": (4, 3000, 1200),
    "documentation": (5, 2500, 1000),
    "code_review": (6, 4000, 700),
    "code_generation": (10, 2500, 900),
    "debugging": (12, 3000, 600),
}


def workload(task_type: Optional[str], observed: Optional[Mapping[str, Any]], *,
             min_observed_turns: int = 2) -> Workload:
    """The expected workload of a turn of `task_type`. `observed` holds what
    this session's turns showed (``turns``, ``rounds``, ``growth``,
    ``output``: running means); it replaces the prior once it covers
    `min_observed_turns` turns."""
    if observed and int(observed.get("turns") or 0) >= min_observed_turns:
        try:
            return Workload(max(1.0, float(observed["rounds"])), max(0.0, float(observed["growth"])),
                            max(0.0, float(observed["output"])), "observed")
        except (KeyError, TypeError, ValueError):
            pass
    rounds, growth, output = ROUND_PRIORS.get(task_type or "default", ROUND_PRIORS["default"])
    return Workload(rounds, growth, output, "prior")


def task_cost(pricing: Pricing, behaviour: CacheBehaviour, work: Workload, *, prefix: float,
              warm_prefix: float, first_hit_rate: float) -> float:
    """Expected cost ($) of running `work` on a model, starting from a prompt
    of `prefix` tokens. Up to `warm_prefix` of those tokens may be in the
    model's cache, served from it with probability `first_hit_rate`. A cache
    miss pays the write price (input times the cache-write multiplier). After
    the first round, each round's prompt is the previous one plus that
    round's output and new input, and it hits the cache at the provider's
    rate."""
    total = 0.0
    prompt, cached, hit = float(prefix), float(warm_prefix), first_hit_rate
    rounds = max(1, int(round(work.rounds)))
    for _ in range(rounds):
        inp, out, read = pricing.at(prompt)
        write = inp * behaviour.write_multiplier
        reused = min(cached, prompt)
        total += reused * (hit * read + (1 - hit) * write) + (prompt - reused) * write
        total += work.output_per_round * out
        cached, hit = prompt, behaviour.hit_rate
        prompt += work.growth_per_round + work.output_per_round
    return total / 1e6


def warm_prefix(cache: Optional[Mapping[str, Any]], behaviour: CacheBehaviour, now: float) -> tuple[float, float]:
    """``(prefix tokens, hit rate)`` a model's cache offers at `now`, given its
    cache record (``{"t": last use, "prefix": prompt tokens it was last
    sent}``). An unused cache past its TTL offers nothing. A cache that has
    not expired keeps its prefix after the session switches away, so
    switching back can still hit it."""
    if not cache:
        return 0.0, 0.0
    try:
        t, prefix = float(cache["t"]), float(cache["prefix"])
    except (KeyError, TypeError, ValueError):
        return 0.0, 0.0
    if now - t > behaviour.ttl_s or prefix <= 0:
        return 0.0, 0.0
    return prefix, behaviour.hit_rate


# --- the switch decision -------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    #: a cheaper model must save at least this much ($) per task...
    min_savings_usd: float = 0.01
    #: ...and at least this share of staying's cost
    rel_margin: float = 0.2
    #: added to rel_margin when the workload is a prior, not observed
    prior_margin_extra: float = 0.15
    #: turns and seconds after a switch before another
    cooldown_turns: int = 2
    cooldown_s: float = 120.0
    #: token counts differ between tokenizers: the margin on the transcript
    tokenizer_margin: float = 0.2
    #: kept free in the target's window for the answer and its reasoning
    output_reserve_tokens: int = 32_000
    reasoning_reserve_tokens: int = 16_000
    #: a switch to a model with a higher size tier is for quality: "quality"
    #: applies only the hard gates and the cooldown to it; "cost" also needs
    #: it to be cheaper, which in practice means never switching up
    upgrades: str = "quality"
    #: switch over transitions not yet tested to work (see transition_ok)
    allow_untested: bool = False


@dataclass
class SessionView:
    """What the policy needs to know about the session."""

    current_model: Optional[str]
    #: prompt tokens of the latest request (after any compaction); None when
    #: unknown (a resumed session before its first answer)
    context_tokens: Optional[int]
    now: float
    #: model -> {"t", "prefix"}; see warm_prefix
    caches: dict[str, Any] = field(default_factory=dict)
    #: model -> context window the agent itself reported for it
    windows: dict[str, int] = field(default_factory=dict)
    turn: int = 0
    last_switch_turn: Optional[int] = None
    last_switch_t: Optional[float] = None
    #: running means of the session's turns (see workload)
    observed: Optional[dict[str, Any]] = None


@dataclass
class Decision:
    """One turn's routing decision.

    `recommended_model` is what routing advised (a catalog id) and
    `applied_model` the model the turn runs on (a catalog id), which differ
    when a gate kept the current model or the policy only shadows. `reason`
    is the verdict: ``switch``, or why the current model stayed.
    """

    reason: str
    current_model: Optional[str]
    recommended_model: Optional[str]
    recommended_host_model: Optional[str]
    applied_model: Optional[str]
    applied_host_model: Optional[str]
    routed_on: Optional[str] = None
    task_type: Optional[str] = None
    task_confidence: Optional[float] = None
    shadow: bool = False
    gate: dict[str, Any] = field(default_factory=dict)

    @property
    def switches(self) -> bool:
        """True when the session must change model before the turn."""
        return (self.reason == SWITCH and not self.shadow
                and self.applied_model != self.current_model)

    def record(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {})}


def context_required(context_tokens: int, new_prompt_tokens: int, cfg: PolicyConfig) -> int:
    """Tokens the target's window must hold for this turn: the latest
    request's prompt (it already includes the system prompt and tool
    definitions) with a tokenizer margin, the new prompt, and the room the
    answer and its reasoning need."""
    return (math.ceil(context_tokens * (1 + cfg.tokenizer_margin)) + new_prompt_tokens
            + cfg.output_reserve_tokens + cfg.reasoning_reserve_tokens)


def decide(
    advice: Optional[Mapping[str, Any]],
    routed_on: Optional[str],
    session: SessionView,
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    new_prompt_tokens: int,
    cfg: PolicyConfig = PolicyConfig(),
    transition_ok: Callable[[str, str], Optional[bool]] = lambda a, b: None,
    window_cap: Optional[int] = None,
    shadow: bool = False,
    manual: bool = False,
) -> Decision:
    """The decision for one turn.

    `advice` is the routing it follows (:func:`choose_route`), `catalog` the
    model listing keyed by id. `transition_ok(a, b)` says whether switching
    from `a` to `b` is known to work (True), known to fail (False) or
    untested (None). `window_cap` bounds the window assumed for a model
    the agent has not reported one for: agents often run a model on less
    than its API window. With `shadow`, the decision is made and recorded,
    but the current model stays. With `manual`, the user chose the model
    and nothing switches it.
    """
    current = session.current_model
    keep = dict(current_model=current, applied_model=current,
                applied_host_model=None, shadow=shadow)
    if advice is None:
        return Decision(NO_ROUTE, recommended_model=None, recommended_host_model=None,
                        routed_on=routed_on, **keep)
    target, target_host = advice.get("model"), advice.get("host_model")
    base = dict(recommended_model=target, recommended_host_model=target_host, routed_on=routed_on,
                task_type=advice.get("task_type"), task_confidence=advice.get("task_confidence"))

    def stay(reason: str, **gate: Any) -> Decision:
        return Decision(reason, gate=gate, **base, **keep)

    if manual:
        return stay(MANUAL_MODEL)
    if target and target == current:
        return stay(ALREADY_ON)
    if routed_on != ROUTED_ON_PROMPT:
        return stay(FOLLOW_UP)  # a follow-up never switches on its own

    cur_p = Pricing.from_catalog(catalog.get(current or ""))
    tgt_p = Pricing.from_catalog(catalog.get(target or ""))
    cur_b = CACHE_BEHAVIOUR.get(cur_p.family or "") if cur_p else None
    tgt_b = CACHE_BEHAVIOUR.get(tgt_p.family or "") if tgt_p else None
    if not (current and target and cur_p and tgt_p and cur_b and tgt_b):
        return stay(UNKNOWN_PRICING, current_priced=bool(cur_p and cur_b),
                    target_priced=bool(tgt_p and tgt_b))

    if session.context_tokens is None:
        return stay(UNKNOWN_CONTEXT)
    window = session.windows.get(target)
    window_source = "agent"
    if not window:
        window, window_source = tgt_p.context_window, "catalog"
        if window and window_cap:
            window, window_source = min(window, window_cap), "catalog, capped"
    if not window:
        return stay(UNKNOWN_CONTEXT, window=None)
    required = context_required(session.context_tokens, new_prompt_tokens, cfg)
    gate: dict[str, Any] = {
        "context_tokens": session.context_tokens,
        "context_required": required,
        "window": window,
        "window_source": window_source,
    }
    if required > window:
        return stay(GATE_CONTEXT, **gate)

    known = transition_ok(current, target)
    gate["transition"] = {True: "ok", False: "failed", None: "untested"}[known]
    if known is False:
        return stay(TRANSITION_UNSUPPORTED, **gate)
    if known is None and not cfg.allow_untested:
        return stay(TRANSITION_UNTESTED, **gate)

    if session.last_switch_turn is not None:
        turns = session.turn - session.last_switch_turn
        secs = session.now - (session.last_switch_t or 0.0)
        gate.update(turns_since_switch=turns, seconds_since_switch=round(secs, 1))
        if turns < cfg.cooldown_turns or secs < cfg.cooldown_s:
            return stay(GATE_COOLDOWN, **gate)

    work = workload(advice.get("task_type"), session.observed)
    prefix = session.context_tokens + new_prompt_tokens
    cur_warm, cur_hit = warm_prefix(session.caches.get(current), cur_b, session.now)
    tgt_warm, tgt_hit = warm_prefix(session.caches.get(target), tgt_b, session.now)
    stay_cost = task_cost(cur_p, cur_b, work, prefix=prefix, warm_prefix=cur_warm, first_hit_rate=cur_hit)
    switch_cost = task_cost(tgt_p, tgt_b, work, prefix=prefix, warm_prefix=tgt_warm, first_hit_rate=tgt_hit)
    margin = cfg.rel_margin + (cfg.prior_margin_extra if work.basis == "prior" else 0.0)
    needed = max(cfg.min_savings_usd, margin * stay_cost)
    upgrade = (tgt_p.size_tier or 0) > (cur_p.size_tier or 0)
    gate.update(
        workload=asdict(work),
        stay_cost_usd=round(stay_cost, 6),
        switch_cost_usd=round(switch_cost, 6),
        savings_usd=round(stay_cost - switch_cost, 6),
        savings_needed_usd=round(needed, 6),
        current_cache_prefix=int(cur_warm),
        target_cache_prefix=int(tgt_warm),
        direction="up" if upgrade else "down" if (tgt_p.size_tier or 0) < (cur_p.size_tier or 0) else "same",
    )
    if not (upgrade and cfg.upgrades == "quality") and stay_cost - switch_cost < needed:
        return stay(GATE_SAVINGS, **gate)
    return Decision(SWITCH, current_model=current, applied_model=current if shadow else target,
                    applied_host_model=None if shadow else target_host, shadow=shadow, gate=gate, **base)
