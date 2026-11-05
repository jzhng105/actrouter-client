# Learning from agents and users

ActRouter's routing is set by its task types, its model catalog and its
scoring. None of that knows whether the model it picked actually did the
work well. The people and agents using it do, and they show it: a user
says "that's wrong", an agent ignores a delegation, someone switches
`/model` right after a recommendation, a subagent fails, the same task is
redone on a bigger model. This feature records those responses and turns
them into small, bounded adjustments to routing, so the router updates
itself as evidence accumulates.

```
 your machine (this package)                                                    ActRouter API
 agent session ──▶ hooks / chat ──▶ routing ledger ──▶ learner ──▶ learned.json ──▶ request `learned` ──▶ heuristic router
                   (feedback.py)    (labels only)     (learning.py)                                     (actrouter.learned)
                        ▲                                                                                       │
                        └──────────────────────────────── next routing decision ◀───────────────────────────────┘
```

Tracking (`src/actrouter_client/feedback.py`) and learning
(`src/actrouter_client/learning.py`) run here, on your machine: the ledger
never leaves it. With learning `on`, each routing request carries the
result as its `learned` field, and the ActRouter API applies it to that
request only (§3; `actrouter.learned` in the engine).

## 1. What is tracked

Each routed turn leaves a small **work record** per session
(`~/.actrouter/sessions/<session>.work.json`): the task type, the model
routing recommended, the model the session ran on, and each subagent the
turn started and the model it ran on. When the next prompt arrives, the
record is judged and then cleared.

| Event | Recorded as | When |
|---|---|---|
| The user's next message corrects the work ("that's wrong", "it still fails", "you broke the build", "try again") | `feedback` `signal: correction` | Claude Code prompt hook, `chat` |
| ... approves it ("thanks, works", "perfect", "LGTM") | `feedback` `signal: approval` | same |
| The prompt hook asked Claude to delegate to the routed model, and nothing in the turn ran on it | `feedback` `signal: declined` | prompt hook |
| The user switched to the recommended model within 30 minutes | `feedback` `signal: accepted` | `PostModelSwitch`, `chat` `/model` |
| ... to another model | `feedback` `signal: override`, `chosen_model` | same |
| Claude started a subagent on a model of its own choosing | `skipped` `reason: explicit_model`, now with `task_type`, `routed_model` (what routing would have chosen) and `chosen_model` | subagent hook, only with `ACTROUTER_LEARN_AGENT_CHOICE=on` |
| A subagent or chat turn finished, failed or was interrupted | existing `outcome` / `turn` events | unchanged |

Reaction detection is deliberately narrow (`feedback.reaction`). A reply
to a proposal ("no, use the other one", "yes, go ahead") and a new request
are neither a correction nor an approval. A reaction more than an hour after
the work is ignored.

**Privacy.** The user's message is read in the hook process, on the
machine, and never written: the ledger gets only the label, the task type
and the model ids. Classifying an explicitly-modelled subagent for
`agent_choice` sends its brief to the API as one more (metered) routing
call, so the client does it only when `ACTROUTER_LEARN_AGENT_CHOICE=on`.
`ACTROUTER_FEEDBACK=off` stops the tracking; `ACTROUTER_LEDGER=off` stops
all recording.

## 2. What is learned

`learning.signals` reads the ledger and derives one signal per piece of
evidence about a `(task type, model)`:

| Signal | Effect | Weight | Source |
|---|---|---|---|
| `correction` | bad | 1.0 | feedback |
| `approval` | good | 1.0 | feedback |
| `declined` | bad | 0.5 | feedback |
| `accepted` | good | 1.0 | feedback |
| `override` | bad, and a vote for the chosen model | 1.0 | feedback |
| `failed` | bad | 1.0 | a routed subagent's `outcome` (joined by `tool_use_id`, or `agent_id` for background subagents), a chat `turn` |
| `interrupted` | bad | 0.5 | same |
| `completed` | good | 0.25 | same; weak, since finishing is not succeeding |
| `escalated` | bad, and a vote for the larger model | 1.0 | the same task type started again on a larger `size_tier` in the same session within 20 minutes |
| `agent_choice` | bad, and a vote for the agent's model | 0.25 | an explicitly-modelled subagent whose model differs from routing's |
| `unclaimed` | bad | 0.25 | Claude Code `route_task` advice no subagent claimed within 15 minutes |

Weights decay exponentially with age (half-life 30 days) and only the last
120 days are read, so routing follows recent behaviour and old mistakes
stop counting.

Per `(task type, model)`:

```
quality   = (good + 4) / (good + bad + 5)          # Beta(4, 1) prior: 0.8
penalty   = min(0, quality - 0.8)
promotion = votes / (votes + 4)
bias      = clamp(0.4 * penalty + 0.08 * promotion, ±max_adjustment)   # 0 below min_evidence
```

Design choices:

- **Trusted until shown otherwise.** The prior says routing is right 80%
  of the time. Good signals only offset bad ones; they never promote a
  model above the prior. Otherwise the models routing already picks would
  collect the most "completed" signals and lock themselves in.
- **Promotion comes from preference, not volume.** A model gains only from
  votes: a user switching to it, work escalated to it, or (weakly) an agent
  choosing it.
- **Bounded.** No adjustment applies until the decayed evidence reaches
  `min_evidence` (3). Each is clamped to `max_adjustment` (0.12 score
  points), and the router clamps again at 0.15 whatever the file says.
  Adjacent size tiers usually score 0.04-0.09 apart, so the most a task
  type moves is about one tier. Three corrections with nothing good reach
  the cap.
- **Aliases.** Claude Code runs a subagent on an alias (`haiku`), which
  runs the newest model of its line whichever one routing chose. Claude
  Code evidence about such a model therefore counts for every catalog
  model the alias covers. Otherwise a penalty would only move routing
  from one Haiku to another, and the same model would keep running. Picking
  `haiku` when routing chose an older Haiku is not an `agent_choice`.

The result is written to `~/.actrouter/learned.json` (`ACTROUTER_LEARNED`)
with every `(task type, model)`'s evidence, so each adjustment can be
traced to the signals behind it.

## 3. How it is applied

The heuristic router adds `confidence × bias` to each candidate's score for
the classified task type, and says so in the ranking's reasons
(`learned=-0.118`; the agent sees it in the advice's `reasons`). The
`rules` router is unchanged: its policies are explicit pins.

| Mode (client `ACTROUTER_LEARNING`) | Tracks | Learns | Routes with it |
|---|---|---|---|
| `off` | yes, unless `ACTROUTER_FEEDBACK=off` | no | no |
| `shadow` (default) | yes | yes | no: `learn` shows what it would change |
| `on` | yes | yes | yes: each routing request carries `learned` |

**The `learned` field.** The client sends only the non-zero biases, and
nothing when there are none. `POST /v1/route` and `POST /v1/route/task` accept
an optional `learned`: task type → model id → bias, at most 64 task types
of 64 models each. It applies to that request only and the service stores
nothing, so one customer's learning never routes another's work. Each bias
is clamped to ±0.15 (`actrouter.learned.MAX_BIAS`) whatever the request
says. A deployment that predates the field rejects it (HTTP 422); the
client then routes without it and the response says so
(`compat_learned`).

```json
{"task": {"description": "Find where the config is loaded"},
 "learned": {"code_search": {"claude-haiku-4-5": -0.08}}}
```

The client relearns when a routing process starts and its file is older
than `refresh_hours` (6). That needs the catalog (`GET /v1/models`), which
is not metered.

## 4. Commands and settings

```bash
actrouter-route learn                   # relearn now and show every adjustment and its evidence
actrouter-route learn --json --dry-run
```

| Variable | Default | |
|---|---|---|
| `ACTROUTER_LEARNING` | `shadow` | `off`, `shadow` or `on` |
| `ACTROUTER_FEEDBACK` | on | `off` stops the tracking |
| `ACTROUTER_LEARNED` | `~/.actrouter/learned.json` | where the adjustments are kept |
| `ACTROUTER_LEARN_MIN_EVIDENCE` | 3 | |
| `ACTROUTER_LEARN_MAX_ADJUSTMENT` | 0.12 | the service clamps at 0.15 |
| `ACTROUTER_LEARN_HALF_LIFE_DAYS` | 30 | |
| `ACTROUTER_LEARN_WINDOW_DAYS` | 120 | |
| `ACTROUTER_LEARN_REFRESH_HOURS` | 6 | |
| `ACTROUTER_LEARN_AGENT_CHOICE` | off | `on` classifies explicitly-modelled subagents (metered) |


Rollout follows the same steps as `chat`: run in `shadow`, read `learn`
for a few weeks, then turn it `on`.

## 5. Limits

- **The signals are proxies.** A correction can be about the request, not
  the model, and an approval can be polite. Weights, the prior, decay and
  the evidence threshold keep any one signal small; review `learn` before
  turning it on.
- **Task types are fixed.** Learning moves models within the existing task
  types. It does not learn new keywords or task types; `actrouter
  mine-rules` covers that from classifier decisions.
- **One learned file per machine**, not per project or per user. The
  API applies whatever the request carries; it does not pool learning
  across customers.
- **No exploration.** A model routing never picks gathers no evidence of
  its own, except through votes.
- **Codex** has no hooks here, so from Codex only `chat codex` turns
  (statuses, reactions, `/model`) are learned from.
