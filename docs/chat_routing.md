# Per-turn main-model routing: `actrouter-route chat`

The Claude Code hook routes prompts and subagents, but it cannot switch the
session's own model: no hook output requests a switch. `actrouter-route
launch` chooses the model once, then `exec`s the agent and gives up control
of every later prompt. `chat` switches the main model before each turn. It
owns the input loop and drives the agent through the agent's own API. For
each prompt it routes the prompt, decides whether a switch is worth it, and
switches the existing session's model with the agent's own call. Then it
runs the turn.

```
person ─▶ chat REPL ─▶ RoutedSession.turn(prompt)
                         1. Advisor.for_task(prompt[:600])          ActRouter API
                         2. choose_route: own | pinned | exchange   routing_policy (pure)
                         3. decide: gates + stay-vs-switch cost     routing_policy (pure)
                         4. agent.set_model(applied)                Codex turn/start model | SDK set_model
                         5. agent.run_turn: stream, approvals       Codex app-server | Claude Agent SDK
                         6. observe usage → ChatState, ledger       session_state, ledger
```

This is a new client execution path; the routing endpoint and the
model-selection algorithm are unchanged. `launch` still picks a model for
an agent you want to drive yourself.

## Usage

```bash
pip install 'actrouter-client[claude]'        # the Claude Agent SDK, for `chat claude`
actrouter-route chat codex                    # shadow mode: decides and records, never switches
actrouter-route chat claude --route on "find where the retry policy is set"
actrouter-route chat codex --resume <thread-id>
actrouter-route chat claude --test-transitions sonnet,haiku,opus
actrouter-route report                        # includes a "chat turns" section
```

In the session:

| Command | Effect |
|---|---|
| `/model <name>` | switch now and keep it: routing leaves a model the person chose alone |
| `/model auto` | hand the model back to routing |
| `/route on\|shadow\|off` | switch per turn \| only record \| do not route |
| `/status` | session, model, mode, and the last decision with its costs |
| `/help`, `/quit` | |
| Ctrl-C during a turn | interrupt the turn (`turn/interrupt`, SDK `interrupt()`) |
| Ctrl-D | end the session; it prints the `--resume` id |

Anything else, including other slash commands, goes to the agent as a
prompt. The REPL deliberately supports only this set. It is a small agent
client, not a copy of either agent's TUI.

Options: `--route` (default `shadow`, or `$ACTROUTER_CHAT_ROUTING`),
`--model`, `--resume`, `--effort` (set once and kept, never routed),
`--preference`, `--min-savings`, `--margin`, `--cooldown-turns`,
`--cooldown-seconds`, `--upgrades quality|cost`, `--allow-untested`,
`--window-cap`. Codex: `--codex-bin`, `--approval-policy`, `--sandbox`,
`-c key=value`. Claude: `--permission-mode`, `--claude-bin`.

## Choosing the routing a prompt follows

This is shared with the Claude Code hook (`routing_policy.choose_route`),
so a prompt follows the same routing whether `chat` switches the model or
the hook only advises:

* a prompt the router classifies with confidence ≥ 0.5 routes on its own words;
* a question about the agent's own work in the conversation ("did you run
  the tests?", `about_own_work`) is not routed at all: only the model that
  did the work can answer. The model stays (`conversation`) and the pin is
  kept;
* only a short prompt (≤ 6 words), or one that opens with an assent ("yes",
  "go ahead", "the second one"), counts as a follow-up (`continues`). A
  longer prompt the router cannot read ("Commit the working tree changes
  with a sensible message") is a request of its own: it is recorded as
  `routed_on: unrouted`, keeps the model (`no_route`), and does not inherit
  the pin, which stays for a real follow-up. The hook records it as an
  `unrouted` event and, with `delegate`, tells Claude to call `route_task`
  once it knows what the work is;
* a follow-up ("yes, go ahead") follows the conversation's **pin**;
* with nothing pinned, a follow-up is routed on the exchange it continues: the
  follow-up, the agent's last reply and the request before it
  (`continuation_text`; `chat` keeps the last exchange in memory only).

The decision keeps the **recommended** model apart from the **applied**
one. The pin records both. When a gate rejects a recommendation, the pin's
`applied_model` is the model that actually runs. **A follow-up never
switches on its own** (`follow_up`), so the follow-ups stay on the running
model. Only a prompt that routes on its own words can switch, which keeps
a recommendation that the cooldown blocked from coming back through a "yes".

## The switch decision

`routing_policy.decide` is pure: every input is passed in, and every
decision can be replayed from its ledger record. Checks run in this order,
and the first that fails gives the reason:

| Reason | Meaning |
|---|---|
| `no_route` | routing gave nothing usable (low confidence, API down, mode `off`, or a request the router cannot read: `routed_on: unrouted`) |
| `conversation` | a question about the agent's own work: not routed, the pin is kept |
| `manual_model` | the person chose the model (`/model`, `--model`) |
| `already_on` | the recommendation is the model running |
| `follow_up` | the prompt followed a pin or the previous exchange |
| `unknown_pricing` | a model has no catalog price, or its provider's cache behaviour is unknown |
| `unknown_context` | the latest request's size, or the target's window, is unknown |
| `gate_context` | the target's window cannot hold the turn |
| `transition_unsupported` / `transition_untested` | the pair is known to fail / was never tested |
| `gate_cooldown` | too soon after the last switch |
| `gate_savings` | switching is not expected to save enough |
| `switch` | switch (in `shadow`, recorded with `shadow: true` and not applied) |

**Unknown inputs keep the current model.** An unknown context size, price
or workload keeps the model, and the record says why. This gives a reliable
baseline before more aggressive routing is turned on.

### Context window (hard gate)

```
required = ceil(last_request_prompt_tokens × (1 + 0.2))   # tokenizer margin
         + new_prompt_estimate                            # len/3, deliberately high
         + 32k output reserve + 16k reasoning reserve
switch only if required ≤ window(target)
```

* `last_request_prompt_tokens` is the latest request's prompt, after any
  compaction. It comes from the agent's per-request usage, not from
  cumulative session tokens. It already includes the system prompt and tool
  definitions, so they are not added again.
* `window(target)` is the window the agent itself reported for that model
  when there is one. Codex reports `modelContextWindow`; Claude reports
  `model_usage[m].contextWindow`. Otherwise it is the catalog's window,
  capped per host: Claude Code 200k, Codex 272k (`--window-cap`), because
  agents often run a model on less than its API window.
* A new session starts at an estimated 25k tokens (system prompt and tools).
  Both caches are cold then, so the estimate weighs the same on both sides.

### Stay versus switch

The decision compares the **total expected cost of finishing the task** on
each model. It does not use the target's cache penalty alone. For a model
`m`, starting from a prompt of `P` tokens with `W` of them possibly in
`m`'s cache:

```
for each of R expected rounds:
    reused = min(cached, prompt)
    cost  += reused × (hit × read + (1 − hit) × write)  +  (prompt − reused) × write
    cost  += output × out_price
    cached = prompt; hit = provider hit rate
    prompt += growth + output
where write = input_price × cache_write_multiplier, priced at the request's long-context tier
```

* **The current model's likely cache hit.** Its cache record (time of the
  last request, prefix tokens) counts if it is still within the provider's
  TTL.
* **The target's cold input and write premium.** A cache that has not
  expired still counts: switching back to a model whose cache is warm can
  hit it. A compaction rewrites the history, so it clears every model's
  cache record.
* **Provider behaviour** (`CACHE_BEHAVIOUR`):

  | Provider | Write multiplier | TTL | Hit rate within TTL |
  |---|---|---|---|
  | Anthropic | 1.25 | 300 s | 0.95 |
  | OpenAI | 1.0 (automatic caching) | 300 s | 0.8 (no retention guarantee) |

  Any other family counts as unknown pricing.
* **Expected rounds.** `R`, `growth` (new input per round) and `output` come
  from `ROUND_PRIORS` for the task type. Once the session has two observed
  turns, running means of its own rounds replace them. **Priors are a
  starting estimate, not evidence of savings**, so a decision based on a
  prior needs `rel_margin + 0.15`.
* **Margin and cooldown.** A switch must save
  `max(min_savings_usd, margin × stay_cost)`. The defaults are $0.01 and 20%
  (35% on priors). After a switch, at least 2 turns and 120 s must pass
  before the next one.
* **Upgrades** (a target with a higher `size_tier`) are made for quality,
  not cost. With `--upgrades quality` (the default) they need only the hard
  gates and the cooldown. With `--upgrades cost` they must also be cheaper,
  which in practice means never switching up.

Reasoning effort is not routed. `--effort` sets it once, and it persists
(Codex `turn/start effort`, Claude `ClaudeAgentOptions.effort`). Changing
the model and the effort together would make savings and quality harder to
assess.

## Transitions known to work

The official switching call keeps the runtime informed about the model's
capabilities, but that does not prove every transition works. Reasoning
items or thinking blocks that one model wrote may not be accepted by
another. `--test-transitions a,b,c` tests every ordered pair. Each test
opens a fresh session on `a` and runs a turn that reasons and makes one tool
call; only the probe command is approved. It then switches to `b` with the
same call routing uses and asks about the earlier turn. A pair works when
that second turn completes on `b`. Results go to
`~/.actrouter/sessions/transitions.json`. Without `--allow-untested`, the
policy switches only over pairs recorded as working. A switch that fails at
runtime marks its pair as failed, moves the session back to the previous
model, and is never tried again until it is tested again.

## Agents

**Codex**, through `codex app-server`: JSON-RPC over stdio, one object per
line, no `jsonrpc` field.
* **Start:** `initialize`, the `initialized` notification, then
  `thread/start`, or `thread/resume` for `--resume` or to recover after the
  server exits. `thread/start` carries `developerInstructions` telling the
  model that ActRouter already routes the session, so the Codex
  model-routing skill does not route the user's requests a second time.
* **Each prompt** is `turn/start {threadId, input, model?, effort?}`. `model`
  is sent only when it changes: it applies "for this turn and subsequent
  turns".
* **Streaming:** `item/agentMessage/delta` (text), `item/started`
  (commands, edits, tools), `thread/tokenUsage/updated` (per-request `last`
  usage and `modelContextWindow`; a repeated report is counted once),
  `model/rerouted`, `warning`, `error`, `turn/completed`.
* **Approvals:** `item/commandExecution/requestApproval` and
  `item/fileChange/requestApproval` (plus the legacy `execCommandApproval`
  and `applyPatchApproval`) are put to the person; the answers are accept /
  accept for session / decline / cancel. `item/permissions/requestApproval`
  grants the request for the turn or session, or grants nothing. Requests
  this client does not support (`item/tool/requestUserInput`, MCP
  elicitation, ...) get a JSON-RPC error, so the agent sees a refusal
  instead of hanging.
* **Interrupt:** `turn/interrupt {threadId, turnId}`.

**Claude Code**, through one persistent `ClaudeSDKClient` in streaming mode:
* **Each prompt:** `set_model(...)`, only when the model changes, then
  `query(prompt)`; `receive_response()` runs until the `ResultMessage`.
* **Settings:** user, project and local settings, CLAUDE.md and plugins load
  as in the CLI. The ActRouter prompt hook is turned off for this session
  with `ACTROUTER_PROMPT_ROUTING=off`, so routing does not happen twice; the
  subagent hook keeps working.
* **Approvals:** tool permissions arrive at `can_use_tool`. "Always" applies
  the permission suggestions the CLI attached.
* **Usage:** each main-session `AssistantMessage` is one request, counted
  once per message id. `ResultMessage.total_cost_usd` is the session's
  running total on a persistent client, so a turn's **reported** cost is its
  increase since the last result. It is recorded apart from the
  **computed** cost; Claude Code's figure includes subagents and background
  calls.
* **Recovery:** reconnect with `resume=<session id>`.

**Recovery.** When the connection is lost (the app-server or CLI exits),
the session reconnects to the same thread or session, and the person
resends the prompt. A Codex thread that never finished a turn has nothing
on disk to resume, so it starts afresh.

## State, ledger and privacy

* `~/.actrouter/sessions/<session>.chat.json` (`ACTROUTER_STATE_DIR`) holds:
  the model, the latest context size, each model's cache record, the
  windows the agent reported, the last switch, running means of the
  workload, the pin and the manual flag. It holds **no prompt text**: the
  last exchange is kept in memory only. Resuming a session restores this
  state, cooldown included.
* Each turn writes one `turn` event to the routing ledger. The event holds
  the full decision: reason, recommended and applied models, whether it ran
  in shadow, and the gate's inputs (context, window, workload and its
  basis, stay and switch cost, savings needed, cache prefixes, direction,
  transition). It also holds the outcome: status, served model, requests,
  token totals, `cost_computed_usd` (main session, from tokens and catalog
  prices) and `cost_reported_usd` (Claude Code's own figure). `/model`
  writes a `model_set` event.
* `actrouter-route report` sums the turns: switches, the switches shadow
  mode would have made, both costs, the decision reasons, and the
  policy's *estimated* savings. Neither cost proves a counterfactual saving:
  another model would have used other tokens.
* As with the hook, the first 600 characters of each prompt go to the
  ActRouter API (`ACTROUTER_CHAT_PROMPT_CHARS`; 0 keeps prompts on the
  machine and turns routing off).

## Rollout

1. **Shadow (the default).** The policy runs on real context sizes, cache
   states and workloads, and records what it would do. This is the
   baseline.
2. **`--route on`** with the conservative defaults, after
   `--test-transitions` for the models you use.
3. **Tune `--margin`, `--min-savings` and the cooldown** from the shadow
   records.

## Verified, and not

Checked here against the real tools:

* **Claude** (claude-agent-sdk 0.2.172, Claude Code 2.1.316):
  * A session started on `opus`; `set_model("haiku")` was then sent on the
    same client, and the turn really ran on `claude-haiku-5-5` (from the
    `AssistantMessage` model). The probe turn made its tool call (`Bash:
    echo actrouter-transition-probe`).
  * `--test-transitions sonnet,haiku`: both directions (`sonnet -> haiku`,
    `haiku -> sonnet`) completed after a reasoning-and-tool turn.
  * A three-turn `--route on` session, routed by the real engine: the first
    prompt (`code_search`) started the session on `haiku`, and the follow-up
    stayed pinned. Each turn's ledger event has its requests, prompt tokens
    and both costs. This run found two bugs, now fixed and tested:
    `total_cost_usd` is a running total, and Claude Code's alias `haiku`
    runs the newest Haiku even when the engine chose an older one, so the
    policy now prices and checks the model the host really runs.
* **Codex** (codex-cli 0.166):
  * The protocol types were read from `codex app-server generate-ts`.
    Against the real server: `initialize`, `thread/start` (it reports the
    model) and `turn/start` with a `model` override, which applied to that
    turn (the server warned about that model's metadata).
  * `turn/interrupt` ended the turn as `interrupted`. Recovery after a
    restart started a fresh thread when there was nothing to resume.
  * No Codex turn could reach OpenAI from the test environment (the proxy
    refused it), so a Codex turn's usage, approvals and
    `--test-transitions` were checked only against a scripted stand-in
    server (`tests/fake_codex.py`) built from those types.

Not done:

* **No per-turn effort routing**, on purpose (see above).
* **The HTML report has no chat section yet**; the text and JSON reports do.
* **The cost model is an estimate.** Provider TTLs and hit rates are
  assumptions. Claude Code streams `AssistantMessage` usage before a
  response finishes, so per-request output counts can be low. That is one
  more reason to treat computed costs as estimates, and to record Claude's
  own cost separately.
