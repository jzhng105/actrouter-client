# Agent integration: ActRouter inside Claude Code, Codex and other agents

Status: **implemented**. Package: `actrouter-client` (client, MCP server,
Claude Code hook, launcher). Plugins: `integrations/`. Routing runs on the
hosted ActRouter API.

---

## 1. The problem

ActRouter picks a model for a *function* from its static metadata. Coding
agents need a model picked for two other things:

1. **Their own work.** The user's requests ("where is the config
   loaded?", "fix the failing auth test") and the units of work an agent
   delegates to subagents. A codebase search does not need the model a
   subtle concurrency bug does.
2. **The code they write.** When an agent writes a function that calls an
   LLM, it has to put a model in the code, usually by habit.

(2) is ActRouter's existing API, reached from inside the agent. (1) needs a
new kind of request: a unit of work described in text, not a function.

## 2. Integration mechanisms, and why these

| Mechanism | What it can do | Used for |
|---|---|---|
| **MCP server** | Tools any MCP host can call: Claude Code, Codex, Cursor, Gemini CLI, ... | Both (1) on request and (2). The portable core. |
| **Plugin** (Claude Code; Codex) | Bundles the MCP server, a skill that says when to call it, and hooks; one-command install | Distribution |
| **UserPromptSubmit hook** (Claude Code) | Sees each prompt; can add context for the model and a message for the user, but cannot switch the session's model | (1), automatically: routes each request, and has Claude do it on the routed model through a subagent |
| **PreToolUse hook** (Claude Code) | Rewrites a tool call before it runs | (1), automatically: fills `model` on `Agent` calls that have none |
| **Launcher** (`actrouter-route launch`) | Picks the model before the agent starts | (1) for a whole session, any CLI agent |
| **Routed chat** (`actrouter-route chat`) | Owns the input loop; switches the main session's model before each turn through the agent's own API (Codex app-server `turn/start model`, Claude Agent SDK `set_model`) | (1) per turn, for the main model itself. See [chat_routing.md](chat_routing.md) |
| LLM proxy (`ANTHROPIC_BASE_URL`, Codex `base_url`) | Rewrites every model request | **Not built.** See §6 |

MCP alone is advisory: the agent must decide to call the tool. The hooks
are what make routing seamless in Claude Code. For subagents: the agent does nothing
differently, and a subagent started without a model gets a routed one. A
model the agent or user chose explicitly is never overridden.

**Prompt routing.** Claude Code changes the session's model only on `/model`,
the model picker or an SDK call (checked in 2.1.308: `PreModelSwitch` and
`PostModelSwitch` hooks observe a switch, no hook output requests one). So
the `UserPromptSubmit` hook routes the prompt and, when the routed model
differs from the session's, acts by mode (`ACTROUTER_PROMPT_ROUTING`):

- `delegate` (default): a `systemMessage` tells the user, and
  `additionalContext` tells Claude to do the work in an `Agent` call with the
  routed `model` unless it needs no tool calls. The subagent sees only the
  context Claude passes it, which is the cost of running on another model
  without switching the session.
- `advise`: the user is told, and `/model <model>` switches the session.

Some prompts are not routed at all, and keep the conversation's pin: slash
commands, and messages that are only tagged blocks Claude Code delivers as
prompts, such as the `<task-notification>` a background task sends when it
finishes (these used to route as summarization, ask Claude to delegate a
notice to haiku, and repin the conversation). Work that lives in the
conversation itself, `chat` by default (`ACTROUTER_PROMPT_ADVISE_TASKS`), is
advised but never delegated, and the delegation context tells Claude to
answer directly when a request is about the conversation or its own work in
it: a subagent starts cold and cannot see either.

**Follow-ups.** "Yes, go ahead" says nothing about the work it approves.
A prompt that does not classify with confidence
`ACTROUTER_PROMPT_MIN_CONFIDENCE` (0.5) or more on its own words is a
follow-up, and it **stays pinned** to the conversation's routing, as
commercial routers pin a conversation (OpenRouter's Auto Router keeps a
conversation on the model it chose, chiefly to keep the prompt cache warm).
Every routing decision pins, also one that keeps the session's model, and a
pin lasts until a prompt routes on its own words. This is cheap and
predictable; the price is that a follow-up approving different work ("sure"
to an offer of release notes after a search) keeps the search's model until
the user states the new task.

With nothing pinned (a new session, or a resumed one whose state is gone),
a follow-up is routed on the exchange it continues, read from the end of
the transcript
(`actrouter_client/transcript.py`): the agent's last reply, which usually
ends with the proposal being approved, and the user's previous request.
Tool results, tagged messages (command output, reminders) and sidechain
entries are skipped. The reply is compressed to prose: code blocks dropped,
its first sentence and its end kept. Of the routing budget, the prompt gets
a quarter, the reply half, the previous request a quarter. If that is no
clearer ("thanks" after small talk), the prompt is left alone. The ledger's
`prompt` event says what was routed on (`routed_on`: `prompt`, `pinned` or
`previous exchange`).

Slash commands, prompts injected by `/loop` and other non-user sources, and
prompts in subagents are left alone.
The session's model comes from `SessionStart` (`model`) and
`PostModelSwitch` (`to_model`), kept per session under
`~/.actrouter/sessions` (`ACTROUTER_STATE_DIR`), then from the transcript.

## 3. Where the code lives

This repository is the open client side; the routing engine is the hosted
ActRouter API.

```
src/actrouter_client/   actrouter-client: zero-dependency client
  client.py               HTTP client for the ActRouter API
  hosts.py                what each agent can run, and how it spells it
  advisor.py              routing restricted to the host, answered in its spelling
  metadata.py             FunctionInfo from Python source via ast
  mcp_server.py           MCP server            (extra: [mcp])
  claude_hook.py          Claude Code hook
  routing_policy.py       pinning and the per-turn switch decision (pure)
  session_state.py        per-session state, transitions known to work
  chat/                   actrouter-route chat: Codex and Claude adapters, REPL
  cli.py                  actrouter-route, incl. `launch` and `chat`
integrations/claude-code/ Claude Code plugin (.claude-plugin/)
integrations/codex/       Codex plugin (.codex-plugin/)
.claude-plugin/marketplace.json    Claude Code marketplace
.agents/plugins/marketplace.json   Codex marketplace
```

`actrouter-client` has no required dependencies, so other tools can embed it.

## 4. The API the client calls

`POST /v1/route` routes a function described by its static metadata;
`POST /v1/route/task` routes a unit of agent work described in a sentence
or two (`description`, optional `title`, `task_kind`, `needs`, budgets),
with `preference` set to `quality`, `balanced` or `cost`. `needs` (`tools`,
`vision`) are hard filters: if no candidate qualifies, the API returns 400.
The task description is capped at 2000 characters, data-shaped fields
(`input`, `payload`, ...) are rejected, and the description is never logged.
Every request carries the customer's API key as `x-api-key`; without a
valid one the API answers 401.

Deployments that predate `/v1/route/task` return 404. The HTTP client then
routes the task through `/v1/route`, carries the task kind as a `task:` tag,
applies `needs` as a candidate filter, and adds a `compat` note to the answer.


## 5. Client: hosts, backends, privacy

**The API.** `connect()` calls the hosted ActRouter API, or the deployment
`ACTROUTER_URL` names, with the key in `ACTROUTER_API_KEY`. The plugins run
`uvx --from actrouter-client[mcp]`, about 1 s per warm start, plus one HTTP
round trip per routing call. A 401 says how to set the key; the hooks fail
open on it like on any other error.

**Hosts.** An agent can run only some catalog models, and has its own
spelling for them:

| Host | Candidates | Spelling | Where it goes |
|---|---|---|---|
| `claude-code` | `anthropic` family | `haiku` / `sonnet` / `opus` / `fable` (or full id with `ACTROUTER_MODEL_STYLE=id`) | Agent tool `model`, subagent frontmatter, `claude --model` |
| `codex` | `openai` family | the model id | `spawn_agent` `model`, `codex -m`, `model =` |
| `generic` | whole catalog | the model id | your client |

Claude Code resolves aliases to the newest model of the family, so an alias
stays correct after the catalog goes stale. `ACTROUTER_FAMILIES` and
`ACTROUTER_CANDIDATES` widen or pin the candidates, e.g. for Claude Code
behind a gateway that serves other vendors.

**What leaves the machine** (sent to the ActRouter API):

- `route_task`: the description and title the agent writes. The skill tells
  it to describe the kind of work, never to paste code.
- `route_function`: the function's name, signature, decorators and
  docstring, parsed with `ast`. The body is never read into the request.
- The subagent hook: the `Agent` call's short `description` only. The
  prompt is included only when `ACTROUTER_HOOK_PROMPT_CHARS` is set.
- The prompt hook: the first `ACTROUTER_HOOK_PROMPT_CHARS` characters of
  the user's prompt (default 600). Set it to 0 to keep prompts on the
  machine; that turns prompt routing off. For a follow-up, the same budget covers the compressed
  last reply and previous request, without code blocks. The per-session state file holds models and the routing
  decision, never the prompt.
- `launch`: the first 600 characters of the prompt, or `--describe`.

**The hook fails open.** Any error, timeout or malformed event produces no
output, and the subagent runs as it would have without the hook.

## 6. Verified behaviour, and what was not built

Checked against the real tools:

- `claude plugin validate` passes for the plugin and the marketplace.
- In Claude Code 2.1.306 with the plugin loaded, the MCP server connects
  (`plugin:actrouter:actrouter`, 5 tools). An Explore subagent started
  without a model reaches the hook, the hook returns
  `updatedInput.model = "haiku"` with `permissionDecision: "allow"`, and the
  subagent runs on Haiku while the main loop stays on Sonnet.
- In Claude Code 2.1.308, `claude -p --model opus` with the hooks
  installed: the prompt "Search this repository for every place where the
  routing ledger is written to ..." is routed `code_search -> haiku`
  against the session's `claude-opus-5-5` (known from `SessionStart`), the
  user sees the notice, Claude starts an Explore subagent with
  `model: "haiku"`, the ledger records it as a `prompt_hook` route, and
  `actrouter-route report` measures it from the subagent's real usage.
- Follow-ups, in the same setup: a first turn ends with Claude offering to
  search the repository for every caller of `ledger.record`; the resumed
  turn "yes go ahead" stays pinned to the first turn's routing
  (`code_search -> haiku`, `routed_on: pinned`), and Claude runs the search
  in a haiku subagent, recorded and measured as delegated work.
- Codex CLI 0.163 lists and installs the Codex plugin from
  `.agents/plugins/marketplace.json`, and registers its MCP server. When
  both marketplace files exist, Codex prefers its own over
  `.claude-plugin/marketplace.json`.
- The MCP server passes the same stdio JSON-RPC tests on mcp 1.30 and 2.3.

Not built:

- **A Codex hook.** Codex 0.163 has `PreToolUse` hooks and a `spawn_agent`
  tool with a `model` field. Whether a hook can rewrite `spawn_agent`'s
  input could not be tested without a Codex login. Codex gets the MCP tools,
  the skill and `actrouter-route launch codex`.
- **An LLM proxy.** It would make routing fully transparent, per request.
  But it sees whole prompts, so it breaks the metadata-only design. It
  needs a prompt-content router and provider keys. It also breaks prompt
  caching and tool-call compatibility when it switches models mid-session.
  If it is built, it should run locally (`actrouter proxy`) and call the
  router only for the decision.

## 7. Measuring routing benefits

Routing only pays off if it can be shown to. Every routing decision is
compared with its **baseline**, the model that would have run without
ActRouter, and recorded locally. A report turns the records into savings.

**The baseline** (`actrouter_client/baseline.py`) is found the way the host
itself would have picked the model:

| Where | Baseline, first match wins |
|---|---|
| Claude Code prompt (hook) | the override → the session's model: what `SessionStart` / `PostModelSwitch` reported → the newest assistant message in the transcript → `ANTHROPIC_MODEL` / settings `model`. Whether to act compares with the session's model, never the override |
| Claude Code subagent (hook) | `ACTROUTER_BASELINE_MODEL` (or the plugin's `baseline_model`) → `CLAUDE_CODE_SUBAGENT_MODEL` → the subagent definition's `model:` (`.claude/agents/<type>.md`, project then user) unless `inherit` → a built-in agent's fixed model (`statusline-setup`: Sonnet) → **the session's model**, from the newest assistant message in the transcript, which is what `inherit` means and what the built-in Explore, Plan and general-purpose agents use → `ANTHROPIC_MODEL` / settings `model` |
| `actrouter-route launch` | the override → the CLI's configured model (`~/.codex/config.toml` `model`; Claude Code's `ANTHROPIC_MODEL` / settings) |
| MCP tools | `current_model`, which the agent passes (the skills ask it to) → the override |
| `actrouter-route task/function` | `--baseline` → the override |

The built-in agents' defaults were read from Claude Code 2.1.307. Aliases
(`sonnet`) resolve to the newest catalog model of that line, as Claude Code
resolves them; dated ids (`…-20260205`) to their catalog id.

**The server** costs the baseline with the same function and token estimate
as the chosen model. Requests take `baseline_model`, and responses carry
`baseline` (`model_id`, `known`, estimated cost and latency) and
`estimated_savings_usd`. A baseline outside the candidate list is still
costed. `POST /v1/route/task` also returns `token_estimate`: the tokens the
costs are for and their `basis`, `given` (the request's estimate) or
`prior` (the classified task type's typical agent work, `agent_tokens` in
`task_types.yaml`). Without it a subagent run was priced as one 100+100-token
call, about 1/500th of real usage. The route log line records both models and the saving. An HTTP
deployment that predates the field rejects it with a 422; the client then
routes without it rather than fail.

**The ledger** (`~/.actrouter/routing-ledger.jsonl`, or `ACTROUTER_LEDGER`;
`off` disables it) has three event kinds:

- `route`: task type, routed model, baseline model and its source, both
  estimated costs, session and tool-call ids;
- `skipped`: a subagent already had a model, so routing left it alone. It
  counts towards coverage. The exception is a model the MCP `route_task`
  tool just advised: the MCP server leaves the advice claimable for 15
  minutes (per project, model ids and costs only), and a subagent started
  on that model claims it and is recorded as a `route` with `surface: mcp`
  and `claims: <request_id>`, so its outcome is measured and the advice is
  counted once;
- `prompt`: a user's prompt fit another model than the session's (task
  type, both models, mode). When Claude then starts a subagent on the routed
  model, that subagent is recorded as a `route` with `surface: prompt_hook`
  (not `skipped`), so its outcome is measured like any routed subagent;
- `outcome`: the Claude Code **`PostToolUse`** hook records what the subagent
  actually ran on (`resolvedModel`) and its real `usage` (input, output,
  cache-read and cache-write tokens), duration and status, all from the
  Agent tool's result. A background subagent returns at launch
  (`status: async_launched`) without usage, so the launch records its
  `agent_id`, and the **`SubagentStop`** hook records its usage when it
  finishes, from its own transcript (`agent_transcript_path`; model and usage
  fields only, each API request counted once), with the largest single
  prompt so long-context tiers are priced per request. The report joins the
  two by agent id and keeps the `PostToolUse` usage when both exist.

No task description, prompt or tool output is recorded, and a write failure
never affects routing.

**The report** (`actrouter-route report [--days N] [--html FILE] [--json]`;
MCP `routing_report`) keeps two numbers apart:

- **Measured savings:** each routed subagent's real tokens priced on the
  model that ran and on the baseline. Same tokens on both sides, so it
  isolates the price effect of the switch. Another model would not have
  used exactly the same tokens, and the page says so.
- **Estimated savings:** the router's before-the-fact estimate, covering
  calls with no outcome (launches, MCP advice, background subagents).

Prices come from the catalog, including the cached-token rates and
long-context tiers synced from ActLLMInfer. Cache writes are priced at the
input rate, which is slightly low for providers that charge a premium.

`--html` writes a self-contained dashboard with no external scripts or
fonts, so it works offline and sends nothing anywhere. It shows headline
tiles, a daily baseline-vs-routed chart, and tables by task type and by
model switch. Negative savings, where routing chose a pricier model for
harder work, are shown, not hidden.

Not built: a team-wide dashboard. Each ledger is per machine. A team can
collect them, or, when routing over HTTP, use the server's route log
(baseline, chosen model and saving per request) in CloudWatch Logs
Insights.
