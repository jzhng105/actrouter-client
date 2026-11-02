# ActRouter in your coding agent

ActRouter recommends the model that fits a piece of work. Inside a coding
agent, that means three things:

- **Your prompts get the right model.** Each request you type is routed. When
  it fits another model than the session's, the agent does the work on that
  model.
- **Subagents and tasks get the right model.** A codebase search goes to a
  small fast model; a subtle bug goes to a strong one.
- **LLM-calling code gets the right model.** When the agent writes a function
  that calls an LLM, it asks ActRouter which model to put in the code.

Design and privacy notes: [docs/agent_integration.md](../docs/agent_integration.md).

Everything below needs:

- **An ActRouter API key** (starts with `ar_live_`). Routing runs on the
  hosted ActRouter API, which rejects requests without one.
- [uv](https://docs.astral.sh/uv/) (for `uvx`). The first start takes a
  minute or two while uv builds the environment; after that a start takes
  about a second.

  Every `uvx` invocation below passes `--python 3.12`. The client needs
  Python >= 3.10, and without the flag `uvx` uses whichever `python3` comes
  first on `PATH` - which, in the environment an agent launches MCP servers
  from, is often the system Python 3.9 even when newer ones are installed.
  The resolve then fails with `No solution found when resolving tool
  dependencies` and the agent reports only a closed connection. uv downloads
  a 3.12 if the machine has none.

What is sent to the API, and what never is: [privacy](../docs/agent_integration.md#5-client-hosts-backends-privacy).

The very first start can exceed an agent's MCP startup timeout. If the
server shows as failed the first time, run this once to warm uv's cache,
then restart the agent:

```bash
ACTROUTER_API_KEY=ar_live_... uvx --python 3.12 --from "actrouter-client[mcp] @ git+https://github.com/jzhng105/actrouter-client@main" \
  python -m actrouter_client.cli models --host claude-code
```

The plugins start ActRouter with `python -m` rather than its console
scripts: uv's launcher for those scripts calls `realpath`, which macOS only
ships from version 13, so on older Macs they exit before ActRouter runs.

## Claude Code

Install the plugin from this repository's marketplace:

```
/plugin marketplace add jzhng105/actrouter-client
/plugin install actrouter@actrouter
```

Or from a shell:

```bash
claude plugin marketplace add jzhng105/actrouter-client
claude plugin install actrouter@actrouter
```

Claude Code asks for your API key on install. It is stored as a sensitive
setting (in the system keychain where there is one), not in `settings.json`.

You get:

| Part | What it does |
|---|---|
| Hook on your prompts | Each prompt is routed. If it fits another model than the session's, you see `ActRouter: code search -> haiku (this session runs on ...)`, and Claude does the work on that model through a subagent. A follow-up such as "yes, go ahead" stays on the model the conversation was routed to; with nothing routed yet (a new or resumed session), it is routed on the agent's last reply and your previous request. Slash commands are left alone. Fails open. |
| Hook on `Agent` | Every subagent started **without** a model gets a routed one (`haiku` / `sonnet` / `opus`). Explicit choices are never changed. Fails open. |
| MCP server `actrouter` | `route_task`, `route_function`, `route_function_metadata`, `list_models`, `list_task_types` |
| Skill `model-routing` | Tells Claude when to call the tools, and to describe work without pasting code |

Plugin settings (prompted on install, changeable in `/plugin`):

- `api_key`: your ActRouter API key.
- `router_url`: leave empty for the hosted API; set it only for a dedicated
  deployment you were given.
- `prompt_routing`: `delegate` (Claude does the work on the routed model
  through a subagent), `advise` (only tells you; `/model` switches) or `off`.
- `subagent_routing`: `on` / `off`, for the automatic subagent hook.
- `preference`: `quality` / `balanced` / `cost`, used by both hooks.

Claude Code gives hooks no way to switch the session's own model, so
`delegate` runs the work in a subagent on the routed model. The subagent
starts with only the context Claude gives it. If you would rather switch the
whole session, use `advise` and `/model`.

## Codex

Install the plugin (Codex 0.16x+ supports plugins):

```bash
export ACTROUTER_API_KEY=ar_live_...     # in your shell profile
codex plugin marketplace add jzhng105/actrouter-client
codex plugin add actrouter@actrouter
```

Or register only the MCP server:

```bash
codex mcp add actrouter --env ACTROUTER_HOST=codex -- \
  uvx --python 3.12 --quiet \
    --from "actrouter-client[mcp] @ git+https://github.com/jzhng105/actrouter-client@main" \
    python -m actrouter_client.mcp_server
```

which writes this to `~/.codex/config.toml`:

```toml
[mcp_servers.actrouter]
command = "uvx"
args = ["--python", "3.12", "--quiet", "--from", "actrouter-client[mcp] @ git+https://github.com/jzhng105/actrouter-client@main", "python", "-m", "actrouter_client.mcp_server"]

env_vars = ["ACTROUTER_API_KEY"]

[mcp_servers.actrouter.env]
ACTROUTER_HOST = "codex"
```

Codex passes MCP servers only a few environment variables by default, and
`env_vars` adds the key from your shell. The plugin sets it for you; with
`codex mcp add`, add that line yourself.

Codex sub-agents inherit the session's model unless one is set, so to steer
them, add [AGENTS.md.snippet](codex/AGENTS.md.snippet) to your `AGENTS.md`.

The snippet also asks Codex to route each request you make and hand
substantial work to a sub-agent on the routed model. Codex has no hook for
this yet, so it is advisory.

To route a whole session, start Codex through the launcher. It picks the
model from your prompt and runs `codex -m <model> ...`:

```bash
uvx --python 3.12 --from "actrouter-client @ git+https://github.com/jzhng105/actrouter-client@main" \
  python -m actrouter_client.cli launch codex -- "add retries to the http client"
```

## Any other MCP host (Cursor, Gemini CLI, Windsurf, ...)

Most hosts accept the common `mcpServers` JSON:

```json
{
  "mcpServers": {
    "actrouter": {
      "command": "uvx",
      "args": ["--python", "3.12", "--quiet", "--from", "actrouter-client[mcp] @ git+https://github.com/jzhng105/actrouter-client@main", "python", "-m", "actrouter_client.mcp_server"],
      "env": { "ACTROUTER_HOST": "generic", "ACTROUTER_API_KEY": "ar_live_..." }
    }
  }
}
```

Set `ACTROUTER_FAMILIES` to the vendors you can call (for example
`anthropic,openai`), so recommendations are always models you can use.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `ACTROUTER_API_KEY` | | Your API key, sent as `x-api-key`. Required |
| `ACTROUTER_URL` | hosted API | Base URL of a dedicated deployment |
| `ACTROUTER_HOST` | `generic` | `claude-code`, `codex` or `generic`: which models are candidates and how they are spelled |
| `ACTROUTER_FAMILIES` | per host | Comma-separated vendors to route over |
| `ACTROUTER_CANDIDATES` | | Explicit model-id whitelist |
| `ACTROUTER_MODEL_STYLE` | `alias` | `id` gives Claude Code full model ids |
| `ACTROUTER_PROMPT_ROUTING` | `delegate` | Claude Code prompt hook: `delegate`, `advise` or `off` |
| `ACTROUTER_PROMPT_MIN_CONFIDENCE` | `0.5` | Below this, a short prompt or one opening with an assent is a follow-up: it keeps the conversation's routing, else is routed on the exchange it continues. A longer prompt the router cannot read is a request of its own: it is recorded `unrouted` and Claude is asked to route it itself |
| `ACTROUTER_SUBAGENT_MIN_CONFIDENCE` | `0.5` | Below this, a subagent keeps the model it would have run on, rather than being routed on a guess |
| `ACTROUTER_PROMPT_ADVISE_TASKS` | `chat` | Task types the prompt hook only advises on, never delegates: work that lives in the conversation, which a subagent cannot see |
| `ACTROUTER_SUBAGENT_ROUTING` | `on` | Claude Code subagent hook switch |
| `ACTROUTER_HOOK_PREFERENCE` | `balanced` | Hook's quality/cost trade-off |
| `ACTROUTER_HOOK_PROMPT_CHARS` | prompts: 600; subagents: 0 | How much of your prompt, or a subagent's, the hooks route on. That text is sent to the API; 0 keeps it on the machine and turns prompt routing off |
| `ACTROUTER_HOOK_DECISION` | `allow` | `permissionDecision` sent with the change: `allow`, `ask` or `none` |
