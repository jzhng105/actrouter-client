# actrouter-client

Routes your coding agent's work to the right model. A codebase search goes
to a small fast model; a subtle bug goes to a strong one. Routing runs on
the hosted ActRouter API; this repository is the open client side:

- **Plugins for Claude Code and Codex.** Your prompts and every subagent get
  a routed model automatically, and code the agent writes that calls an LLM
  gets a recommended model.
- **An MCP server** for any MCP host (Cursor, Gemini CLI, Windsurf, ...).
- **`actrouter-client`**, a Python package with no required dependencies:
  the API client, the Claude Code hooks, and the `actrouter-route` CLI,
  including `launch` (start `codex` or `claude` on a routed model) and
  `report` (what routing saved, from a ledger kept on your machine).

You need an ActRouter API key (starts with `ar_live_`).

| Plan | Price | Routing calls |
|---|---|---|
| Free | $0 | 30 a month |
| Pro | $10 a month | 1,500 included, then $10 per 1,000 |

Only routing calls count; listing models is free. `actrouter-route upgrade`
moves your key to Pro, and `actrouter-route billing` manages your card,
invoices and plan. When the free calls run out, the plugin tells you once
per session and Claude carries on unrouted.

## Claude Code

```
/plugin marketplace add jzhng105/actrouter-client
/plugin install actrouter@actrouter
```

Claude Code asks for your API key on install.

## Codex

```bash
export ACTROUTER_API_KEY=ar_live_...     # in your shell profile
codex plugin marketplace add jzhng105/actrouter-client
codex plugin add actrouter@actrouter
```

## Python

```bash
pip install "actrouter-client[mcp] @ git+https://github.com/jzhng105/actrouter-client@main"
export ACTROUTER_API_KEY=ar_live_...

actrouter-route task "Fix the failing auth test" --host claude-code
actrouter-route function src/news.py summarize_article
actrouter-route launch codex -- "add retries to the http client"
actrouter-route report --days 30 --html savings.html
```

```python
from actrouter_client import Advisor

advice = Advisor(host="codex").for_task("Search the repo for usages of parse_args")
advice["host_model"]   # 'gpt-5-mini'
```

Setup for each agent and every setting: [integrations/README.md](integrations/README.md).
Design, and exactly what is sent to the API: [docs/agent_integration.md](docs/agent_integration.md).

## Development

```bash
pip install -e '.[dev]'
pytest
```

The routing engine is not in this repository. The tests run against a
stand-in API that replays real API answers from
`tests/fixtures/api_replay.json`; see `tests/api_replay.py`.
