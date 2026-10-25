---
name: model-routing
description: Choose which model should do a piece of work with ActRouter. Use before delegating to a subagent when the model matters (hard debugging, large refactors, bulk mechanical work), when the user asks which model to use, and whenever you write or change code that calls an LLM and must pick its model.
---

# Model routing with ActRouter

ActRouter classifies work (search, debugging, code review, summarization, ...)
and recommends the model that fits it best for the quality, cost and latency
trade-off. The `actrouter` MCP server exposes it.

## Your requests

This plugin's hook routes each prompt the user submits. When the prompt fits
another model than yours, you get context starting "ActRouter routed this
request as ...". Follow it: do the work in an Agent call with the `model` it
names, pass that model explicitly, and relay the result. Answer directly only
when no tool calls are needed, when the request is about this conversation or
your own work in it (a subagent cannot see the conversation), when the user
named a model, or when the routing is plainly wrong for the request.

## Subagents

Subagents you start without a `model` are routed automatically by this
plugin's hook. You only need to act when you want to steer it:

- For an unusually hard or high-stakes task (security review, subtle
  concurrency bug, architecture), call `route_task` with
  `preference: "quality"` and pass its `host_model` as the Agent tool's
  `model`.
- For bulk or mechanical work (many similar edits, searches, renames), use
  `preference: "cost"`.
- A model the user named always wins over a recommendation.

Write the `description` for `route_task` as one or two sentences about the
*kind* of work: "Find where the retry policy is configured", "Fix the failing
auth integration test". Never include code, logs, secrets or file contents.

## Code that calls an LLM

When you write or modify a function that calls an LLM (an `actdecor`
decorator, an OpenAI/Anthropic/LiteLLM call, an agent step):

1. Write the function's name, signature and docstring first; they are what
   ActRouter routes on.
2. Call `route_function` with the file path and function name (Python), or
   `route_function_metadata` with name, signature and doc comment (any other
   language). Add token estimates or budgets when you know them.
3. Use the returned `model` in the code, e.g.
   `@summarize_with_llm(model="anthropic/claude-haiku-4-5")`, or ActRouter's
   own routing with `router_url=` when the code should keep routing at
   runtime.
4. Tell the user which model you chose and the one-line reason
   (`task_type`, `reasons`).

## Reading the answer

- `host_model`: the value Claude Code takes (`haiku`, `sonnet`, `opus`).
- `model`: the catalog id, for code and other tools.
- `task_type` / `task_confidence`: what ActRouter thinks the work is. If it
  is wrong, call again with an explicit `task_kind` (see `list_task_types`).
- `estimated_savings_usd`: present when you pass `current_model` (the model
  the work would otherwise run on: your own model alias, e.g. `sonnet`). Pass it so routing's
  benefit is recorded; `routing_report` sums it up when the user asks what
  routing saved.
- `note`: read it; it says when the answer is degraded (an old deployment,
  or a model this host cannot run).
