---
name: model-routing
description: Choose which model should do a piece of work with ActRouter. Use when the user makes a new request that needs tool calls, before spawning a sub-agent for a sizeable or unusually hard task, when the user asks which model to use, and whenever you write or change code that calls an LLM and must pick its model.
---

# Model routing with ActRouter

ActRouter classifies work (search, debugging, code review, summarization, ...)
and recommends the model that fits it for the quality, cost and latency
trade-off. The `actrouter` MCP server exposes it.

## The user's requests

When the user makes a new request that needs tool calls, route it before you
start: call `route_task` with one or two sentences on the kind of work and
your own model as `current_model`. If its `host_model` differs from your
model, spawn a sub-agent with that `model`, give it the request and the
context it needs, and relay its result. Skip follow-ups such as "yes, go
ahead", and requests you can answer from the conversation.

## Sub-agents and tasks

Sub-agents inherit your model unless you set one. When the user has asked
you to route by model, or has enabled ActRouter for this purpose:

1. Call `route_task` with a one or two sentence description of the *kind*
   of work ("Search the repo for usages of parse_args", "Fix the failing
   auth integration test"). Never include code, logs, secrets or file
   contents.
2. Pass its `host_model` as the `model` of `spawn_agent`.

Use `preference: "quality"` for hard debugging, design or security work and
`"cost"` for bulk or mechanical work. A model the user named always wins.

## Code that calls an LLM

When you write or modify a function that calls an LLM:

1. Write the function's name, signature and docstring first; they are what
   ActRouter routes on.
2. Call `route_function` with the file path and function name (Python), or
   `route_function_metadata` with name, signature and doc comment (any other
   language).
3. Use the returned `model` in the code and tell the user which model you
   chose and why (`task_type`, `reasons`).

## Reading the answer

- `host_model`: the OpenAI model id Codex takes.
- `model`: the catalog id, for code and other tools.
- `task_type` / `task_confidence`: what ActRouter thinks the work is; if it
  is wrong, call again with an explicit `task_kind` (see `list_task_types`).
- `estimated_savings_usd`: present when you pass `current_model` (the model
  the work would otherwise run on: your own model id). Pass it so routing's
  benefit is recorded; `routing_report` sums it up when the user asks what
  routing saved.
- `note`: read it; it says when the answer is degraded.
