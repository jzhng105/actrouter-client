"""``actrouter-route chat``: an agent session whose main model is routed per turn.

``actrouter-route launch`` picks the model once and hands the terminal to the
agent. ``chat`` keeps the session instead. It reads each prompt, routes it,
and switches the agent's own session to the routed model, through the
agent's model-switching API, before the turn runs. Switching is decided by
:mod:`actrouter_client.routing_policy`. The supported agents are Codex (its
app-server, ``turn/start`` with ``model``) and Claude Code (a persistent
Claude Agent SDK client, ``set_model``). See ``docs/chat_routing.md``.
"""

from actrouter_client.chat.base import Agent, RequestUsage, TurnResult, UI

__all__ = ["Agent", "RequestUsage", "TurnResult", "UI"]
