"""Claude Code through the Claude Agent SDK: one persistent client, model set per turn.

A :class:`claude_agent_sdk.ClaudeSDKClient` connected in streaming mode keeps
one Claude Code session open. Before each prompt the adapter calls
``set_model`` (the SDK's control request for what ``/model`` does), then
``query`` submits the prompt and ``receive_response`` streams the turn up to
its ``ResultMessage``. Tool permissions come to ``can_use_tool`` and are put
to the person through the :class:`~actrouter_client.chat.base.UI`.

Accounting, from the SDK's messages:

* each main-session ``AssistantMessage`` (``parent_tool_use_id`` None) is one
  API request; its ``usage`` gives the prompt size (input + cache read +
  cache write). Claude Code repeats a response once per content block,
  so each message id counts once;
* ``ResultMessage.total_cost_usd`` is the dollar figure Claude Code itself
  reports. On a persistent client it is the session's running total (seen
  with Claude Code 2.1.316), so a turn's cost is the increase since the last
  result. It covers subagents and background calls too, so it is recorded
  as *reported* cost, apart from the cost the wrapper computes from tokens;
* ``ResultMessage.model_usage[model].contextWindow`` is the window Claude Code
  runs that model with.

Checked against claude-agent-sdk 0.2.172 with Claude Code 2.1.316. Settings,
CLAUDE.md and plugins load as in the CLI. The ActRouter prompt hook, if it
is installed, is turned off for this session (``ACTROUTER_PROMPT_ROUTING=off``)
so routing does not happen twice.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from actrouter_client.chat.base import APPROVE, APPROVE_SESSION, CANCEL, RequestUsage, TurnResult, UI


def _summary(tool: str, tool_input: dict[str, Any]) -> str:
    for key in ("command", "file_path", "path", "pattern", "url", "description", "prompt"):
        value = tool_input.get(key)
        if value:
            text = " ".join(str(value).split())
            return f"{tool}: {text[:200]}"
    return tool


class ClaudeAgent:
    name = "claude"
    host = "claude-code"

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        effort: Optional[str] = None,
        permission_mode: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        cli_path: Optional[str] = None,
        client_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.cwd = cwd or os.getcwd()
        self.effort = effort
        self.permission_mode = permission_mode
        self.env = {"ACTROUTER_PROMPT_ROUTING": "off", **(env or {})}
        self.cli_path = cli_path
        self._factory = client_factory
        self._client: Any = None
        self._ui: Optional[UI] = None
        self.session_id: Optional[str] = None
        self.model: Optional[str] = None
        #: the session's running cost at the last result, on this connection
        self._cost_total = 0.0

    def _options(self, model: Optional[str], resume: Optional[str]) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        kwargs: dict[str, Any] = dict(
            cwd=self.cwd,
            env=self.env,
            include_partial_messages=True,
            can_use_tool=self._can_use_tool,
        )
        if model:
            kwargs["model"] = model
        if resume:
            kwargs["resume"] = resume
        if self.effort:
            kwargs["effort"] = self.effort
        if self.permission_mode:
            kwargs["permission_mode"] = self.permission_mode
        if self.cli_path:
            kwargs["cli_path"] = self.cli_path
        return ClaudeAgentOptions(**kwargs)

    async def start(self, *, model: Optional[str] = None, resume: Optional[str] = None) -> None:
        if self._factory is None:
            try:
                from claude_agent_sdk import ClaudeSDKClient
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise RuntimeError(
                    "actrouter chat claude needs the Claude Agent SDK: "
                    "pip install 'actrouter-client[claude]'"
                ) from exc
            self._factory = ClaudeSDKClient
        self._client = self._factory(self._options(model, resume))
        await self._client.connect()
        self._cost_total = 0.0  # a new CLI process counts from zero
        self.session_id = resume or self.session_id
        self.model = model

    async def recover(self) -> None:
        await self.close()
        await self.start(model=self.model, resume=self.session_id)

    async def set_model(self, model: str) -> None:
        if model != self.model:
            await self._client.set_model(model)
            self.model = model

    async def interrupt(self) -> None:
        if self._client is not None:
            try:
                await self._client.interrupt()
            except Exception:  # noqa: BLE001 - nothing to interrupt
                pass

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001 - already gone
                pass

    async def _can_use_tool(self, tool: str, tool_input: dict[str, Any], context: Any) -> Any:
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        if self._ui is None:
            return PermissionResultDeny(message="no one is there to approve it")
        answer = await self._ui.approve("tool", _summary(tool, tool_input))
        if answer == APPROVE:
            return PermissionResultAllow()
        if answer == APPROVE_SESSION:
            return PermissionResultAllow(updated_permissions=list(getattr(context, "suggestions", None) or []) or None)
        return PermissionResultDeny(message="declined by the user", interrupt=answer == CANCEL)

    async def run_turn(self, prompt: str, ui: UI) -> TurnResult:
        from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock, ToolUseBlock

        try:
            from claude_agent_sdk import StreamEvent
        except ImportError:  # pragma: no cover - older SDKs
            StreamEvent = ()  # type: ignore[assignment,misc]

        self._ui = ui
        result = TurnResult(status="failed")
        by_id: dict[str, RequestUsage] = {}
        streaming = False
        texts: list[str] = []
        try:
            await self._client.query(prompt)
            async for msg in self._client.receive_response():
                if StreamEvent and isinstance(msg, StreamEvent):
                    if msg.parent_tool_use_id is not None:
                        continue
                    ev = msg.event or {}
                    delta = ev.get("delta") or {}
                    if ev.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                        ui.text(str(delta.get("text") or ""))
                        streaming = True
                    elif ev.get("type") == "message_stop" and streaming:
                        ui.end_text()
                        streaming = False
                elif isinstance(msg, AssistantMessage):
                    if msg.parent_tool_use_id is not None:
                        continue  # a subagent's request
                    if msg.model and msg.model != "<synthetic>" and msg.model not in result.models:
                        result.models.append(msg.model)
                    usage = msg.usage or {}
                    if usage:
                        key = msg.message_id or str(len(by_id))
                        req = RequestUsage(
                            prompt=int(usage.get("input_tokens") or 0)
                            + int(usage.get("cache_read_input_tokens") or 0)
                            + int(usage.get("cache_creation_input_tokens") or 0),
                            cached=int(usage.get("cache_read_input_tokens") or 0),
                            cache_write=int(usage.get("cache_creation_input_tokens") or 0),
                            output=int(usage.get("output_tokens") or 0),
                        )
                        old = by_id.get(key)
                        if old is not None:
                            req.output = max(req.output, old.output)  # later copies count further
                        by_id[key] = req
                    for block in msg.content or []:
                        if isinstance(block, ToolUseBlock):
                            ui.activity(_summary(block.name, block.input or {}))
                        elif isinstance(block, TextBlock) and block.text:
                            texts.append(block.text)
                elif isinstance(msg, SystemMessage):
                    data = msg.data or {}
                    if msg.subtype == "init":
                        self.session_id = str(data.get("session_id") or self.session_id or "") or None
                        if data.get("model"):
                            self.model = self.model or str(data["model"])
                    elif msg.subtype == "compact_boundary":
                        result.compacted = True
                        ui.activity("compacting the conversation")
                elif isinstance(msg, ResultMessage):
                    if streaming:
                        ui.end_text()
                        streaming = False
                    self.session_id = msg.session_id or self.session_id
                    reason = str(getattr(msg, "terminal_reason", None) or "")
                    if reason.startswith("aborted"):
                        result.status = "interrupted"
                    elif msg.is_error:
                        result.status = "failed"
                        result.error = "; ".join(msg.errors or []) or msg.result or msg.subtype
                    else:
                        result.status = "completed"
                    if msg.total_cost_usd is not None:
                        total = float(msg.total_cost_usd)
                        # the running total; it restarts when the CLI process does
                        result.cost_reported_usd = round(total - self._cost_total if total >= self._cost_total
                                                         else total, 8)
                        self._cost_total = total
                    usage_by_model = msg.model_usage or {}
                    main = result.models[-1] if result.models else None
                    entry = usage_by_model.get(main) if main else None
                    if entry is None and len(usage_by_model) == 1:
                        entry = next(iter(usage_by_model.values()))
                    if entry and entry.get("contextWindow"):
                        result.context_window = int(entry["contextWindow"])
        except Exception as exc:  # noqa: BLE001 - the CLI died or the SDK failed
            result.status, result.error = "failed", str(exc) or type(exc).__name__
            result.transport_lost = True
        finally:
            self._ui = None
        result.requests = list(by_id.values())
        result.reply = texts[-1] if texts else None
        return result
