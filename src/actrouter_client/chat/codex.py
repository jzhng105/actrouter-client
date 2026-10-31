"""Codex through its app-server: one thread, with the model set per turn.

``codex app-server`` speaks JSON-RPC over stdio, one JSON object per line,
without a ``jsonrpc`` field. The session is a thread: ``thread/start`` (or
``thread/resume``) opens it, and every prompt is a ``turn/start`` on it.
``turn/start`` takes ``model`` (and ``effort``), "for this turn and
subsequent turns". That is how the wrapper switches the main model before
the turn runs, while the thread and its history stay.

During a turn the server streams notifications (``item/agentMessage/delta``,
``item/started``, ``thread/tokenUsage/updated``, ``turn/completed``) and asks
the client for approvals as server requests
(``item/commandExecution/requestApproval``,
``item/fileChange/requestApproval``, ...), which the adapter puts to the
person through the :class:`~actrouter_client.chat.base.UI`. Requests this
client does not support are refused, so the agent sees a clear "no" instead
of hanging. Checked against codex-cli 0.166 (``codex app-server
generate-ts``).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections import deque
from typing import Any, Optional

from actrouter_client import __version__
from actrouter_client.chat.base import (
    APPROVE, APPROVE_SESSION, CANCEL, DENY, RequestUsage, TurnResult, UI,
)

#: tells the model the session's model is already routed, so the Codex
#: model-routing skill does not route the user's requests a second time
DEVELOPER_INSTRUCTIONS = (
    "ActRouter chooses this session's model before each of your turns. Do not route the "
    "user's requests yourself and do not hand them to a sub-agent to change model. Routing "
    "a sub-agent you start for your own reasons, or code that calls an LLM, is still fine."
)

#: line limit for the server's output: a resumed thread can arrive in one line
STREAM_LIMIT = 64 * 1024 * 1024

#: the decision values Codex takes, per UI answer
_V2_DECISIONS = {APPROVE: "accept", APPROVE_SESSION: "acceptForSession", DENY: "decline", CANCEL: "cancel"}
_LEGACY_DECISIONS = {
    APPROVE: "approved",
    APPROVE_SESSION: "approved_for_session",
    DENY: {"denied": {"rejection": "declined by the user"}},
    CANCEL: "abort",
}


class CodexError(RuntimeError):
    pass


class CodexAgent:
    name = "codex"
    host = "codex"

    def __init__(
        self,
        *,
        codex_bin: Optional[str] = None,
        cwd: Optional[str] = None,
        effort: Optional[str] = None,
        approval_policy: Optional[str] = None,
        sandbox: Optional[str] = None,
        developer_instructions: Optional[str] = DEVELOPER_INSTRUCTIONS,
        env: Optional[dict[str, str]] = None,
        config: Optional[list[str]] = None,
    ) -> None:
        self.codex_bin = codex_bin or shutil.which("codex") or "codex"
        self.cwd = cwd or os.getcwd()
        self.effort = effort
        self.approval_policy = approval_policy
        self.sandbox = sandbox
        self.developer_instructions = developer_instructions
        self.env = {**os.environ, "ACTROUTER_PROMPT_ROUTING": "off", **(env or {})}
        self.config = list(config or [])
        self.session_id: Optional[str] = None
        self.model: Optional[str] = None
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._events: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()
        self._reader: Optional[asyncio.Task[None]] = None
        self._stderr: deque[str] = deque(maxlen=40)
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._pending_model: Optional[str] = None
        self._effort_sent = False
        self._turn_id: Optional[str] = None
        self._turns = 0

    # --- transport ------------------------------------------------------------------

    async def _spawn(self) -> None:
        args = [self.codex_bin, "app-server"]
        for c in self.config:
            args += ["-c", c]
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
            limit=STREAM_LIMIT,
        )
        self._events, self._pending = asyncio.Queue(), {}
        self._reader = asyncio.create_task(self._read())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        async for line in self._proc.stderr:
            self._stderr.append(line.decode("utf-8", "replace").rstrip())

    async def _read(self) -> None:
        assert self._proc and self._proc.stdout
        # this process's queue and requests: a restarted server gets its own
        events, pending, stdout = self._events, self._pending, self._proc.stdout
        try:
            async for raw in stdout:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if "id" in msg and "method" not in msg:
                    fut = pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        if "error" in msg:
                            err = msg["error"] or {}
                            fut.set_exception(CodexError(str(err.get("message") or err)))
                        else:
                            fut.set_result(msg.get("result"))
                else:
                    await events.put(msg)
        finally:
            gone = CodexError("codex app-server exited" + (f": {self._stderr[-1]}" if self._stderr else ""))
            for fut in pending.values():
                if not fut.done():
                    fut.set_exception(gone)
            pending.clear()
            await events.put(None)

    async def _send(self, msg: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin or self._proc.returncode is not None:
            raise CodexError("codex app-server is not running")
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self._proc.stdin.drain()

    async def request(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        self._next_id += 1
        rid = self._next_id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send({"id": rid, "method": method, "params": params or {}})
        return await fut

    async def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        msg: dict[str, Any] = {"method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    async def _respond(self, rid: Any, result: Any = None, error: Optional[str] = None) -> None:
        if error is not None:
            await self._send({"id": rid, "error": {"code": -32601, "message": error}})
        else:
            await self._send({"id": rid, "result": result})

    # --- session --------------------------------------------------------------------

    async def start(self, *, model: Optional[str] = None, resume: Optional[str] = None) -> None:
        await self._spawn()
        await self.request("initialize", {
            "clientInfo": {"name": "actrouter-chat", "title": "ActRouter chat", "version": __version__},
            "capabilities": {"experimentalApi": False, "requestAttestation": False},
        })
        await self.notify("initialized")
        params: dict[str, Any] = {"cwd": self.cwd}
        if model:
            params["model"] = model
        if self.approval_policy:
            params["approvalPolicy"] = self.approval_policy
        if self.sandbox:
            params["sandbox"] = self.sandbox
        if resume:
            resp = await self.request("thread/resume", {"threadId": resume, "excludeTurns": True, **params})
        else:
            if self.developer_instructions:
                params["developerInstructions"] = self.developer_instructions
            resp = await self.request("thread/start", params)
        self.session_id = str(resp["thread"]["id"])
        self.model = resp.get("model") or model
        self._pending_model = None

    async def recover(self) -> None:
        """Restart the app-server and rejoin the thread, after it exited. A
        thread that never finished a turn has nothing on disk to resume
        (Codex writes its rollout with the first turn), so it starts afresh."""
        model = self._pending_model or self.model
        await self.close()
        try:
            await self.start(model=model, resume=self.session_id)
        except CodexError as exc:
            if self._turns or "no rollout" not in str(exc):
                raise
            await self.close()
            await self.start(model=model)

    async def set_model(self, model: str) -> None:
        # applied by the next turn/start, which keeps it for the turns after
        self._pending_model = model if model != self.model else None

    async def interrupt(self) -> None:
        if self.session_id and self._turn_id:
            try:
                await self.request("turn/interrupt", {"threadId": self.session_id, "turnId": self._turn_id})
            except CodexError:
                pass

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin:
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        for task in (self._reader, self._stderr_task):
            if task:
                task.cancel()

    # --- turns ----------------------------------------------------------------------

    async def run_turn(self, prompt: str, ui: UI) -> TurnResult:
        if not self.session_id:
            raise CodexError("no thread: call start() first")
        params: dict[str, Any] = {
            "threadId": self.session_id,
            "input": [{"type": "text", "text": prompt, "text_elements": []}],
        }
        model = self._pending_model or self.model
        if self._pending_model:
            params["model"] = self._pending_model
        if self.effort and not self._effort_sent:
            params["effort"] = self.effort  # persists, like the model
        result = TurnResult(status="failed")
        try:
            resp = await self.request("turn/start", params)
        except CodexError as exc:
            result.error = str(exc)
            result.transport_lost = self._proc is None or self._proc.returncode is not None
            return result
        self._effort_sent = self._effort_sent or bool(self.effort)
        self._turns += 1
        if self._pending_model:
            self.model, self._pending_model = self._pending_model, None
        self._turn_id = str((resp.get("turn") or {}).get("id") or "") or None
        if model:
            result.models.append(model)
        last_total: Optional[int] = None
        streaming = False
        try:
            while True:
                msg = await self._events.get()
                if msg is None:
                    result.error = "codex app-server exited" + (f": {self._stderr[-1]}" if self._stderr else "")
                    result.transport_lost = True
                    return result
                method, p = msg.get("method"), msg.get("params") or {}
                if "id" in msg:
                    await self._server_request(msg["id"], str(method), p, ui)
                    continue
                if p.get("threadId") not in (None, self.session_id):
                    continue  # a sub-agent's thread
                if method == "item/agentMessage/delta":
                    ui.text(str(p.get("delta") or ""))
                    streaming = True
                elif method == "item/started":
                    self._activity(p.get("item") or {}, ui, result)
                elif method == "item/completed":
                    item = p.get("item") or {}
                    if item.get("type") == "agentMessage":
                        result.reply = str(item.get("text") or "") or result.reply
                        if streaming:
                            ui.end_text()
                            streaming = False
                    elif item.get("type") == "contextCompaction":
                        result.compacted = True
                elif method == "thread/tokenUsage/updated":
                    usage = p.get("tokenUsage") or {}
                    total = (usage.get("total") or {}).get("totalTokens")
                    if total is not None and total == last_total:
                        continue  # the same request reported again
                    last_total = total
                    last = usage.get("last") or {}
                    result.requests.append(RequestUsage(
                        prompt=int(last.get("inputTokens") or 0),
                        cached=int(last.get("cachedInputTokens") or 0),
                        cache_write=int(last.get("cacheWriteInputTokens") or 0),
                        output=int(last.get("outputTokens") or 0),
                        reasoning=int(last.get("reasoningOutputTokens") or 0),
                    ))
                    if usage.get("modelContextWindow"):
                        result.context_window = int(usage["modelContextWindow"])
                elif method in ("thread/compacted", "contextCompacted"):
                    result.compacted = True
                elif method == "model/rerouted":
                    to = str(p.get("toModel") or "")
                    if to:
                        result.models.append(to)
                        self.model = to
                        ui.notice(f"Codex rerouted the turn: {p.get('fromModel')} -> {to} ({p.get('reason')})")
                elif method == "warning":
                    ui.notice(f"Codex: {p.get('message')}")
                elif method == "error":
                    err = p.get("error") or {}
                    ui.notice(f"Codex error: {err.get('message')}" + (" (retrying)" if p.get("willRetry") else ""))
                elif method == "turn/completed":
                    turn = p.get("turn") or {}
                    if self._turn_id and turn.get("id") not in (None, self._turn_id):
                        continue
                    if streaming:
                        ui.end_text()
                    status = str(turn.get("status") or "completed")
                    result.status = status if status in ("completed", "interrupted", "failed") else "completed"
                    if turn.get("error"):
                        result.error = str((turn["error"] or {}).get("message") or turn["error"])
                    return result
        finally:
            self._turn_id = None

    @staticmethod
    def _activity(item: dict[str, Any], ui: UI, result: TurnResult) -> None:
        kind = item.get("type")
        if kind == "commandExecution":
            ui.activity(f"$ {item.get('command')}")
        elif kind == "fileChange":
            paths = [c.get("path") for c in item.get("changes") or [] if isinstance(c, dict)]
            ui.activity("edit " + ", ".join(str(p) for p in paths if p) if paths else "edit files")
        elif kind == "mcpToolCall":
            ui.activity(f"tool {item.get('server')}.{item.get('tool')}")
        elif kind == "webSearch":
            ui.activity(f"search {item.get('query') or ''}".rstrip())
        elif kind == "contextCompaction":
            result.compacted = True
            ui.activity("compacting the conversation")

    async def _server_request(self, rid: Any, method: str, p: dict[str, Any], ui: UI) -> None:
        if method == "item/commandExecution/requestApproval":
            summary = str(p.get("command") or p.get("reason") or "run a command")
            if p.get("reason") and p.get("command"):
                summary += f"  ({p['reason']})"
            answer = await ui.approve("command", summary)
            await self._respond(rid, {"decision": _V2_DECISIONS[answer]})
        elif method == "item/fileChange/requestApproval":
            answer = await ui.approve("file change", str(p.get("reason") or "apply the proposed file changes"))
            await self._respond(rid, {"decision": _V2_DECISIONS[answer]})
        elif method == "execCommandApproval":
            answer = await ui.approve("command", " ".join(map(str, p.get("command") or [])) or "run a command")
            await self._respond(rid, {"decision": _LEGACY_DECISIONS[answer]})
        elif method == "applyPatchApproval":
            files = ", ".join(sorted((p.get("fileChanges") or {}).keys())) or "files"
            answer = await ui.approve("file change", f"patch {files}")
            await self._respond(rid, {"decision": _LEGACY_DECISIONS[answer]})
        elif method == "item/permissions/requestApproval":
            answer = await ui.approve("permissions", str(p.get("reason") or json.dumps(p.get("permissions"))))
            granted = p.get("permissions") if answer in (APPROVE, APPROVE_SESSION) else {}
            scope = "session" if answer == APPROVE_SESSION else "turn"
            await self._respond(rid, {"permissions": granted or {}, "scope": scope})
        else:
            ui.notice(f"Codex asked for {method}, which actrouter chat does not support; refused.")
            await self._respond(rid, error=f"{method} is not supported by actrouter chat")
