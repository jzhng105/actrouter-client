"""The terminal side of ``actrouter-route chat``: streamed text, notices, approvals.

Input is read on a daemon thread per request, so a pending prompt never
keeps the process from exiting. Nothing is read ahead: a line typed during a
turn waits in the terminal, and is never taken as the answer to an approval.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import Optional, TextIO

from actrouter_client.chat.base import APPROVE, APPROVE_SESSION, CANCEL, DENY

_ANSWERS = {"y": APPROVE, "yes": APPROVE, "a": APPROVE_SESSION, "always": APPROVE_SESSION,
            "n": DENY, "no": DENY, "c": CANCEL, "cancel": CANCEL}


async def _readline(prompt: str) -> Optional[str]:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[Optional[str]] = loop.create_future()

    def run() -> None:
        try:
            line: Optional[str] = input(prompt)
        except EOFError:
            line = None
        except Exception:  # noqa: BLE001 - a closed or broken stdin is end of input
            line = None
        loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(line))

    threading.Thread(target=run, daemon=True).start()
    return await fut


class TerminalUI:
    def __init__(self, out: TextIO = sys.stdout, err: TextIO = sys.stderr) -> None:
        self.out, self.err = out, err
        self._mid_line = False

    def text(self, delta: str) -> None:
        if delta:
            self.out.write(delta)
            self.out.flush()
            self._mid_line = not delta.endswith("\n")

    def end_text(self) -> None:
        if self._mid_line:
            self.out.write("\n")
            self.out.flush()
            self._mid_line = False

    def _line(self, text: str) -> None:
        self.end_text()
        self.err.write(text + "\n")
        self.err.flush()

    def notice(self, message: str) -> None:
        self._line(f"actrouter: {message}")

    def activity(self, message: str) -> None:
        self._line(f"  · {message}")

    async def approve(self, kind: str, summary: str, *, allow_session: bool = True) -> str:
        self._line(f"[{kind}] {summary}")
        choices = "[y]es / [a]lways this session / [n]o / [c]ancel the turn" if allow_session \
            else "[y]es / [n]o / [c]ancel the turn"
        while True:
            line = await _readline(f"  allow? {choices}: ")
            if line is None:
                return DENY
            answer = _ANSWERS.get(line.strip().lower())
            if answer == APPROVE_SESSION and not allow_session:
                answer = APPROVE
            if answer:
                return answer

    async def read(self, prompt: str) -> Optional[str]:
        self.end_text()
        return await _readline(prompt)


class ScriptedUI:
    """A UI that answers from lists, for tests and the transition matrix.
    `approve_if` decides approvals from the summary (default: deny)."""

    def __init__(self, inputs: Optional[list[str]] = None, approve_if=None) -> None:
        self.inputs = list(inputs or [])
        self.approve_if = approve_if or (lambda kind, summary: False)
        self.texts: list[str] = []
        self.notices: list[str] = []
        self.activities: list[str] = []
        self.approvals: list[tuple[str, str, str]] = []

    def text(self, delta: str) -> None:
        self.texts.append(delta)

    def end_text(self) -> None:
        self.texts.append("\n")

    def notice(self, message: str) -> None:
        self.notices.append(message)

    def activity(self, message: str) -> None:
        self.activities.append(message)

    async def approve(self, kind: str, summary: str, *, allow_session: bool = True) -> str:
        answer = APPROVE if self.approve_if(kind, summary) else DENY
        self.approvals.append((kind, summary, answer))
        return answer

    async def read(self, prompt: str) -> Optional[str]:
        return self.inputs.pop(0) if self.inputs else None
