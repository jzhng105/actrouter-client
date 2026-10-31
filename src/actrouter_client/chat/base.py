"""What the chat wrapper needs from an agent, and from the person using it.

An :class:`Agent` adapter drives one agent session through the agent's own
API (the Codex app-server, the Claude Agent SDK). It starts or resumes the
session, sets the model for the next turn through the agent's own
model-switching call, runs a turn while streaming its output, handles
approvals, can be interrupted, and reports what the turn used
(:class:`TurnResult`).

A :class:`UI` is how the adapter talks to the person: streamed text,
notices, and approval questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

#: approval answers a UI gives
APPROVE = "yes"
APPROVE_SESSION = "session"
DENY = "no"
CANCEL = "cancel"


@dataclass
class RequestUsage:
    """One API request's tokens. `prompt` is the whole prompt; `cached` and
    `cache_write` are parts of it."""

    prompt: int = 0
    cached: int = 0
    cache_write: int = 0
    output: int = 0
    reasoning: int = 0


@dataclass
class TurnResult:
    #: completed | interrupted | failed
    status: str
    error: Optional[str] = None
    #: the models that served the turn, as the agent named them
    models: list[str] = field(default_factory=list)
    #: one entry per API request of the main session, in order
    requests: list[RequestUsage] = field(default_factory=list)
    #: totals for the turn, when the agent reports them (else the sum of requests)
    totals: Optional[RequestUsage] = None
    compacted: bool = False
    #: the context window the agent reported for the model it ran
    context_window: Optional[int] = None
    #: what the agent itself reported the turn cost ($), when it does
    cost_reported_usd: Optional[float] = None
    #: the agent's final message text (kept in memory to route follow-ups)
    reply: Optional[str] = None
    #: the connection to the agent was lost: recover() before the next turn
    transport_lost: bool = False

    def total(self) -> RequestUsage:
        if self.totals is not None:
            return self.totals
        out = RequestUsage()
        for r in self.requests:
            out.prompt += r.prompt
            out.cached += r.cached
            out.cache_write += r.cache_write
            out.output += r.output
            out.reasoning += r.reasoning
        return out


class UI(Protocol):
    def text(self, delta: str) -> None:
        """Streamed agent text."""

    def end_text(self) -> None:
        """The end of a streamed message."""

    def notice(self, message: str) -> None:
        """A line from the wrapper (routing, status)."""

    def activity(self, message: str) -> None:
        """What the agent is doing (a command, an edit, a tool)."""

    async def approve(self, kind: str, summary: str, *, allow_session: bool = True) -> str:
        """Ask the person; answer APPROVE, APPROVE_SESSION, DENY or CANCEL."""

    async def read(self, prompt: str) -> Optional[str]:
        """The next line the person types, or None at end of input."""


class Agent(Protocol):
    #: "codex" | "claude"
    name: str
    #: the ActRouter host the agent runs as ("codex", "claude-code")
    host: str
    #: the session id, once started
    session_id: Optional[str]
    #: the model the agent says the session runs on, as it spells it
    model: Optional[str]

    async def start(self, *, model: Optional[str], resume: Optional[str]) -> None: ...

    async def set_model(self, model: str) -> None:
        """Make `model` the session's model from the next turn on, through
        the agent's own model-switching call."""

    async def run_turn(self, prompt: str, ui: UI) -> TurnResult: ...

    async def interrupt(self) -> None: ...

    async def recover(self) -> None:
        """Reconnect after the connection was lost, rejoining the session."""

    async def close(self) -> None: ...
