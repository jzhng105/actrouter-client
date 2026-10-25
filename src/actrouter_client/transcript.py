"""What a follow-up prompt continues, and what a subagent used, read from
Claude Code transcripts.

"Yes, go ahead" cannot be routed on its own words: the work it approves is in
the user's previous request and, above all, in the agent's last reply, which
usually ends with the proposal being approved. :func:`last_exchange` reads
both from the end of the session transcript, and :func:`compress` cuts the
reply down to what routing needs: prose, without code blocks, keeping its
opening sentence and its end.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from actrouter_client.baseline import TRANSCRIPT_TAIL_BYTES

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+")


def _text(content: Any) -> Optional[str]:
    """The text of a message's content, or None when it holds no text
    (tool results, images)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(p for p in parts if p) or None
    return None


def _entries(transcript_path: str) -> list[dict[str, Any]]:
    try:
        with open(transcript_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - TRANSCRIPT_TAIL_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in tail.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # the first line of the tail may be cut
        if isinstance(entry, dict) and not entry.get("isSidechain") and not entry.get("isMeta"):
            out.append(entry)
    return out


def last_exchange(transcript_path: Optional[str], current_prompt: str = "") -> tuple[Optional[str], Optional[str]]:
    """``(the user's previous request, the agent's last reply)``, either None
    when the transcript does not have it. `current_prompt` is skipped if
    Claude Code has already written it to the transcript."""
    if not transcript_path:
        return None, None
    reply: list[str] = []
    for entry in reversed(_entries(transcript_path)):
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        text = _text(message.get("content"))
        if entry.get("type") == "assistant":
            if text:
                reply.append(text)
        elif entry.get("type") == "user" and text:
            text = text.strip()
            # tagged messages are command output, notifications and reminders
            if not text or text.startswith("<"):
                continue
            if not reply and text == current_prompt.strip():
                continue  # the prompt being routed
            return text, "\n".join(reversed(reply)) or None
    return None, "\n".join(reversed(reply)) or None


def compress(text: Optional[str], chars: int) -> str:
    """`text` as at most `chars` characters of prose: code blocks dropped,
    whitespace collapsed, and when it is too long, its first sentence plus
    its end, cut at a sentence boundary."""
    if not text or chars <= 0:
        return ""
    prose = " ".join(_FENCE.sub(" ", text).split())
    if len(prose) <= chars:
        return prose
    first = _SENTENCE_END.split(prose, 1)[0][: chars // 3]
    budget = chars - len(first) - 5
    end = prose[-budget:] if budget > 0 else ""
    if (cut := _SENTENCE_END.search(end)) and cut.end() < len(end) // 2:
        end = end[cut.end():]  # start the end at a sentence
    return f"{first} ... {end}" if end else first


#: the usage a subagent outcome records, as Claude Code's Agent tool names it
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def subagent_usage(transcript_path: str) -> Optional[dict[str, Any]]:
    """What a subagent's own transcript says it used: tokens summed over its
    API requests, how many there were, the largest prompt one sent (which
    decides a long-context price tier; the sum does not) and its models.
    None when the file is missing or has no usage.

    Claude Code writes a response once per content block, each copy with
    the same message id and usage, so each id counts once. Only usage and
    model fields are read.
    """
    by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f):
                if '"usage"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                message = entry.get("message") if isinstance(entry, dict) else None
                if entry.get("type") != "assistant" or not isinstance(message, dict):
                    continue
                usage, model = message.get("usage"), str(message.get("model") or "")
                if isinstance(usage, dict) and model != "<synthetic>":
                    by_id[str(message.get("id") or n)] = (model, usage)
    except OSError:
        return None
    if not by_id:
        return None
    total = {k: 0 for k in USAGE_KEYS}
    largest = 0
    for _, usage in by_id.values():
        counts = {k: usage.get(k) if isinstance(usage.get(k), int) else 0 for k in USAGE_KEYS}
        for k in USAGE_KEYS:
            total[k] += counts[k]
        largest = max(largest, counts["input_tokens"] + counts["cache_read_input_tokens"]
                      + counts["cache_creation_input_tokens"])
    models = [m for m, _ in by_id.values() if m]
    return {
        **total,
        "requests": len(by_id),
        "max_request_prompt_tokens": largest,
        "resolved_model": models[-1] if models else None,
        "models_used": sorted(set(models)) or None,
    }

