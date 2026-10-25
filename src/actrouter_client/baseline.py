"""The model that would have run without ActRouter -- the routing baseline.

Routing benefits are measured against this model, so it must be what the host
would really have used, found the way the host itself decides it. Every
resolver returns ``(model, source)``: the model as the host spells it (an
alias such as ``sonnet``, or a full id) and where it came from, so a report
can say how each baseline was established. ``(None, "unknown")`` when it
cannot be determined; such calls are counted but not costed.

Claude Code subagent started without a model (checked against Claude Code
2.1.307), first match wins:

1. ``ACTROUTER_BASELINE_MODEL`` -- an explicit override;
2. ``CLAUDE_CODE_SUBAGENT_MODEL`` -- Claude Code's own override for subagents;
3. the subagent definition's ``model:`` (``.claude/agents/<type>.md`` in the
   project, then ``~/.claude/agents``), unless it is ``inherit``;
4. a built-in agent's fixed model (``statusline-setup`` runs on Sonnet);
5. the session's model -- what ``inherit`` means, and the default for the
   built-in Explore, Plan and general-purpose agents -- read from the newest
   assistant message in the session transcript;
6. ``ANTHROPIC_MODEL``, then the ``model`` in Claude Code's settings.

Claude Code main session (a user's prompt): the override, then the model
Claude Code last reported to a hook, then steps 5 and 6.

Agent CLI started by ``actrouter-route launch``: the override, then the
model the CLI is configured with (``~/.codex/config.toml``; Claude Code's
``ANTHROPIC_MODEL`` / settings).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from actrouter_client.client import env

Baseline = tuple[Optional[str], str]
UNKNOWN: Baseline = (None, "unknown")

#: built-in Claude Code subagents whose model is fixed rather than inherited
BUILTIN_SUBAGENT_MODELS = {"statusline-setup": "sonnet"}

#: how much of a transcript's end to scan for the session's model
TRANSCRIPT_TAIL_BYTES = 512 * 1024

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---", re.S)


def override() -> Optional[Baseline]:
    model = env("ACTROUTER_BASELINE_MODEL")
    return (model, "ACTROUTER_BASELINE_MODEL") if model else None


def _frontmatter(path: Path) -> dict[str, str]:
    try:
        match = _FRONTMATTER.match(path.read_text(encoding="utf-8", errors="replace")[:8192])
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in (match.group(1) if match else "").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip().lower()] = value.strip().strip("'\"")
    return out


def _agent_definition_model(subagent_type: str, cwd: Optional[str]) -> Optional[Baseline]:
    """``model:`` of the named subagent's definition file, project first."""
    if not subagent_type:
        return None
    dirs = [Path(cwd) / ".claude" / "agents"] if cwd else []
    dirs.append(Path.home() / ".claude" / "agents")
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.md")):
            meta = _frontmatter(path)
            if subagent_type.lower() in (path.stem.lower(), meta.get("name", "").lower()):
                model = meta.get("model", "")
                if model and model.lower() != "inherit":
                    return model, f"subagent definition {path.name}"
                return None  # found it; it inherits
    return None


def session_model(transcript_path: Optional[str]) -> Optional[str]:
    """The model of the newest assistant message in a Claude Code transcript."""
    if not transcript_path:
        return None
    try:
        with open(transcript_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TRANSCRIPT_TAIL_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if '"model"' not in line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # the first line of the tail may be cut
        message = entry.get("message") if isinstance(entry, dict) else None
        if isinstance(message, dict) and entry.get("type") == "assistant" and message.get("model"):
            model = str(message["model"])
            if model != "<synthetic>":
                return model
    return None


def _claude_settings_model(cwd: Optional[str]) -> Optional[Baseline]:
    if model := env("ANTHROPIC_MODEL"):
        return model, "ANTHROPIC_MODEL"
    files = []
    if cwd:
        files += [Path(cwd) / ".claude" / "settings.local.json", Path(cwd) / ".claude" / "settings.json"]
    files.append(Path.home() / ".claude" / "settings.json")
    for path in files:
        try:
            model = json.loads(path.read_text()).get("model")
        except (OSError, ValueError, AttributeError):
            continue
        if model:
            return str(model), f"settings {path.name}"
    return None


def claude_code_subagent(event: dict[str, Any]) -> Baseline:
    """The model a subagent started without one would run on."""
    tool_input = event.get("tool_input") if isinstance(event.get("tool_input"), dict) else {}
    subagent = str(tool_input.get("subagent_type") or "").strip()
    cwd = event.get("cwd")
    if found := override():
        return found
    if model := env("CLAUDE_CODE_SUBAGENT_MODEL"):
        return model, "CLAUDE_CODE_SUBAGENT_MODEL"
    if found := _agent_definition_model(subagent, cwd):
        return found
    if subagent.lower() in BUILTIN_SUBAGENT_MODELS:
        return BUILTIN_SUBAGENT_MODELS[subagent.lower()], f"built-in {subagent}"
    if model := session_model(event.get("transcript_path")):
        return model, "session model (inherit)"
    return _claude_settings_model(cwd) or UNKNOWN


def claude_code_session(event: dict[str, Any], known: Optional[str] = None) -> Baseline:
    """The model the main session runs on: what the user's request would run
    on without routing. `known` is the model Claude Code last reported to a
    hook (``SessionStart``, ``PostModelSwitch``): fresher than the transcript
    right after ``/model``, and the only source before the first answer.

    This is the model the session really runs on; the baseline for a prompt
    is :func:`override`, else this."""
    if known:
        return known, "session model"
    if model := session_model(event.get("transcript_path")):
        return model, "session model"
    return _claude_settings_model(event.get("cwd")) or UNKNOWN


def _codex_config_model() -> Optional[Baseline]:
    home = Path(env("CODEX_HOME") or Path.home() / ".codex")
    try:
        text = (home / "config.toml").read_text()
    except OSError:
        return None
    # Top-level `model = "..."` only: a [profiles.x] table may set its own.
    for line in text.splitlines():
        if line.startswith("["):
            break
        match = re.match(r'\s*model\s*=\s*["\']([^"\']+)["\']', line)
        if match:
            return match.group(1), "codex config.toml"
    return None


def agent_cli(agent: str, cwd: Optional[str] = None) -> Baseline:
    """The model `agent` (``codex`` | ``claude``) would start on."""
    if found := override():
        return found
    if agent == "codex":
        return _codex_config_model() or UNKNOWN
    if agent == "claude":
        return _claude_settings_model(cwd or os.getcwd()) or UNKNOWN
    return UNKNOWN


# --- resolving a host's spelling to a catalog id ----------------------------------

_DATE_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")
_NUMBER = re.compile(r"\d+")


def _version(model_id: str) -> tuple[int, ...]:
    return tuple(int(n) for n in _NUMBER.findall(model_id))


def to_catalog_id(model: Optional[str], catalog: list[dict[str, Any]]) -> Optional[str]:
    """The catalog id for `model` as a host spells it, or None.

    Accepts a catalog id in any case, an id with a date suffix
    (``claude-sonnet-4-6-20260205``), a provider prefix
    (``anthropic/claude-...``), or an alias (``sonnet``), which -- as Claude
    Code resolves aliases -- means the newest catalog model of that line.
    """
    if not model:
        return None
    ids = [str(m["id"]) for m in catalog if m.get("id")]
    by_lower = {i.lower(): i for i in ids}
    name = model.strip().lower().rsplit("/", 1)[-1].split(":", 1)[-1]
    for candidate in (name, _DATE_SUFFIX.sub("", name)):
        if candidate in by_lower:
            return by_lower[candidate]
    if re.fullmatch(r"[a-z]+", name):  # an alias such as "sonnet"
        matches = [i for i in ids if re.search(rf"\b{name}\b", i.lower().replace("-", " "))]
        if matches:
            return max(matches, key=lambda i: (_version(i), i))
    return None
