"""Agent hosts: which models each can run, and how it spells them.

ActRouter routes over its whole catalog and names models by catalog id
(``claude-haiku-4-5``). An agent host can only run some of those, and wants
its own spelling: Claude Code's subagent ``model`` takes an alias
(``haiku``); Codex takes the OpenAI model id. A :class:`Host` captures both,
so an agent is never handed a model it cannot use.

The host comes from ``ACTROUTER_HOST`` (the plugin and Codex config set it)
and the defaults can be widened per machine:

* ``ACTROUTER_FAMILIES=anthropic,openai`` -- e.g. Claude Code behind a
  gateway that serves other vendors;
* ``ACTROUTER_CANDIDATES=id1,id2`` -- an explicit model whitelist;
* ``ACTROUTER_MODEL_STYLE=id`` -- hand Claude Code the full model id rather
  than the alias.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from actrouter_client.client import env

#: Claude Code's subagent model aliases, matched against the catalog id
CLAUDE_ALIASES = ("opus", "sonnet", "haiku", "fable")


def _claude_alias(model_id: str) -> Optional[str]:
    if (env("ACTROUTER_MODEL_STYLE") or "alias").lower() == "id":
        return model_id if model_id.startswith("claude-") else None
    for alias in CLAUDE_ALIASES:
        if re.search(rf"\b{alias}\b", model_id.replace("-", " ")):
            return alias
    return None


def _identity(model_id: str) -> Optional[str]:
    return model_id


@dataclass(frozen=True)
class Host:
    name: str
    #: catalog families the host can run; None = any
    families: Optional[frozenset[str]]
    #: catalog id -> the value the host takes, or None if it cannot run it
    spell: Callable[[str], Optional[str]]
    #: where the spelled value goes, for humans and agents
    usage: str


HOSTS: dict[str, Host] = {
    "claude-code": Host(
        "claude-code",
        frozenset({"anthropic"}),
        _claude_alias,
        "the `model` parameter of the Agent tool, or `model:` in a subagent's frontmatter",
    ),
    "codex": Host(
        "codex",
        frozenset({"openai"}),
        _identity,
        "`codex -m <model>`, `/model`, or `model =` in ~/.codex/config.toml",
    ),
    "generic": Host(
        "generic",
        None,
        _identity,
        "the model id your LLM client takes",
    ),
}

ALIASES = {"claude": "claude-code", "claude_code": "claude-code", "openai-codex": "codex"}


def get_host(name: Optional[str] = None) -> Host:
    raw = (name or env("ACTROUTER_HOST") or "generic").strip().lower()
    key = ALIASES.get(raw, raw)
    if key not in HOSTS:
        raise ValueError(f"unknown host {raw!r}; expected one of {sorted(HOSTS)}")
    return HOSTS[key]


def _csv(value: Optional[str]) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def candidate_models(host: Host, catalog: list[dict[str, Any]]) -> Optional[list[str]]:
    """The model ids to route over for `host`, given the catalog listing
    (``GET /v1/models``). None means the whole catalog."""
    if explicit := _csv(env("ACTROUTER_CANDIDATES")):
        return explicit
    families = frozenset(_csv(env("ACTROUTER_FAMILIES"))) or host.families
    if families is None:
        return None
    ids = [
        m["id"]
        for m in catalog
        if m.get("family") in families and host.spell(m["id"]) is not None
    ]
    return ids or None
