"""ActRouter for agents.

* :func:`connect` -- the ActRouter API, configured from the environment;
* :class:`Advisor` -- routing restricted to what an agent host can run,
  answered in the host's spelling;
* :func:`function_metadata_from_source` -- static metadata for a Python
  function, parsed from source without running it.

Entry points: ``actrouter-mcp`` (MCP server), ``actrouter-claude-hook``
(Claude Code subagent routing) and ``actrouter-route`` (CLI and launcher).
"""

from actrouter_client.advisor import Advisor
from actrouter_client.client import (
    DEFAULT_URL,
    ActRouterError,
    HttpRouter,
    Router,
    connect,
)
from actrouter_client.hosts import HOSTS, Host, get_host
from actrouter_client.metadata import FunctionNotFound, function_metadata_from_source

__all__ = [
    "DEFAULT_URL",
    "HOSTS",
    "ActRouterError",
    "Advisor",
    "FunctionNotFound",
    "Host",
    "HttpRouter",
    "Router",
    "connect",
    "function_metadata_from_source",
    "get_host",
]

__version__ = "0.2.0"
