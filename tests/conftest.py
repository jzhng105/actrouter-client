import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator

import pytest

from actrouter_client import Advisor, HttpRouter

from api_replay import ReplayAPI

ENV = (
    "ACTROUTER_URL", "ACTROUTER_HOST", "ACTROUTER_FAMILIES", "ACTROUTER_CANDIDATES",
    "ACTROUTER_MODEL_STYLE", "ACTROUTER_SUBAGENT_ROUTING", "ACTROUTER_PROMPT_ROUTING", "ACTROUTER_PROMPT_MIN_CONFIDENCE", "ACTROUTER_PROMPT_ADVISE_TASKS",
    "ACTROUTER_STATE_DIR", "CLAUDE_PLUGIN_OPTION_PROMPT_ROUTING", "ACTROUTER_HOOK_PROMPT_CHARS",
    "ACTROUTER_HOOK_PREFERENCE", "ACTROUTER_HOOK_DECISION", "ACTROUTER_PROJECT_DIR",
    "CLAUDE_PROJECT_DIR", "ACTROUTER_API_KEY", "CLAUDE_PLUGIN_OPTION_API_KEY", "CLAUDE_PLUGIN_OPTION_ROUTER_URL",
    "CLAUDE_PLUGIN_OPTION_SUBAGENT_ROUTING", "CLAUDE_PLUGIN_OPTION_PREFERENCE",
    "CLAUDE_PLUGIN_OPTION_BASELINE_MODEL", "CLAUDE_PLUGIN_OPTION_LEDGER",
    "ACTROUTER_BASELINE_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL", "ANTHROPIC_MODEL", "CODEX_HOME",
)


@pytest.fixture(scope="session")
def api_url() -> Iterator[str]:
    """The stand-in ActRouter API (see api_replay.py)."""
    api = ReplayAPI()
    yield api.start()
    api.stop()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, api_url: str) -> None:
    for name in ENV:
        # setenv first so teardown also removes what the code under test sets
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    # connect() finds the stand-in API, never the hosted one
    monkeypatch.setenv("ACTROUTER_URL", api_url)


@pytest.fixture(autouse=True)
def ledger_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Every test records to its own ledger, never to ~/.actrouter, and the
    home directory is a temp dir so no real Claude/Codex settings leak in."""
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("ACTROUTER_LEDGER", str(path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return path


@pytest.fixture()
def router(api_url: str) -> HttpRouter:
    return HttpRouter(api_url)


@pytest.fixture()
def claude(router: HttpRouter) -> Advisor:
    return Advisor(router, host="claude-code")


Handler = Callable[[str, str, Any], tuple[int, Any]]


@pytest.fixture()
def http_stub() -> Iterator[Callable[[Handler], str]]:
    """Start an HTTP server answering with `handler(method, path, body)`;
    returns its base URL. Requests are recorded on the handler."""
    servers: list[ThreadingHTTPServer] = []

    def start(handler: Handler) -> str:
        class H(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                status, payload = handler(self.command, self.path, body)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _answer

            def log_message(self, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for s in servers:
        s.shutdown()
