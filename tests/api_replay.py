"""A stand-in ActRouter API for the tests: real engine answers, replayed.

The routing engine is not part of this repository, so the tests run against
an HTTP server that answers each request with the response the real API gave
to that exact request, recorded in ``fixtures/api_replay.json``.
``ACTROUTER_TEST_ROUTER`` picks where answers come from:

* ``replay`` (default) -- the recording; an unrecorded request fails with
  HTTP 599 and says how to re-record.
* ``live`` -- the engine (the ``actrouter`` package, importable), called
  through its FastAPI app with auth off. The engine repository runs the
  client's tests this way to check the two still agree.
* ``record`` -- like ``live``, and rewrite the recording with every request
  this run made. Run the whole suite when recording.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

RECORDING = Path(__file__).parent / "fixtures" / "api_replay.json"
MODES = ("replay", "live", "record")
NOT_RECORDED = 599

Answer = tuple[int, Any]


def request_key(method: str, path: str, body: Any) -> str:
    return f"{method} {path} {json.dumps(body, sort_keys=True, separators=(',', ':'))}"


def _engine() -> Callable[[str, str, Any], Answer]:
    from fastapi.testclient import TestClient

    from actrouter.api import create_app

    client = TestClient(create_app(require_auth=False))

    def call(method: str, path: str, body: Any) -> Answer:
        r = client.request(method, path, json=body)
        return r.status_code, r.json()

    return call


class ReplayAPI:
    def __init__(self, mode: Optional[str] = None) -> None:
        self.mode = mode or os.environ.get("ACTROUTER_TEST_ROUTER", "replay")
        if self.mode not in MODES:
            raise ValueError(f"ACTROUTER_TEST_ROUTER must be one of {MODES}, not {self.mode!r}")
        self.recorded: dict[str, dict[str, Any]] = {}
        if self.mode == "replay" and RECORDING.exists():
            self.recorded = json.loads(RECORDING.read_text())
        self.seen: dict[str, dict[str, Any]] = {}
        self._engine = _engine() if self.mode != "replay" else None
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None

    def answer(self, method: str, path: str, body: Any) -> Answer:
        key = request_key(method, path, body)
        if self._engine is None:
            hit = self.recorded.get(key)
            if hit is None:
                return NOT_RECORDED, {"detail": (
                    f"not recorded: {key}. Re-record with the engine installed: "
                    "ACTROUTER_TEST_ROUTER=record pytest"
                )}
            return hit["status"], hit["body"]
        status, payload = self._engine(method, path, body)
        with self._lock:
            self.seen.setdefault(key, {"status": status, "body": payload})
        return status, payload

    def start(self) -> str:
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                status, payload = api.answer(self.command, self.path, body)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _answer

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self.mode == "record":
            RECORDING.parent.mkdir(exist_ok=True)
            RECORDING.write_text(json.dumps(dict(sorted(self.seen.items())), indent=1) + "\n")
