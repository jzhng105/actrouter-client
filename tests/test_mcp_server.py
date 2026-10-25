"""The MCP server over real stdio JSON-RPC, as an agent host would drive it."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")


class Session:
    def __init__(self, env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "actrouter_client.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **env},
        )
        self.next_id = 0

    def send(self, method: str, params: dict | None = None, notify: bool = False) -> Any:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self.next_id += 1
            msg["id"] = self.next_id
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        while True:
            line = self.proc.stdout.readline()
            assert line, self.proc.stderr.read()
            reply = json.loads(line)
            if reply.get("id") == self.next_id:
                return reply

    def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        reply = self.send("tools/call", {"name": tool, "arguments": arguments})
        result = reply["result"]
        if result.get("isError"):
            return {"error": result["content"][0]["text"]}
        if result.get("structuredContent") is not None:
            sc = result["structuredContent"]
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        return json.loads(result["content"][0]["text"])

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


@pytest.fixture()
def session(tmp_path: Path, api_url: str):
    s = Session({"ACTROUTER_URL": api_url, "ACTROUTER_HOST": "claude-code", "ACTROUTER_PROJECT_DIR": str(tmp_path)})
    init = s.send(
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
    )
    assert init["result"]["serverInfo"]["name"] == "actrouter"
    assert "ActRouter" in init["result"].get("instructions", "")
    s.send("notifications/initialized", notify=True)
    yield s
    s.close()


def test_lists_read_only_tools(session: Session) -> None:
    tools = {t["name"]: t for t in session.send("tools/list")["result"]["tools"]}
    assert set(tools) == {
        "route_task", "route_function", "route_function_metadata", "list_models", "list_task_types",
        "routing_report",
    }
    assert all(t["annotations"]["readOnlyHint"] for t in tools.values())
    assert tools["route_task"]["inputSchema"]["required"] == ["description"]


def test_route_task(session: Session) -> None:
    advice = session.call("route_task", description="Search the repo for usages of parse_args")
    assert advice["host_model"] == "haiku"
    assert advice["task_type"] == "code_search"
    good = session.call("route_task", description="Fix the failing auth test", preference="quality")
    assert good["host_model"] in {"sonnet", "opus"}


def test_route_function_reads_source_locally(session: Session, tmp_path: Path) -> None:
    (tmp_path / "triage.py").write_text(
        textwrap.dedent(
            '''
            def classify_ticket(ticket: str) -> "Literal['bug', 'feature']":
                """Classify a support ticket."""
            '''
        )
    )
    advice = session.call("route_function", path="triage.py", function="classify_ticket")
    assert advice["task_type"] == "classification"
    assert advice["function"]["module"] == "triage"


def test_route_function_rejects_non_python(session: Session, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("KEY=1")
    assert "Python source only" in session.call("route_function", path=".env", function="x")["error"]


def test_route_function_metadata(session: Session) -> None:
    advice = session.call(
        "route_function_metadata", name="translateLabels", signature="(labels: string[], locale: string): string[]"
    )
    assert advice["task_type"] == "translation"


def test_listings(session: Session) -> None:
    models = session.call("list_models")
    assert {m["host_model"] for m in models["models"]} == {"opus", "sonnet", "haiku"}
    types = session.call("list_task_types")
    assert "debugging" in {t["id"] for t in types["data"]}


def test_current_model_is_compared_recorded_and_reported(session: Session, ledger_file) -> None:
    advice = session.call(
        "route_task", description="Search the repo for usages of parse_args", current_model="sonnet"
    )
    assert advice["baseline_model"].startswith("claude-sonnet")
    assert advice["estimated_savings_usd"] > 0
    [event] = [json.loads(line) for line in ledger_file.read_text().splitlines()]
    assert (event["surface"], event["tool"], event["baseline_source"]) == ("mcp", "route_task", "current_model (agent)")
    report = session.call("routing_report", days=0)
    assert report["calls"]["routed"] == 1 and report["estimated"]["calls"] == 1
