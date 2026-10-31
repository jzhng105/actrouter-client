"""A stand-in for ``codex app-server``: the JSON-RPC lines the real one
exchanges (codex-cli 0.166), scripted by the prompt's words.

Every message it receives is appended to ``$FAKE_CODEX_LOG`` (JSON lines).
Prompt words: ``run`` asks to approve a command, ``ask`` sends a server
request the client does not support, ``slow`` waits for ``turn/interrupt``,
``crash`` exits mid-turn, ``reroute`` reports a model reroute.
"""

import json
import os
import sys

LOG = os.environ.get("FAKE_CODEX_LOG")
state = {"turn": 0, "model": "gpt-5.5", "total": 1000}


def log(msg):
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(msg) + "\n")


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    msg = json.loads(line)
    log(msg)
    return msg


def wait_for(rid):
    while True:
        msg = read()
        if msg.get("id") == rid and "method" not in msg:
            return msg


def usage(thread, turn, prompt, cached, output):
    state["total"] += prompt + output
    send({"method": "thread/tokenUsage/updated", "params": {
        "threadId": thread, "turnId": turn,
        "tokenUsage": {
            "total": {"totalTokens": state["total"], "inputTokens": 0, "cachedInputTokens": 0,
                      "cacheWriteInputTokens": 0, "outputTokens": 0, "reasoningOutputTokens": 0},
            "last": {"totalTokens": prompt + output, "inputTokens": prompt, "cachedInputTokens": cached,
                     "cacheWriteInputTokens": 0, "outputTokens": output, "reasoningOutputTokens": 10},
            "modelContextWindow": 258400,
        }}})


def turn(msg):
    p = msg["params"]
    thread, text = p["threadId"], p["input"][0]["text"]
    if p.get("model"):
        state["model"] = p["model"]
    state["turn"] += 1
    tid = f"turn-{state['turn']}"
    send({"id": msg["id"], "result": {"turn": {"id": tid, "status": "inProgress", "items": []}}})
    send({"method": "turn/started", "params": {"threadId": thread, "turn": {"id": tid}}})
    if "crash" in text:
        sys.exit(3)
    send({"method": "item/agentMessage/delta", "params": {"threadId": thread, "turnId": tid, "itemId": "m1",
                                                         "delta": f"on {state['model']}: "}})
    status = "completed"
    if "run" in text:
        send({"id": "srv-1", "method": "item/commandExecution/requestApproval",
              "params": {"threadId": thread, "turnId": tid, "itemId": "c1", "command": "ls -la", "reason": None}})
        decision = wait_for("srv-1")["result"]["decision"]
        send({"method": "item/started", "params": {"threadId": thread, "turnId": tid,
                                                   "item": {"type": "commandExecution", "id": "c1", "command": "ls -la"}}})
        send({"method": "item/agentMessage/delta", "params": {"threadId": thread, "turnId": tid, "itemId": "m1",
                                                             "delta": f"command {decision}"}})
    if "ask" in text:
        send({"id": "srv-2", "method": "item/tool/requestUserInput", "params": {"threadId": thread, "questions": []}})
        wait_for("srv-2")
    if "reroute" in text:
        send({"method": "model/rerouted", "params": {"threadId": thread, "turnId": tid, "fromModel": state["model"],
                                                     "toModel": "gpt-5.4-mini", "reason": "highRiskCyberActivity"}})
        state["model"] = "gpt-5.4-mini"
    if "slow" in text:
        while True:
            m = read()
            if m.get("method") == "turn/interrupt":
                send({"id": m["id"], "result": {}})
                status = "interrupted"
                break
    usage(thread, tid, 30000, 20000, 400)
    send({"method": "thread/tokenUsage/updated", "params": {  # the same request reported again
        "threadId": thread, "turnId": tid,
        "tokenUsage": {"total": {"totalTokens": state["total"]}, "last": {"inputTokens": 1}}}})
    usage(thread, tid, 33000, 30000, 300)
    send({"method": "item/started", "params": {"threadId": "sub-thread", "turnId": "x",
                                               "item": {"type": "commandExecution", "command": "sub-agent"}}})
    send({"method": "item/completed", "params": {"threadId": thread, "turnId": tid,
                                                 "item": {"type": "agentMessage", "id": "m1", "text": "done"}}})
    send({"method": "turn/completed", "params": {"threadId": thread, "turn": {"id": tid, "status": status,
                                                                              "error": None}}})


def main():
    while True:
        msg = read()
        method = msg.get("method")
        if method == "initialize":
            send({"id": msg["id"], "result": {"userAgent": "fake/0", "codexHome": "/tmp", "platformFamily": "unix",
                                              "platformOs": "linux"}})
        elif method == "thread/start":
            state["model"] = msg["params"].get("model") or state["model"]
            send({"id": msg["id"], "result": {"thread": {"id": "thr-1"}, "model": state["model"]}})
        elif method == "thread/resume":
            state["model"] = msg["params"].get("model") or state["model"]
            send({"id": msg["id"], "result": {"thread": {"id": msg["params"]["threadId"]}, "model": state["model"]}})
        elif method == "turn/start":
            turn(msg)
        elif "id" in msg and method:
            send({"id": msg["id"], "error": {"code": -32601, "message": f"unknown {method}"}})


if __name__ == "__main__":
    main()
