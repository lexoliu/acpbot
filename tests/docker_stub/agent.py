"""Minimal ACP agent for the docker e2e test.

Speaks just enough ACP over newline-delimited JSON-RPC on stdio:
`initialize`, `session/new`, `session/set_*`, `session/prompt`. On prompt it
spawns the chat MCP server advertised in the session cwd's
`.devin/mcp_config.json`, performs the MCP handshake, calls `send_message`,
and reports the daemon's response back through a `session/update` tool_call
notification — so the transcript proves the whole container → bridge →
daemon loop, not just the ACP leg.
"""

import json
import subprocess
import sys


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def mcp_request(proc, req_id, method, params):
    proc.stdin.write(
        json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        + "\n"
    )
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("chat MCP bridge closed its stdout")
        msg = json.loads(line)
        if msg.get("id") == req_id:
            return msg


def on_prompt(req):
    session_id = req["params"]["sessionId"]
    with open(".devin/mcp_config.json") as f:
        server = json.load(f)["mcpServers"]["chat"]

    proc = subprocess.Popen(
        [server["command"], *server["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        mcp_request(
            proc,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "stub-agent", "version": "0.1.0"},
            },
        )
        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        proc.stdin.flush()
        call = mcp_request(
            proc,
            2,
            "tools/call",
            {"name": "send_message", "arguments": {"text": "docker-stub-probe"}},
        )
    finally:
        proc.kill()

    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "stub-call-1",
                    "title": "Called send_message from chat",
                    "rawInput": {"text": "docker-stub-probe"},
                    "rawOutput": call.get("result", call.get("error")),
                    "_meta": {
                        "cognition.ai/inferenceToolName": "mcp__chat__send_message"
                    },
                },
            },
        }
    )
    return {"stopReason": "end_turn"}


RESULTS = {
    "initialize": {
        "protocolVersion": 1,
        "agentCapabilities": {"loadSession": False, "promptCapabilities": {"text": True}},
        "agentInfo": {"name": "stub-agent", "version": "0.1.0"},
    },
    "session/new": {"sessionId": "stub-session"},
    "session/set_mode": {},
    "session/set_config_option": {"configOptions": []},
}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "id" not in msg or "method" not in msg:
            continue
        method = msg["method"]
        result = on_prompt(msg) if method == "session/prompt" else RESULTS.get(method, {})
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})


if __name__ == "__main__":
    main()
