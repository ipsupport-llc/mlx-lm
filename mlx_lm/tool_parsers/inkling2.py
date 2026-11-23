# Copyright © 2026 Apple Inc.

"""Inkling 2 tool calls: {"name": ..., "args": {...}} in a message of its own."""

import json

tool_call_start = "<|content_invoke_tool_json|>"

tool_call_end = "<|end_message|>"


def parse_tool_call(text, tools=None):
    call = json.loads(text.strip())
    # The server drops a call it can't parse on ValueError.
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError(f"not an Inkling 2 tool call: {text!r}")
    return {"name": call["name"], "arguments": call.get("args") or {}}
