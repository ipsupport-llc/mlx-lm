# Copyright © 2026 Apple Inc.

"""Inkling 2 tool calls: {"name": ..., "args": {...}} in a message of its own."""

import json

tool_call_start = "<|content_invoke_tool_json|>"

tool_call_end = "<|end_message|>"


def parse_tool_call(text, tools=None):
    call = json.loads(text.strip())
    return {"name": call["name"], "arguments": call.get("args", {})}
