# Copyright © 2026 Apple Inc.

import ast
import json
from typing import Any

import regex as re

"""
Tool parser for Olmo 3 function call format.

Parses assistant responses containing tool calls in formats like:
<function_calls>
function_name(arg1="value1", arg2=2)
</function_calls>

Multiple tool calls are newline-separated within the tags. Argument values
are JSON literals (null, true, false) instead of Python literals.
"""


_tool_call_regex = re.compile(r"^\s*(\w+)\((.*)\)\s*$", re.MULTILINE)

_JSON_LITERALS = {"null": None, "true": True, "false": False}


def _coerce(value: str):
    value = value.strip()
    if value in _JSON_LITERALS:
        return _JSON_LITERALS[value]
    try:
        return json.loads(value)
    except ValueError:
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _split_args(args: str):
    """The `key=value` pairs at the top level: commas inside brackets,
    braces or quotes don't split (arrays and objects stay whole)."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(args):
        ch = args[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(args[start:i])
            start = i + 1
        i += 1
    parts.append(args[start:])
    return [p.strip() for p in parts if p.strip()]


def parse_tool_call(text: str, tools: Any | None = None):
    calls = []
    for match in _tool_call_regex.finditer(text):
        func_name = match.group(1)
        args_str = match.group(2)
        arguments = {}
        for part in _split_args(args_str or ""):
            key, sep, value = part.partition("=")
            if not sep:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                arguments[key.strip()] = _coerce(value) if "\\" in value else value[1:-1]
            else:
                arguments[key.strip()] = _coerce(value)
        calls.append({"name": func_name, "arguments": arguments})

    if not calls:
        raise ValueError("No function provided.")

    return calls if len(calls) > 1 else calls[0]


tool_call_start = "<function_calls>"
tool_call_end = "</function_calls>"
