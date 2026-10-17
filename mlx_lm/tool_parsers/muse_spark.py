# Copyright © 2026 Apple Inc.

"""Tool call parser for Muse Spark (XML_ATEM)."""

import json
from typing import Any, Optional

import regex as re

tool_call_start = "<atem:function_calls>"
tool_call_end = "</atem:function_calls>"

# The names as response_template in tokenizer_config.json matches them.
_invoke_regex = re.compile(
    r"<atem:invoke\b[^>]*?\bname=\"(?P<name>[^\"]+)\">"
    r"(?P<body>(?:<atem:parameter\b.*?</atem:parameter>|.)*?)"
    r"(?:</atem:invoke>|$)",
    re.DOTALL,
)
_param_regex = re.compile(
    r"<atem:parameter\b[^>]*?\bname=\"(?P<name>[^\"]+)\"[^>]*?>"
    r"(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)


def _string_params(name: str, tools: Optional[Any]) -> set[str]:
    for tool in tools or []:
        func = tool.get("function") or {}
        if func.get("name") == name:
            properties = (func.get("parameters") or {}).get("properties") or {}
            return {k for k, v in properties.items() if v.get("type") == "string"}
    return set()


def _param_value(raw: str, is_string: bool) -> Any:
    # A parameter the schema declares a string stays one ("123", "true").
    if is_string:
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw.strip()


def parse_tool_call(
    model_output: str, tools: Optional[Any] = None
) -> dict[str, Any] | list[dict[str, Any]]:
    calls = []
    for invoke in _invoke_regex.finditer(model_output):
        strings = _string_params(invoke.group("name"), tools)
        arguments = {
            param.group("name"): _param_value(
                param.group("value"), param.group("name") in strings
            )
            for param in _param_regex.finditer(invoke.group("body"))
        }
        calls.append({"name": invoke.group("name"), "arguments": arguments})
    if not calls:
        raise ValueError("No function provided.")
    if len(calls) == 1:
        return calls[0]
    return calls
