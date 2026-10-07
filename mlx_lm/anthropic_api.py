# Copyright © 2026 Apple Inc.

"""Anthropic Messages API (``POST /v1/messages``) on top of the server's
chat completions: the request body is converted to a chat completions body,
and the chat completions output (a whole response, or the stream's chunks)
back to Anthropic's message / server-sent events.

Only conversion lives here (plain dicts in, dicts / bytes out); the server
runs the request as any chat request.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def _text_of(content) -> str:
    """The text of a content field: a string, or a list of blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def _image_part(block: dict) -> dict:
    source = block.get("source", {})
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    elif source.get("type") == "url":
        url = source.get("url", "")
    else:
        raise ValueError(f"Unsupported image source type: {source.get('type')!r}")
    return {"type": "image_url", "image_url": {"url": url}}


def _user_messages(content) -> List[dict]:
    """One Anthropic user turn as chat messages: its tool results first (one
    ``tool`` message each, in order), then the rest as a user message."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    out, parts = [], []
    for block in content:
        kind = block.get("type")
        if kind == "tool_result":
            result = block.get("content")
            text = _text_of(result) if not isinstance(result, str) else result
            if block.get("is_error"):
                text = f"Error: {text}"
            out.append(
                {"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": text}
            )
            # A tool message holds text only: its images go with the user part.
            if isinstance(result, list):
                if any(b.get("type") not in ("text", "image") for b in result):
                    raise ValueError("Only text and image tool results are supported.")
                parts.extend(_image_part(b) for b in result if b.get("type") == "image")
        elif kind == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image":
            parts.append(_image_part(block))
        elif kind == "document":
            raise ValueError("Document content blocks are not supported.")
        # Anything else (e.g. cache hints carried as blocks) carries no text.
    if parts:
        if all(p["type"] == "text" for p in parts):
            out.append({"role": "user", "content": "".join(p["text"] for p in parts)})
        else:
            out.append({"role": "user", "content": parts})
    if not out:
        out.append({"role": "user", "content": ""})
    return out


def _assistant_message(content) -> dict:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    text, reasoning, calls = "", "", []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            text += block.get("text", "")
        elif kind == "thinking":
            reasoning += block.get("thinking", "")
        elif kind == "tool_use":
            calls.append(
                {
                    "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        elif kind != "redacted_thinking":
            raise ValueError(f"Unsupported assistant content block: {kind!r}")
    message = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message


def to_chat_request(body: dict) -> dict:
    """An Anthropic messages request body as a chat completions body."""
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise ValueError("Request did not contain messages")
    messages = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system)})
    for m in body["messages"]:
        if m.get("role") == "user":
            messages.extend(_user_messages(m.get("content")))
        elif m.get("role") == "assistant":
            messages.append(_assistant_message(m.get("content")))
        else:
            raise ValueError(f"Unsupported message role: {m.get('role')!r}")

    chat: Dict[str, Any] = {"messages": messages, "stream": bool(body.get("stream", False))}
    for key in ("model", "max_tokens", "temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            chat[key] = body[key]
    if body.get("stop_sequences"):
        chat["stop"] = body["stop_sequences"]

    tools = body.get("tools") or []
    if any(not isinstance(t, dict) or "name" not in t for t in tools):
        raise ValueError("Every tool needs a name.")
    # Server tools (web search etc., they carry a "type") run on Anthropic's
    # side; a local model can't call them.
    tools = [t for t in tools if t.get("type") in (None, "custom")]
    tool_choice = body.get("tool_choice") or {}
    # Generation can't force a call: "tool" offers only that tool, "any" is "auto".
    if tool_choice.get("type") == "tool":
        tools = [t for t in tools if t.get("name") == tool_choice.get("name")]
        if not tools:
            raise ValueError(
                f"tool_choice names a tool this server can't run: {tool_choice.get('name')!r}"
            )
    if tools and tool_choice.get("type") != "none":
        chat["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    thinking = body.get("thinking") or {}
    if thinking.get("type") in ("enabled", "disabled"):
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs["enable_thinking"] = thinking["type"] == "enabled"
        chat["chat_template_kwargs"] = kwargs
    return chat


def _usage(prompt_tokens: int, output_tokens: int, cached: Optional[int]) -> dict:
    cached = max(cached or 0, 0)
    return {
        "input_tokens": max(prompt_tokens - cached, 0),
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


def _tool_use(call: dict) -> dict:
    fn = call.get("function", {})
    try:
        arguments = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        arguments = {}
    tool_id = call.get("id") or uuid.uuid4().hex[:24]
    if not tool_id.startswith("toolu_"):
        tool_id = f"toolu_{tool_id}"
    return {"type": "tool_use", "id": tool_id, "name": fn.get("name", ""), "input": arguments}


def _stop_reason(finish_reason, stop_sequence, made_tool_call) -> str:
    if finish_reason == "stop" and stop_sequence is not None:
        return "stop_sequence"
    if finish_reason == "tool_calls" and not made_tool_call:
        # The call didn't parse; the client has no tool_use block to answer.
        return "end_turn"
    return STOP_REASONS.get(finish_reason, "end_turn")


def message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def to_message(response: dict, model: str, stop_sequence: Optional[str] = None) -> dict:
    """A whole chat completion response as an Anthropic message."""
    choice = response["choices"][0]
    out = choice.get("message", {})
    content = []
    if out.get("reasoning"):
        content.append({"type": "thinking", "thinking": out["reasoning"], "signature": ""})
    if out.get("content"):
        content.append({"type": "text", "text": out["content"]})
    content.extend(_tool_use(c) for c in out.get("tool_calls") or [])
    if not content:
        content.append({"type": "text", "text": ""})
    usage = response.get("usage", {})
    return {
        "id": message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason(
            choice.get("finish_reason"),
            stop_sequence,
            any(b["type"] == "tool_use" for b in content),
        ),
        "stop_sequence": stop_sequence,
        "usage": _usage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        ),
    }


def _event(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


class MessageStream:
    """Chat completion stream chunks as Anthropic message stream events:
    message_start, then a content block per run of thinking / text and one
    per tool call, then message_delta (stop reason, output tokens) and
    message_stop."""

    def __init__(self, model: str):
        self.model = model
        self.index = -1
        self.open: Optional[str] = None  # the open block's type
        self.finish_reason = None
        self.made_tool_call = False

    def start(self, prompt_tokens: int, cached: Optional[int]) -> bytes:
        message = {
            "id": message_id(),
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": _usage(prompt_tokens, 0, cached),
        }
        return _event("message_start", {"type": "message_start", "message": message}) + _event(
            "ping", {"type": "ping"}
        )

    def _close(self) -> bytes:
        if self.open is None:
            return b""
        self.open = None
        return _event("content_block_stop", {"type": "content_block_stop", "index": self.index})

    def _begin(self, block: dict) -> bytes:
        out = self._close()
        self.index += 1
        self.open = block["type"]
        return out + _event(
            "content_block_start",
            {"type": "content_block_start", "index": self.index, "content_block": block},
        )

    def _delta(self, delta: dict) -> bytes:
        return _event(
            "content_block_delta",
            {"type": "content_block_delta", "index": self.index, "delta": delta},
        )

    def chunk(self, response: dict) -> bytes:
        """Events for one chat completion chunk."""
        choice = response["choices"][0]
        delta = choice.get("delta", {})
        out = b""
        if delta.get("reasoning"):
            if self.open != "thinking":
                out += self._begin({"type": "thinking", "thinking": "", "signature": ""})
            out += self._delta({"type": "thinking_delta", "thinking": delta["reasoning"]})
        if delta.get("content"):
            if self.open != "text":
                out += self._begin({"type": "text", "text": ""})
            out += self._delta({"type": "text_delta", "text": delta["content"]})
        for call in delta.get("tool_calls") or []:
            # The server sends a tool call whole, once it has closed.
            block = _tool_use(call)
            arguments = json.dumps(block.pop("input"), ensure_ascii=False)
            out += self._begin(dict(block, input={}))
            out += self._delta({"type": "input_json_delta", "partial_json": arguments})
            out += self._close()
            self.made_tool_call = True
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        return out

    def stop(self, output_tokens: int, stop_sequence: Optional[str] = None) -> bytes:
        reason = _stop_reason(self.finish_reason, stop_sequence, self.made_tool_call)
        # An empty answer still has a text block, as a whole message does.
        empty = self._begin({"type": "text", "text": ""}) if self.index == -1 else b""
        return (
            empty
            + self._close()
            + _event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": reason, "stop_sequence": stop_sequence},
                    "usage": {"output_tokens": output_tokens},
                },
            )
            + _event("message_stop", {"type": "message_stop"})
        )


def error_body(status: int, message: str) -> dict:
    kind = {
        400: "invalid_request_error",
        404: "not_found_error",
        411: "invalid_request_error",
        413: "request_too_large",
    }.get(
        status, "api_error"
    )
    return {"type": "error", "error": {"type": kind, "message": message}}
