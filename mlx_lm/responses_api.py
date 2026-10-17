# Copyright © 2026 Apple Inc.

"""OpenAI Responses API (``POST /v1/responses``) on top of the server's chat
completions: the request body becomes a chat completions body, and the chat
completions output (a whole response, or the stream's chunks) becomes a
response object or the Responses server-sent events.

Stateless: ``previous_response_id`` is refused, the client sends the whole
conversation in ``input`` (Codex does, with ``store: false``). Only
conversion lives here; the server runs the request as any chat request.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from . import anthropic_api

# A custom (freeform) tool takes one string; the model calls it as a function
# with this argument.
CUSTOM_INPUT = "input"


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def call_id(call_id: Optional[str]) -> str:
    """A tool call's id as the client gets it and sends it back."""
    return call_id or _id("call")


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "")
        for p in content
        if p.get("type") in ("input_text", "output_text", "text", "refusal", "summary_text", "reasoning_text")
    )


def _image_part(part: dict) -> dict:
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if not url:
        raise ValueError("Only input_image parts with an image_url are supported.")
    return {"type": "image_url", "image_url": {"url": url}}


def _user_content(content):
    if content is None or isinstance(content, str):
        return content or ""
    parts = []
    for p in content:
        kind = p.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": p.get("text", "")})
        elif kind == "input_image":
            parts.append(_image_part(p))
        else:
            raise ValueError(f"Unsupported input content part: {kind!r}")
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def _tool_output(output) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _text_of(output)
    return json.dumps(output, ensure_ascii=False)


def _messages(body: dict) -> List[dict]:
    messages: List[dict] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise ValueError("Request did not contain input")

    turn: Optional[dict] = None  # the assistant turn being built
    reasoning = ""  # reasoning not yet given to an assistant turn
    unanswered = set()  # call ids of the turn without their output yet
    held: List[str] = []  # system text waiting for the call outputs

    def assistant() -> dict:
        nonlocal turn, reasoning
        # Reasoning after the turn's text or calls starts the next turn.
        if turn is None or (reasoning and (turn["content"] or turn.get("tool_calls"))):
            turn = {"role": "assistant", "content": ""}
            messages.append(turn)
        if reasoning:
            turn["reasoning_content"] = reasoning
            reasoning = ""
        return turn

    def call(name, arguments, cid):
        cid = call_id(cid)
        unanswered.add(cid)
        assistant().setdefault("tool_calls", []).append(
            {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}
        )

    def system(text):
        if unanswered:
            held.append(text)
        else:
            anthropic_api._add_system(messages, text)

    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role == "assistant":
                assistant()["content"] += _text_of(item.get("content"))
            elif role in ("system", "developer"):
                system(_text_of(item.get("content")))
            elif role == "user":
                turn, reasoning = None, ""
                messages.append({"role": "user", "content": _user_content(item.get("content"))})
            else:
                raise ValueError(f"Unsupported message role: {role!r}")
        elif kind == "reasoning":
            # The raw reasoning when the client sent it back, else its summary.
            reasoning += _text_of(item.get("content")) or _text_of(item.get("summary"))
        elif kind == "function_call":
            call(item.get("name", ""), item.get("arguments") or "{}", item.get("call_id"))
        elif kind == "custom_tool_call":
            arguments = json.dumps({CUSTOM_INPUT: item.get("input", "")}, ensure_ascii=False)
            call(item.get("name", ""), arguments, item.get("call_id"))
        elif kind in ("function_call_output", "custom_tool_call_output"):
            turn = None
            unanswered.discard(item.get("call_id"))
            messages.append(
                {"role": "tool", "tool_call_id": item.get("call_id", ""), "content": _tool_output(item.get("output"))}
            )
            if not unanswered:
                for text in held:
                    anthropic_api._add_system(messages, text)
                held = []
        else:
            raise ValueError(f"Unsupported input item: {kind!r}")
    for text in held:
        anthropic_api._add_system(messages, text)
    if all(m["role"] == "system" for m in messages):
        raise ValueError("Request did not contain a user or assistant message")
    return messages


def _tools(body: dict) -> List[dict]:
    """The function and custom tools as chat tools. Hosted tools (web search,
    file search, ...) run on OpenAI's side; a local model can't call them."""
    out = []
    for t in body.get("tools") or []:
        kind = t.get("type")
        if kind == "function":
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            name = fn.get("name")
            parameters = fn.get("parameters") or {"type": "object", "properties": {}}
            description = fn.get("description", "")
        elif kind == "custom":
            name = t.get("name")
            parameters = {
                "type": "object",
                "properties": {CUSTOM_INPUT: {"type": "string"}},
                "required": [CUSTOM_INPUT],
            }
            description = t.get("description", "")
            fmt = t.get("format") or {}
            if fmt.get("type") == "grammar" and fmt.get("definition"):
                description += f"\n\nThe {CUSTOM_INPUT} follows this {fmt.get('syntax', '')} grammar:\n{fmt['definition']}"
        else:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError("Every tool needs a name.")
        out.append({"type": "function", "function": {"name": name, "description": description, "parameters": parameters}})
    return out


def custom_tools(body: dict) -> set:
    return {t.get("name") for t in body.get("tools") or [] if t.get("type") == "custom"}


def to_chat_request(body: dict) -> dict:
    """A Responses request body as a chat completions body."""
    if body.get("previous_response_id"):
        raise ValueError(
            "previous_response_id isn't supported: this server keeps no responses. "
            "Send the whole conversation in input."
        )
    chat: Dict[str, Any] = {"messages": _messages(body), "stream": bool(body.get("stream", False))}
    # top_k and max_tokens: not Responses fields, but a proxy in front (LLMTray's
    # profile) may add them.
    for key in ("model", "temperature", "top_p", "top_k", "max_tokens"):
        if body.get(key) is not None:
            chat[key] = body[key]
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]

    tools = _tools(body)
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("name"):
        # Generation can't force a call: offer only that tool.
        tools = [t for t in tools if t["function"]["name"] == choice["name"]]
        if not tools:
            raise ValueError(f"tool_choice names a tool this server can't run: {choice['name']!r}")
    if tools and choice != "none":
        chat["tools"] = tools

    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if effort in ("none", "minimal"):
            kwargs["enable_thinking"] = False
        else:
            kwargs["enable_thinking"] = True
            kwargs["reasoning_effort"] = effort
        chat["chat_template_kwargs"] = kwargs
    return chat


def _usage(prompt_tokens: int, output_tokens: int, cached: Optional[int]) -> dict:
    return {
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": max(cached or 0, 0)},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt_tokens + output_tokens,
    }


def _response(rid: str, created: int, model: str, body: dict, status: str, output: list, usage) -> dict:
    out = {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "model": model,
        "output": output,
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") or [],
        "instructions": body.get("instructions"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_output_tokens": body.get("max_output_tokens"),
        "previous_response_id": None,
        "reasoning": body.get("reasoning") or {},
        "text": body.get("text") or {"format": {"type": "text"}},
        "store": False,
        "metadata": body.get("metadata") or {},
        "usage": usage,
    }
    return out


def _reasoning_item(text: str) -> dict:
    # Clients show the summary; the raw text (content) comes back next turn.
    return {
        "type": "reasoning",
        "id": _id("rs"),
        "summary": [{"type": "summary_text", "text": text}],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def _message_item(text: str, status: str = "completed") -> dict:
    return {
        "type": "message",
        "id": _id("msg"),
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call_item(call: dict, custom: set, status: str = "completed") -> dict:
    fn = call.get("function", {})
    name, arguments = fn.get("name", ""), fn.get("arguments") or "{}"
    if name in custom:
        try:
            value = json.loads(arguments).get(CUSTOM_INPUT, "")
        except (ValueError, AttributeError):
            value = arguments
        return {"type": "custom_tool_call", "id": _id("ctc"), "status": status,
                "call_id": call_id(call.get("id")), "name": name, "input": value}
    return {"type": "function_call", "id": _id("fc"), "status": status,
            "call_id": call_id(call.get("id")), "name": name, "arguments": arguments}


def _status(finish_reason) -> str:
    return "incomplete" if finish_reason == "length" else "completed"


def to_response(response: dict, model: str, body: dict) -> dict:
    """A whole chat completion response as a Responses response object."""
    choice = response["choices"][0]
    out = choice.get("message", {})
    custom = custom_tools(body)
    output = []
    if out.get("reasoning"):
        output.append(_reasoning_item(out["reasoning"]))
    if out.get("content") or not out.get("tool_calls"):
        output.append(_message_item(out.get("content") or "", _status(choice.get("finish_reason"))))
    output.extend(_call_item(c, custom) for c in out.get("tool_calls") or [])
    usage = response.get("usage", {})
    return _response(
        _id("resp"), int(time.time()), model, body, _status(choice.get("finish_reason")), output,
        _usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
               usage.get("prompt_tokens_details", {}).get("cached_tokens")),
    )


class ResponseStream:
    """Chat completion stream chunks as Responses events: response.created,
    an output item per run of reasoning / text and one per tool call, then
    response.completed with the whole response."""

    def __init__(self, model: str, body: dict):
        self.model = model
        self.body = body
        self.custom = custom_tools(body)
        self.id = _id("resp")
        self.created = int(time.time())
        self.seq = 0
        self.output: List[dict] = []
        self.open: Optional[dict] = None
        self.finish_reason = None
        self.prompt_tokens = 0
        self.cached = None

    def _event(self, name: str, data: dict) -> bytes:
        data = dict(data, type=name, sequence_number=self.seq)
        self.seq += 1
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

    def _snapshot(self, status: str, usage=None) -> dict:
        return _response(self.id, self.created, self.model, self.body, status, self.output, usage)

    def start(self, prompt_tokens: int, cached: Optional[int]) -> bytes:
        self.prompt_tokens, self.cached = prompt_tokens, cached
        snap = self._snapshot("in_progress")
        return self._event("response.created", {"response": snap}) + self._event(
            "response.in_progress", {"response": snap}
        )

    def _close(self) -> bytes:
        item = self.open
        if item is None:
            return b""
        self.open = None
        index = len(self.output) - 1
        out = b""
        if item["type"] == "message":
            part = item["content"][0]
            out += self._event("response.output_text.done",
                               {"item_id": item["id"], "output_index": index, "content_index": 0, "text": part["text"]})
            out += self._event("response.content_part.done",
                               {"item_id": item["id"], "output_index": index, "content_index": 0, "part": part})
            item["status"] = _status(self.finish_reason)
        elif item["type"] == "reasoning":
            part = item["summary"][0]
            out += self._event("response.reasoning_summary_text.done",
                               {"item_id": item["id"], "output_index": index, "summary_index": 0, "text": part["text"]})
            out += self._event("response.reasoning_summary_part.done",
                               {"item_id": item["id"], "output_index": index, "summary_index": 0, "part": part})
        return out + self._event("response.output_item.done", {"output_index": index, "item": item})

    def _begin(self, item: dict) -> bytes:
        out = self._close()
        self.output.append(item)
        self.open = item
        index = len(self.output) - 1
        out += self._event("response.output_item.added", {"output_index": index, "item": item})
        if item["type"] == "message":
            out += self._event("response.content_part.added",
                               {"item_id": item["id"], "output_index": index, "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []}})
        elif item["type"] == "reasoning":
            out += self._event("response.reasoning_summary_part.added",
                               {"item_id": item["id"], "output_index": index, "summary_index": 0,
                                "part": {"type": "summary_text", "text": ""}})
        return out

    def chunk(self, response: dict) -> bytes:
        """Events for one chat completion chunk."""
        choice = response["choices"][0]
        delta = choice.get("delta", {})
        out = b""
        index = lambda: len(self.output) - 1
        if delta.get("reasoning"):
            if self.open is None or self.open["type"] != "reasoning":
                out += self._begin(_reasoning_item(""))
            self.open["summary"][0]["text"] += delta["reasoning"]
            self.open["content"][0]["text"] += delta["reasoning"]
            out += self._event("response.reasoning_summary_text.delta",
                               {"item_id": self.open["id"], "output_index": index(), "summary_index": 0,
                                "delta": delta["reasoning"]})
        if delta.get("content"):
            if self.open is None or self.open["type"] != "message":
                out += self._begin(_message_item("", status="in_progress"))
            self.open["content"][0]["text"] += delta["content"]
            out += self._event("response.output_text.delta",
                               {"item_id": self.open["id"], "output_index": index(), "content_index": 0,
                                "delta": delta["content"]})
        for call in delta.get("tool_calls") or []:
            # The server sends a tool call whole, once it has closed.
            item = _call_item(call, self.custom)
            added = dict(item, status="in_progress")
            out += self._close()
            self.output.append(item)
            i = index()
            if item["type"] == "custom_tool_call":
                out += self._event("response.output_item.added", {"output_index": i, "item": dict(added, input="")})
                out += self._event("response.custom_tool_call_input.delta",
                                   {"item_id": item["id"], "output_index": i, "delta": item["input"]})
                out += self._event("response.custom_tool_call_input.done",
                                   {"item_id": item["id"], "output_index": i, "input": item["input"]})
            else:
                out += self._event("response.output_item.added", {"output_index": i, "item": dict(added, arguments="")})
                out += self._event("response.function_call_arguments.delta",
                                   {"item_id": item["id"], "output_index": i, "delta": item["arguments"]})
                out += self._event("response.function_call_arguments.done",
                                   {"item_id": item["id"], "output_index": i, "arguments": item["arguments"]})
            out += self._event("response.output_item.done", {"output_index": i, "item": item})
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        return out

    def stop(self, output_tokens: int, stop_sequence: Optional[str] = None) -> bytes:
        # An empty answer still has a message, as a whole response does.
        out = self._begin(_message_item("", status="in_progress")) if not self.output else b""
        out += self._close()
        status = _status(self.finish_reason)
        name = "response.incomplete" if status == "incomplete" else "response.completed"
        snap = self._snapshot(status, _usage(self.prompt_tokens, output_tokens, self.cached))
        return out + self._event(name, {"response": snap})


def error_body(status: int, message: str) -> dict:
    kind = "invalid_request_error" if status in (400, 404, 411, 413) else "server_error"
    return {"error": {"message": message, "type": kind, "param": None, "code": None}}
