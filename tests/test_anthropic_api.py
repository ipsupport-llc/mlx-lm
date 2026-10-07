# Copyright © 2026 Apple Inc.

import http
import json
import threading
import unittest

import requests

from mlx_lm import anthropic_api
from mlx_lm.server import APIHandler, LRUPromptCache, ResponseGenerator
from tests.test_server import DummyModelProvider


def events(raw: bytes):
    out = []
    for block in raw.decode().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


class TestRequestConversion(unittest.TestCase):
    def test_conversation_with_tools(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "Be brief."}],
            "stop_sequences": ["END"],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [
                {"name": "read", "description": "Read a file", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            "messages": [
                {"role": "user", "content": "Read a.txt"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Use the tool.", "signature": "x"},
                        {"type": "text", "text": "Reading."},
                        {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.txt"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1",
                         "content": [{"type": "text", "text": "hello"}]},
                        {"type": "text", "text": "Now summarize."},
                    ],
                },
            ],
        }
        chat = anthropic_api.to_chat_request(body)
        self.assertEqual(chat["max_tokens"], 100)
        self.assertEqual(chat["stop"], ["END"])
        self.assertEqual(chat["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["read"])
        roles = [m["role"] for m in chat["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"])
        assistant = chat["messages"][2]
        self.assertEqual(assistant["content"], "Reading.")
        self.assertEqual(assistant["reasoning_content"], "Use the tool.")
        call = assistant["tool_calls"][0]
        self.assertEqual(call["id"], "toolu_1")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "a.txt"})
        self.assertEqual(chat["messages"][3], {"role": "tool", "tool_call_id": "toolu_1", "content": "hello"})
        self.assertEqual(chat["messages"][4]["content"], "Now summarize.")

    def test_tool_choice_none_drops_tools(self):
        chat = anthropic_api.to_chat_request({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "none"},
        })
        self.assertNotIn("tools", chat)

    def test_image_becomes_image_url(self):
        chat = anthropic_api.to_chat_request({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
            {"type": "text", "text": "What is it?"},
        ]}]})
        parts = chat["messages"][0]["content"]
        self.assertEqual(parts[0]["image_url"]["url"], "data:image/png;base64,QUJD")
        self.assertEqual(parts[1], {"type": "text", "text": "What is it?"})

    def test_tool_choice_tool_offers_only_that_tool(self):
        chat = anthropic_api.to_chat_request({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "b"},
        })
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["b"])

    def test_tool_result_image_goes_to_the_user_part(self):
        chat = anthropic_api.to_chat_request({"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "text", "text": "shot"},
                {"type": "image", "source": {"type": "url", "url": "http://x/y.png"}},
            ]},
        ]}]})
        self.assertEqual(chat["messages"][0], {"role": "tool", "tool_call_id": "t", "content": "shot"})
        self.assertEqual(chat["messages"][1]["content"][0]["image_url"]["url"], "http://x/y.png")

    def test_empty_messages_is_an_error(self):
        with self.assertRaises(ValueError):
            anthropic_api.to_chat_request({"messages": []})

    def test_bad_role_is_an_error(self):
        with self.assertRaises(ValueError):
            anthropic_api.to_chat_request({"messages": [{"role": "system", "content": "x"}]})


class TestResponseConversion(unittest.TestCase):
    def test_whole_message(self):
        response = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant", "content": "Let me look.", "reasoning": "hmm",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "read", "arguments": '{"path": "a"}'}}],
                },
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 30}},
        }
        msg = anthropic_api.to_message(response, "m")
        self.assertEqual([b["type"] for b in msg["content"]], ["thinking", "text", "tool_use"])
        self.assertEqual(msg["content"][2]["id"], "toolu_c1")
        self.assertEqual(msg["content"][2]["input"], {"path": "a"})
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual(msg["usage"]["input_tokens"], 20)
        self.assertEqual(msg["usage"]["cache_read_input_tokens"], 30)
        self.assertEqual(msg["usage"]["output_tokens"], 7)

    def test_stream_blocks(self):
        s = anthropic_api.MessageStream("m")
        raw = s.start(10, None)
        for delta, finish in (
            ({"reasoning": "a"}, None),
            ({"reasoning": "b"}, None),
            ({"content": "Hi"}, None),
            ({"tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": '{"p": 1}'}}]}, None),
            ({}, "tool_calls"),
        ):
            raw += s.chunk({"choices": [{"delta": delta, "finish_reason": finish}]})
        raw += s.stop(5)
        got = events(raw)
        names = [n for n, _ in got]
        self.assertEqual(names[:2], ["message_start", "ping"])
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        starts = [d["content_block"]["type"] for n, d in got if n == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text", "tool_use"])
        self.assertEqual(names.count("content_block_stop"), 3)
        tool_json = [d["delta"]["partial_json"] for n, d in got
                     if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"]
        self.assertEqual(json.loads(tool_json[0]), {"p": 1})
        final = got[-2][1]
        self.assertEqual(final["delta"]["stop_reason"], "tool_use")
        self.assertEqual(final["usage"]["output_tokens"], 5)


    def test_stop_reasons(self):
        def stop(finish, seq=None, calls=None):
            r = {"choices": [{"finish_reason": finish, "message": {"content": "x", "tool_calls": calls}}], "usage": {}}
            return anthropic_api.to_message(r, "m", seq)
        self.assertEqual(stop("stop")["stop_reason"], "end_turn")
        m = stop("stop", "END")
        self.assertEqual((m["stop_reason"], m["stop_sequence"]), ("stop_sequence", "END"))
        self.assertEqual(stop("length")["stop_reason"], "max_tokens")
        # A call that didn't parse leaves no tool_use block to answer.
        self.assertEqual(stop("tool_calls")["stop_reason"], "end_turn")
        s = anthropic_api.MessageStream("m")
        s.chunk({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]})
        final = events(s.stop(3, "END"))[-2][1]
        self.assertEqual(final["delta"], {"stop_reason": "stop_sequence", "stop_sequence": "END"})


class TestMessagesEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(DummyModelProvider(), LRUPromptCache())
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()
        cls.response_generator.stop_and_join()

    def body(self, **kw):
        return {"model": "chat_model", "max_tokens": 8,
                "messages": [{"role": "user", "content": "Hello!"}], **kw}

    def test_message(self):
        r = requests.post(f"http://localhost:{self.port}/v1/messages", json=self.body())
        self.assertEqual(r.status_code, 200)
        msg = r.json()
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"][0]["type"], "text")
        self.assertIn(msg["stop_reason"], ("end_turn", "max_tokens"))
        self.assertGreater(msg["usage"]["input_tokens"] + msg["usage"]["cache_read_input_tokens"], 0)
        self.assertGreater(msg["usage"]["output_tokens"], 0)

    def test_stream(self):
        r = requests.post(f"http://localhost:{self.port}/v1/messages", json=self.body(stream=True))
        self.assertEqual(r.status_code, 200)
        got = events(r.content)
        names = [n for n, _ in got]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        self.assertNotIn(b"[DONE]", r.content)
        text = "".join(d["delta"]["text"] for n, d in got
                       if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
        self.assertTrue(text)

    def test_stop_sequence(self):
        plain = requests.post(f"http://localhost:{self.port}/v1/messages", json=self.body(max_tokens=16)).json()
        first = plain["content"][0]["text"]
        word = next(w for w in ("!", ".", ",", "?") if w in first)
        r = requests.post(f"http://localhost:{self.port}/v1/messages",
                          json=self.body(max_tokens=16, stop_sequences=[word]))
        msg = r.json()
        self.assertEqual((msg["stop_reason"], msg["stop_sequence"]), ("stop_sequence", word))
        self.assertNotIn(word, msg["content"][0]["text"] if msg["content"] else "")

    def test_bad_request(self):
        r = requests.post(f"http://localhost:{self.port}/v1/messages", json={"max_tokens": 8})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["type"], "error")
        r = requests.post(f"http://localhost:{self.port}/v1/messages", data=b"{nope",
                          headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
        # The OpenAI routes keep their own error shape.
        r = requests.post(f"http://localhost:{self.port}/v1/chat/completions", data=b"{nope")
        self.assertIsInstance(r.json()["error"], str)


if __name__ == "__main__":
    unittest.main()
