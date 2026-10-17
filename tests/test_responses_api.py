# Copyright © 2026 Apple Inc.

import http
import json
import threading
import unittest

import requests

from mlx_lm import responses_api
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
    def test_codex_style_conversation(self):
        body = {
            "model": "m",
            "instructions": "You are a coding agent.",
            "max_output_tokens": 100,
            "reasoning": {"effort": "high"},
            "tools": [
                {"type": "function", "name": "shell", "description": "Run", "parameters": {"type": "object"}},
                {"type": "custom", "name": "apply_patch", "description": "Patch",
                 "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}},
                {"type": "web_search"},
            ],
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Sandbox: on"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "Use ls."}]},
                {"type": "function_call", "name": "shell", "arguments": "{\"cmd\": \"ls\"}", "call_id": "call_1"},
                {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
                {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin", "call_id": "call_2"},
                {"type": "custom_tool_call_output", "call_id": "call_2", "output": "ok"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done."}]},
            ],
        }
        chat = responses_api.to_chat_request(body)
        self.assertEqual(chat["max_tokens"], 100)
        self.assertEqual(chat["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": "high"})
        msgs = chat["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": "You are a coding agent.\n\nSandbox: on"})
        self.assertEqual(msgs[1], {"role": "user", "content": "List files"})
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[2]["reasoning_content"], "Use ls.")
        self.assertEqual(msgs[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(msgs[3], {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"})
        self.assertEqual(json.loads(msgs[4]["tool_calls"][0]["function"]["arguments"]), {"input": "*** Begin"})
        self.assertEqual(msgs[5], {"role": "tool", "tool_call_id": "call_2", "content": "ok"})
        self.assertEqual(msgs[6], {"role": "assistant", "content": "Done."})
        names = [t["function"]["name"] for t in chat["tools"]]
        self.assertEqual(names, ["shell", "apply_patch"])
        self.assertIn("start: x", chat["tools"][1]["function"]["description"])

    def test_string_input_and_no_thinking(self):
        chat = responses_api.to_chat_request({"input": "Hi", "reasoning": {"effort": "minimal"}})
        self.assertEqual(chat["messages"], [{"role": "user", "content": "Hi"}])
        self.assertEqual(chat["chat_template_kwargs"], {"enable_thinking": False})

    def test_image_input(self):
        chat = responses_api.to_chat_request({"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "What?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]}]})
        self.assertEqual(chat["messages"][0]["content"][1],
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})

    def test_errors(self):
        for body in ({}, {"input": []}, {"input": "x", "previous_response_id": "resp_1"},
                     {"input": [{"type": "message", "role": "robot", "content": "x"}]},
                     {"input": [{"type": "file_search_call"}]}):
            with self.assertRaises(ValueError):
                responses_api.to_chat_request(body)

    def test_tool_choice(self):
        tools = [{"type": "function", "name": "a"}, {"type": "function", "name": "b"}]
        chat = responses_api.to_chat_request({"input": "x", "tools": tools, "tool_choice": {"type": "function", "name": "b"}})
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["b"])
        self.assertNotIn("tools", responses_api.to_chat_request({"input": "x", "tools": tools, "tool_choice": "none"}))


    def test_turn_order(self):
        msgs = responses_api.to_chat_request({"input": [
            {"role": "user", "content": "go"},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "R1"}]},
            {"type": "message", "role": "assistant", "content": "A"},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "R2"}]},
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"},
            {"type": "message", "role": "assistant", "content": " (running)"},
            {"type": "message", "role": "developer", "content": "Mode changed."},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "stale"}]},
            {"role": "user", "content": "next"},
        ]})["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "assistant", "tool", "user"])
        self.assertEqual((msgs[1]["reasoning_content"], msgs[1]["content"]), ("R1", "A"))
        # Text after a call stays in its turn, so the output still follows it.
        self.assertEqual((msgs[2]["reasoning_content"], msgs[2]["content"]), ("R2", " (running)"))
        # Instructions sent between a call and its output wait for the output.
        self.assertEqual(msgs[3]["content"], "ok\n\nMode changed.")
        self.assertNotIn("reasoning_content", msgs[4])


class TestResponseConversion(unittest.TestCase):
    def test_whole_response(self):
        chat = {
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": "Patching.", "reasoning": "Think.",
                "tool_calls": [
                    {"id": "call_9", "function": {"name": "apply_patch", "arguments": "{\"input\": \"*** Begin\"}"}},
                    {"id": "call_8", "function": {"name": "shell", "arguments": "{\"cmd\": \"ls\"}"}},
                ]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4}},
        }
        body = {"input": "x", "tools": [{"type": "custom", "name": "apply_patch"}]}
        r = responses_api.to_response(chat, "m", body)
        self.assertEqual(r["object"], "response")
        self.assertEqual(r["status"], "completed")
        self.assertEqual([o["type"] for o in r["output"]], ["reasoning", "message", "custom_tool_call", "function_call"])
        self.assertEqual(r["output"][2]["input"], "*** Begin")
        self.assertEqual(r["output"][3]["call_id"], "call_8")
        self.assertEqual(r["usage"]["input_tokens_details"]["cached_tokens"], 4)
        self.assertEqual(r["usage"]["total_tokens"], 15)

    def test_length_is_incomplete(self):
        chat = {"choices": [{"finish_reason": "length", "message": {"content": "abc"}}], "usage": {}}
        r = responses_api.to_response(chat, "m", {"input": "x"})
        self.assertEqual((r["status"], r["incomplete_details"]), ("incomplete", {"reason": "max_output_tokens"}))
        self.assertEqual(r["output"][0]["status"], "incomplete")

    def test_empty_stream_has_a_message(self):
        s = responses_api.ResponseStream("m", {"input": "x"})
        got = events(s.start(1, 0) + s.chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + s.stop(0))
        self.assertEqual(got[-1][1]["response"]["output"][0]["type"], "message")

    def test_stream_events(self):
        s = responses_api.ResponseStream("m", {"input": "x"})
        raw = s.start(10, 2)
        raw += s.chunk({"choices": [{"delta": {"reasoning": "Hm."}}]})
        raw += s.chunk({"choices": [{"delta": {"content": "Hi "}}]})
        raw += s.chunk({"choices": [{"delta": {"content": "there"}}]})
        raw += s.chunk({"choices": [{"delta": {"tool_calls": [
            {"id": "call_1", "function": {"name": "shell", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]})
        raw += s.stop(7)
        got = events(raw)
        names = [n for n, _ in got]
        self.assertEqual(names[:2], ["response.created", "response.in_progress"])
        self.assertEqual(names[-1], "response.completed")
        self.assertEqual([d["sequence_number"] for _, d in got], list(range(len(got))))
        text = "".join(d["delta"] for n, d in got if n == "response.output_text.delta")
        self.assertEqual(text, "Hi there")
        done = [d["item"]["type"] for n, d in got if n == "response.output_item.done"]
        self.assertEqual(done, ["reasoning", "message", "function_call"])
        final = got[-1][1]["response"]
        self.assertEqual(final["output"][1]["content"][0]["text"], "Hi there")
        self.assertEqual(final["usage"]["output_tokens"], 7)


class TestResponsesEndpoint(unittest.TestCase):
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
        return {"model": "chat_model", "max_output_tokens": 8, "input": "Hello!", **kw}

    def test_response(self):
        r = requests.post(f"http://localhost:{self.port}/v1/responses", json=self.body())
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertEqual(out["object"], "response")
        self.assertIn(out["status"], ("completed", "incomplete"))
        self.assertEqual(out["output"][-1]["type"], "message")
        self.assertGreater(out["usage"]["output_tokens"], 0)

    def test_stream(self):
        r = requests.post(f"http://localhost:{self.port}/v1/responses", json=self.body(stream=True))
        self.assertEqual(r.status_code, 200)
        got = events(r.content)
        names = [n for n, _ in got]
        self.assertEqual(names[0], "response.created")
        self.assertIn(names[-1], ("response.completed", "response.incomplete"))
        self.assertNotIn(b"[DONE]", r.content)
        self.assertTrue("".join(d["delta"] for n, d in got if n == "response.output_text.delta"))

    def test_bad_request(self):
        r = requests.post(f"http://localhost:{self.port}/v1/responses", json={"model": "chat_model"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
