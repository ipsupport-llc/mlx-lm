# Copyright © 2026 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mlx_lm.chat_templates.inkling2 import RESPONSE_FORMAT
from mlx_lm.generate import TextStateMachine, make_text_state_machine
from mlx_lm.tool_parsers import inkling2 as parser


class TestInkling2ResponseFormat(unittest.TestCase):
    def split(self, text, step):
        tokenizer = SimpleNamespace(
            has_thinking=False,
            has_tool_calling=True,
            tool_call_start=parser.tool_call_start,
            tool_call_end=parser.tool_call_end,
            structural_markers=(),
            response_transitions=RESPONSE_FORMAT["transitions"],
        )
        sm = make_text_state_machine(tokenizer)
        state = sm.make_state(RESPONSE_FORMAT["initial_state"])
        out = {"reasoning": "", "tool": "", "normal": ""}
        for i in range(0, len(text), step):
            state, segments = TextStateMachine.split(state, text[i : i + step])
            for name, chunk in segments:
                if name in out:
                    out[name] += chunk
        return out

    def test_messages(self):
        call = '{"name":"get_time","args":{"zone":"UTC"}}'
        text = (
            "<|message_model|><|content_thinking|>think<|end_message|>"
            f"<|message_model|>get_time<|content_invoke_tool_json|>{call}<|end_message|>"
            "<|message_model|><|content_text|>answer<|end_message|>"
        )
        for step in (1, 3, 7, len(text)):
            out = self.split(text, step)
            self.assertEqual(out["reasoning"], "think")
            self.assertEqual(out["tool"], call)
            self.assertEqual(out["normal"], "answer")

    def test_answer_without_thinking(self):
        text = "<|message_model|><|content_text|>Hi there.<|end_message|>"
        for step in (1, 5, len(text)):
            out = self.split(text, step)
            self.assertEqual(out, {"reasoning": "", "tool": "", "normal": "Hi there."})

    def test_parse_tool_call(self):
        self.assertEqual(
            parser.parse_tool_call(' {"name":"f","args":{"a":1}} '),
            {"name": "f", "arguments": {"a": 1}},
        )
        self.assertEqual(
            parser.parse_tool_call('{"name":"f"}'), {"name": "f", "arguments": {}}
        )
        for bad in ('{"args":{}}', "[1]"):
            with self.assertRaises(ValueError):
                parser.parse_tool_call(bad)

    def test_loop_checkpoint_gets_the_format(self):
        from mlx_lm.utils import _tokenizer_defaults

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(
                json.dumps({"model_type": "inkling2_loop"})
            )
            defaults = _tokenizer_defaults(Path(d))
            self.assertEqual(defaults["chat_template_type"], "inkling2")
            self.assertEqual(defaults["tool_parser_type"], "inkling2")


if __name__ == "__main__":
    unittest.main()
