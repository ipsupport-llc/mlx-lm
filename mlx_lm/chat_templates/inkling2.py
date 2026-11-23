# Copyright © 2026 Apple Inc.

"""Response format of Inkling 2.

One turn holds several messages, each opened by <|message_model|> and a
content type, and closed by <|end_message|>:

    <|message_model|><|content_thinking|>reasoning<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name": ..., "args": {...}}<|end_message|>
    <|message_model|><|content_text|>answer<|end_message|>

The text in a header (the role token, a tool's name) is dropped. The state
machine adds the tool call's end transition.
"""

RESPONSE_FORMAT = {
    "initial_state": "header",
    "transitions": {
        "header": [
            ("<|content_thinking|>", "reasoning"),
            ("<|content_text|>", "normal"),
            ("<|content_invoke_tool_json|>", "tool"),
        ],
        "reasoning": [("<|end_message|>", "header")],
        "normal": [("<|end_message|>", "header"), ("<|message_model|>", "header")],
    },
}
