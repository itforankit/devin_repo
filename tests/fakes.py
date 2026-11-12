"""Scriptable fakes for the OpenAI chat API."""

from __future__ import annotations

import json
from types import SimpleNamespace


def tool_call(query: str, call_id: str = "call_1", name: str = "retrieve", raw: str | None = None):
    args = raw if raw is not None else json.dumps({"query": query})
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=args)
    )


def message(content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def stream_events(text: str, tool_calls: list | None = None, size: int = 4):
    events = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content=text[i : i + size], tool_calls=None))
            ]
        )
        for i in range(0, len(text), size)
    ]
    for idx, tc in enumerate(tool_calls or []):
        args = tc.function.arguments
        half = len(args) // 2
        for part, include_meta in ((args[:half], True), (args[half:], False)):
            delta_tc = SimpleNamespace(
                index=idx,
                id=tc.id if include_meta else None,
                function=SimpleNamespace(
                    name=tc.function.name if include_meta else None, arguments=part
                ),
            )
            events.append(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[delta_tc]))
                    ]
                )
            )
    events.append(SimpleNamespace(choices=[]))
    return iter(events)


class FakeCompletions:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        msg = self.script.pop(0)
        if kwargs.get("stream"):
            return stream_events(msg.content or "", msg.tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeChatClient:
    def __init__(self, script: list):
        self.chat = SimpleNamespace(completions=FakeCompletions(script))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls
