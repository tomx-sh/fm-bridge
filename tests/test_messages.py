import base64

import apple_fm_sdk as fm
import pytest

from fm_bridge.messages import (
    RequestError,
    ToolCall,
    Turn,
    fingerprint,
    parse_messages,
    split_prompt,
    trailing_tool_results,
    transcript_entries,
)

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


def test_parse_collects_instructions_and_turns():
    instructions, turns = parse_messages([
        {"role": "system", "content": "Be brief."},
        {"role": "developer", "content": [{"type": "text", "text": "No emoji."}]},
        {"role": "user", "content": [{"type": "text", "text": "Look"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "data"},
    ])
    assert instructions == "Be brief.\n\nNo emoji."
    assert [t.role for t in turns] == ["user", "assistant", "tool"]
    assert turns[0].images[0].mime == "image/png"
    assert turns[1].tool_calls == [ToolCall("c1", "read", '{"path":"a"}')]
    assert trailing_tool_results(turns) == {"c1": "data"}
    assert split_prompt(turns) == (turns, None)


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "system", "content": "only instructions"}],
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}],
    [{"role": "function", "content": "legacy"}],
])
def test_parse_rejects(messages):
    with pytest.raises(RequestError):
        parse_messages(messages)


def test_fingerprint_ignores_argument_order_and_whitespace():
    a = [Turn("assistant", "Done. ", tool_calls=[ToolCall("c1", "f", '{"a": 1, "b": 2}')])]
    b = [Turn("assistant", "Done.", tool_calls=[ToolCall("c1", "f", '{"b":2,"a":1}')])]
    assert fingerprint("k", a) == fingerprint("k", b)
    assert fingerprint("k", a) != fingerprint("other", a)


async def test_transcript_entries_decode():
    turns = [
        Turn("user", "read a"),
        Turn("assistant", "Let me look.", tool_calls=[ToolCall("c1", "read", '{"path":"a"}')]),
        Turn("tool", "contents", tool_call_id="c1"),
        Turn("assistant", "Done."),
    ]
    seed = fm.LanguageModelSession(instructions="hi")  # a transcript does not keep its session alive
    envelope = await seed.transcript.to_dict()
    envelope["transcript"]["entries"] += transcript_entries(turns)
    decoded = (await (await fm.Transcript.from_dict(envelope)).to_dict())["transcript"]["entries"]
    assert [e["role"] for e in decoded] == ["instructions", "user", "response", "response", "tool", "response"]
    assert decoded[2]["contents"][0]["text"] == "Let me look."
    assert decoded[3]["toolCalls"][0]["name"] == "read"
    assert decoded[4]["toolName"] == "read"
