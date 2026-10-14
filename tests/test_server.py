"""End-to-end tests against the on-device model. Skipped when it is unavailable."""

import asyncio
import base64
import json
import re
import struct
import zlib

import apple_fm_sdk as fm
import pytest
from aiohttp.test_utils import TestClient, TestServer

from fm_bridge.bridge import Bridge
from fm_bridge.server import create_app

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not fm.SystemLanguageModel().is_available()[0], reason="on-device model unavailable"),
]

WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]},
}}
SYSTEM = {"role": "system", "content": "You are a helpful assistant. Use the tools when they help."}
# pi-ai's context-overflow detection (packages/ai/src/utils/overflow.ts) matches this.
OVERFLOW = re.compile(r"context[_ ]length[_ ]exceeded", re.I)


async def _client(bridge: Bridge) -> TestClient:
    client = TestClient(TestServer(create_app(bridge)))
    await client.start_server()
    return client


@pytest.fixture
async def bridge():
    return Bridge(tool_timeout=600, idle_ttl=600, max_idle=8)


@pytest.fixture
async def client(bridge):
    client = await _client(bridge)
    yield client
    await client.close()


async def chat(client, messages, **body):
    response = await client.post("/v1/chat/completions", json={"model": "system", "messages": messages, **body})
    return response.status, await response.json()


async def stream(client, messages, **body):
    response = await client.post("/v1/chat/completions", json={
        "model": "system", "messages": messages, "stream": True, "stream_options": {"include_usage": True}, **body})
    assert response.status == 200, await response.text()
    lines = [line for line in (await response.text()).splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    return [json.loads(line[6:]) for line in lines[:-1]]


async def weather_call(client, city="Tokyo"):
    messages = [SYSTEM, {"role": "user", "content": f"What is the weather in {city} right now?"}]
    status, body = await chat(client, messages, tools=[WEATHER])
    assert status == 200, body
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    messages.append(choice["message"])
    return messages, choice["message"]["tool_calls"]


async def test_chat(client):
    status, body = await chat(client, [{"role": "user", "content": "What is the capital of France? One word."}])
    assert status == 200
    assert "paris" in body["choices"][0]["message"]["content"].lower()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] > 0 and body["usage"]["prompt_tokens"] > 0


async def test_stream(client):
    chunks = await stream(client, [{"role": "user", "content": "Count from 1 to 5, separated by commas."}])
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert "1" in text and "5" in text
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] > 0


async def test_tool_round_trip_resumes_the_session(client, bridge):
    messages, calls = await weather_call(client)
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert bridge.stats() == {"suspended": 1}

    messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "Sunny, 31 degrees Celsius"})
    status, body = await chat(client, messages, tools=[WEATHER])
    assert status == 200, body
    answer = body["choices"][0]["message"]["content"]
    assert "31" in answer
    assert bridge.stats() == {"idle": 1}

    # A follow-up continues the cached session instead of rebuilding it.
    messages += [body["choices"][0]["message"], {"role": "user", "content": "Is that hot or cold?"}]
    status, body = await chat(client, messages, tools=[WEATHER])
    assert status == 200, body
    assert len(bridge.conversations) == 1


async def test_streamed_tool_calls(client):
    chunks = await stream(client, [SYSTEM, {"role": "user", "content": "What is the weather in Oslo?"}], tools=[WEATHER])
    calls = [tc for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls") or []]
    assert calls and calls[0]["function"]["name"] == "get_weather" and calls[0]["id"].startswith("call_")
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "tool_calls"


async def test_parallel_tool_calls(client):
    messages = [SYSTEM, {"role": "user", "content": "What is the weather in Tokyo and in Paris?"}]
    status, body = await chat(client, messages, tools=[WEATHER])
    calls = body["choices"][0]["message"]["tool_calls"]
    assert sorted(json.loads(c["function"]["arguments"])["city"] for c in calls) == ["Paris", "Tokyo"]
    messages.append(body["choices"][0]["message"])
    for call, weather in zip(calls, ["Rain, 12 degrees", "Clear, 20 degrees"]):
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": weather})
    status, body = await chat(client, messages, tools=[WEATHER])
    assert status == 200 and body["choices"][0]["finish_reason"] == "stop"


async def test_restart_rebuilds_from_messages(client):
    messages, calls = await weather_call(client)
    messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "Snowing, minus 4 degrees"})
    restarted = await _client(Bridge())  # knows nothing about the pending call
    try:
        status, body = await chat(restarted, messages, tools=[WEATHER])
        assert status == 200, body
        assert re.search(r"snow|-\s?4|minus 4", body["choices"][0]["message"]["content"], re.I)
    finally:
        await restarted.close()


async def test_history_from_another_model(client):
    messages = [
        SYSTEM,
        {"role": "user", "content": "What's the weather in Rome?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "toolu_01", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Rome"}'}}]},
        {"role": "tool", "tool_call_id": "toolu_01", "content": "Foggy, 9 degrees"},
        {"role": "assistant", "content": "It is foggy in Rome, 9 degrees."},
        {"role": "user", "content": "What temperature did you say it was in Rome? Just the number."},
    ]
    status, body = await chat(client, messages, tools=[WEATHER])
    assert status == 200, body
    assert "9" in body["choices"][0]["message"]["content"]


async def test_image_and_follow_up(client, bridge):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "What single color fills this image? One word."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png(220, 20, 60)}"}},
    ]}]
    status, body = await chat(client, messages)
    assert status == 200, body
    assert "red" in body["choices"][0]["message"]["content"].lower()
    # The cached session still refers to the image after the answer.
    messages += [body["choices"][0]["message"], {"role": "user", "content": "Name a fruit of that color."}]
    status, body = await chat(client, messages)
    assert status == 200, body
    assert len(bridge.conversations) == 1


async def test_context_overflow(client):
    status, body = await chat(client, [{"role": "user", "content": "lorem ipsum dolor " * 6000}])
    assert status == 400
    assert body["error"]["code"] == "context_length_exceeded"
    assert OVERFLOW.search(body["error"]["message"])


async def test_oversized_tool_result_is_a_context_overflow(client):
    # The SDK reports this as a bare GenerationError, and the result never reaches the transcript.
    messages, calls = await weather_call(client)
    messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "sunny and warm " * 4000})
    status, body = await chat(client, messages, tools=[WEATHER])
    assert status == 400 and body["error"]["code"] == "context_length_exceeded"


async def test_tool_timeout_cancels_and_rebuild_still_answers(client, bridge):
    bridge.tool_timeout = 0.2
    messages, calls = await weather_call(client)
    await asyncio.sleep(0.4)
    bridge.sweep()
    assert bridge.conversations == [] and bridge.calls == {}
    messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "Windy, 15 degrees"})
    status, body = await chat(client, messages, tools=[WEATHER])
    assert status == 200 and "15" in body["choices"][0]["message"]["content"]


async def test_client_disconnect_closes_the_conversation(client, bridge):
    response = await client.post("/v1/chat/completions", json={
        "model": "system", "stream": True,
        "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]})
    await response.content.readline()
    response.close()
    for _ in range(100):
        await asyncio.sleep(0.05)
        if not bridge.conversations:
            break
    assert bridge.conversations == []
    status, body = await chat(client, [{"role": "user", "content": "Say ok."}])
    assert status == 200


@pytest.mark.parametrize("body, status, param", [
    ({"model": "gpt-4o"}, 404, "model"),
    ({"tool_choice": "required", "tools": [WEATHER]}, 400, "tool_choice"),
    ({"stop": ["\n"]}, 400, "stop"),
    ({"n": 2}, 400, "n"),
    ({"response_format": {"type": "json_object"}}, 400, "response_format"),
    ({"messages": []}, 400, "messages"),
])
async def test_rejects_unsupported_requests(client, body, status, param):
    response = await client.post("/v1/chat/completions", json={
        "model": "system", "messages": [{"role": "user", "content": "hi"}], **body})
    assert response.status == status
    assert (await response.json())["error"]["param"] == param


async def test_models_and_health(client):
    models = await (await client.get("/v1/models")).json()
    assert models["data"][0]["id"] == "system" and models["data"][0]["context_window"] > 0
    health = await client.get("/health")
    assert health.status == 200 and (await health.json())["status"] == "ok"


def _png(r: int, g: int, b: int, size: int = 64) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + bytes([r, g, b]) * size for _ in range(size))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()
