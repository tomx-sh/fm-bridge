"""OpenAI Chat Completions over HTTP, served from the bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

import apple_fm_sdk as fm
from aiohttp import web

from .bridge import Bridge, Conversation, Finished, PendingCall, TextDelta, ToolCalls
from .messages import RequestError, Turn, parse_messages

log = logging.getLogger("fm_bridge")

MODEL_ID = "system"
_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
BRIDGE = web.AppKey("bridge", Bridge)


@dataclass
class ChatRequest:
    instructions: str | None
    turns: list[Turn]
    tools: list[dict]
    options: fm.GenerationOptions | None
    stream: bool
    include_usage: bool


def create_app(bridge: Bridge) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)  # requests carry base64 images
    app[BRIDGE] = bridge
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.cleanup_ctx.append(_sweeper)
    return app


async def _sweeper(app: web.Application):
    async def run() -> None:
        while True:
            await asyncio.sleep(15)
            app[BRIDGE].sweep()

    task = asyncio.create_task(run())
    yield
    task.cancel()
    for conversation in app[BRIDGE].conversations:
        conversation.close()


def parse_request(body: Any) -> ChatRequest:
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object")
    model = body.get("model")
    if model != MODEL_ID:
        raise RequestError(f"The model {model!r} does not exist; use {MODEL_ID!r}", status=404,
                           code="model_not_found", param="model")
    if body.get("n", 1) != 1:
        raise RequestError("Only n=1 is supported", param="n")
    if body.get("stop"):
        raise RequestError("Stop sequences are not supported", param="stop")
    response_format = body.get("response_format") or {}
    if response_format.get("type", "text") != "text":
        raise RequestError("Only text responses are supported", param="response_format")

    tools = body.get("tools") or []
    names = set()
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        if tool.get("type") != "function" or not isinstance(function, dict) or not function.get("name"):
            raise RequestError(f"tools[{index}] must be a function with a name", param="tools")
        if function["name"] in names:
            raise RequestError(f"Duplicate tool name {function['name']!r}", param="tools")
        names.add(function["name"])
    tool_choice = body.get("tool_choice") or "auto"
    if tool_choice == "none":
        tools = []
    elif tool_choice != "auto":
        # The framework decides when to call tools; it cannot be forced to.
        raise RequestError("Only tool_choice 'auto' and 'none' are supported", param="tool_choice")

    instructions, turns = parse_messages(body.get("messages"))
    stream_options = body.get("stream_options") or {}
    return ChatRequest(
        instructions=instructions,
        turns=turns,
        tools=tools,
        options=_options(body),
        stream=bool(body.get("stream")),
        include_usage=bool(stream_options.get("include_usage")),
    )


def _options(body: dict) -> fm.GenerationOptions | None:
    temperature = body.get("temperature")
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    top_p, seed = body.get("top_p"), body.get("seed")
    if temperature is None and max_tokens is None and top_p is None and seed is None:
        return None
    sampling = fm.SamplingMode.random(probability_threshold=top_p, seed=seed) if top_p is not None or seed is not None else None
    try:
        return fm.GenerationOptions(sampling=sampling, temperature=temperature, maximum_response_tokens=max_tokens)
    except (TypeError, ValueError) as error:
        raise RequestError(f"Invalid sampling options: {error}") from error


async def chat_completions(request: web.Request) -> web.StreamResponse:
    bridge: Bridge = request.app[BRIDGE]
    started = time.monotonic()
    try:
        chat = parse_request(await request.json())
        conversation, route = await bridge.begin(chat.instructions, chat.turns, chat.tools, chat.options)
    except json.JSONDecodeError:
        return _error_response(RequestError("The request body is not valid JSON"), bridge)
    except Exception as error:
        return _error_response(error, bridge)

    completion = _Completion(f"chatcmpl-{secrets.token_hex(12)}")
    try:
        if chat.stream:
            response = await _stream(request, conversation, chat, completion, bridge)
        else:
            response = await _collect(conversation, completion, bridge)
    finally:
        bridge.release(conversation)
    tokens = (completion.usage or {}).get("total_tokens", "?")
    log.info("conversation %s (%s): %s in %.1fs, %s tokens in context", conversation.id, route,
             completion.finish or "error", time.monotonic() - started, tokens)
    return response


class _Completion:
    def __init__(self, id: str):
        self.id = id
        self.created = int(time.time())
        self.text = ""
        self.calls: list[PendingCall] = []
        self.finish: str | None = None
        self.usage: dict | None = None

    def apply(self, event: TextDelta | ToolCalls | Finished) -> None:
        if isinstance(event, TextDelta):
            self.text += event.text
        elif isinstance(event, ToolCalls):
            self.calls, self.finish, self.usage = event.calls, "tool_calls", event.usage
        else:
            self.finish, self.usage = event.reason, event.usage

    def chunk(self, delta: dict, finish: str | None = None) -> dict:
        return {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _call_json(call: PendingCall) -> dict:
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}


async def _collect(conversation: Conversation, completion: _Completion, bridge: Bridge) -> web.Response:
    try:
        async for event in conversation.events():
            completion.apply(event)
    except asyncio.CancelledError:
        conversation.close()
        raise
    except Exception as error:
        return _error_response(error, bridge)
    message: dict = {"role": "assistant", "content": completion.text or (None if completion.calls else "")}
    if completion.calls:
        message["tool_calls"] = [_call_json(c) for c in completion.calls]
    return web.json_response({
        "id": completion.id,
        "object": "chat.completion",
        "created": completion.created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": completion.finish or "stop"}],
        "usage": completion.usage or _ZERO_USAGE,
    })


async def _stream(request: web.Request, conversation: Conversation, chat: ChatRequest,
                  completion: _Completion, bridge: Bridge) -> web.StreamResponse:
    # Headers wait for the first event, so an error raised before any output (such as a context
    # overflow) still gets a real HTTP status.
    response: web.StreamResponse | None = None

    async def send(payload: dict | str) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        await response.write(f"data: {data}\n\n".encode())

    async def open_stream() -> None:
        nonlocal response
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)
        await send(completion.chunk({"role": "assistant", "content": ""}))

    try:
        async for event in conversation.events():
            if response is None:
                await open_stream()
            completion.apply(event)
            if isinstance(event, TextDelta):
                await send(completion.chunk({"content": event.text}))
            elif isinstance(event, ToolCalls):
                calls = [{"index": i, **_call_json(c)} for i, c in enumerate(event.calls)]
                await send(completion.chunk({"tool_calls": calls}))
        if response is None:
            await open_stream()
        await send(completion.chunk({}, completion.finish or "stop"))
        if chat.include_usage:
            await send({**completion.chunk({}), "choices": [], "usage": completion.usage or _ZERO_USAGE})
        await send("[DONE]")
        await response.write_eof()
    except (ConnectionResetError, asyncio.CancelledError):
        conversation.close()  # the client went away; stop generating for it
        raise
    except Exception as error:
        if response is None:
            return _error_response(error, bridge)
        status, body = _classify(error, bridge)
        await send(body)
        await send("[DONE]")
        await response.write_eof()
    return response


def _classify(error: Exception, bridge: Bridge) -> tuple[int, dict]:
    def body(kind: str, code: str, message: str, param: str | None = None) -> dict:
        return {"error": {"message": message, "type": kind, "code": code, "param": param}}

    if isinstance(error, RequestError):
        return error.status, body("invalid_request_error", error.code, str(error), error.param)
    if isinstance(error, fm.ExceededContextWindowSizeError):
        return 400, body("invalid_request_error", "context_length_exceeded",
                         f"The conversation exceeds the context window of {bridge.model.context_size} tokens "
                         "(context_length_exceeded).")
    if isinstance(error, (fm.RateLimitedError, fm.ConcurrentRequestsError)):
        return 429, body("rate_limit_exceeded", "rate_limited", str(error))
    if isinstance(error, fm.AssetsUnavailableError):
        return 503, body("server_error", "model_unavailable", str(error))
    if isinstance(error, fm.UnsupportedLanguageOrLocaleError):
        return 400, body("invalid_request_error", "unsupported_language", str(error))
    if isinstance(error, (fm.InvalidGenerationSchemaError, fm.UnsupportedGuideError)):
        return 400, body("invalid_request_error", "invalid_tool_schema", str(error), "tools")
    if isinstance(error, (fm.GuardrailViolationError, fm.RefusalError)):
        return 400, body("invalid_request_error", "content_filter", str(error))
    if isinstance(error, fm.FoundationModelsError):
        return 500, body("server_error", "model_error", str(error))
    log.exception("unexpected error", exc_info=error)
    return 500, body("server_error", "internal_error", f"{type(error).__name__}: {error}")


def _error_response(error: Exception, bridge: Bridge) -> web.Response:
    status, body = _classify(error, bridge)
    if status >= 500 or not isinstance(error, RequestError):
        log.warning("request failed: %s", body["error"]["message"])
    return web.json_response(body, status=status)


async def models(request: web.Request) -> web.Response:
    bridge: Bridge = request.app[BRIDGE]
    return web.json_response({"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "apple",
        "context_window": bridge.model.context_size,
    }]})


async def health(request: web.Request) -> web.Response:
    bridge: Bridge = request.app[BRIDGE]
    available, reason = bridge.model.is_available()
    variant = getattr(bridge.model, "variant", None)
    return web.json_response({
        "status": "ok" if available else "unavailable",
        "reason": str(reason) if reason else None,
        "model": MODEL_ID,
        "variant": getattr(variant, "display_name", None),
        "context_window": bridge.model.context_size,
        "conversations": bridge.stats(),
    }, status=200 if available else 503)
