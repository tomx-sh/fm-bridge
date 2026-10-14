"""Foundation Models sessions whose tool calls are answered by the HTTP client.

Every tool the client offers becomes an `fm.Tool` whose `call()` runs nothing. It hands the
call to the HTTP layer, which returns it to the client as OpenAI `tool_calls`, and then waits.
The client runs the tool and sends the result in its next request; the bridge matches it by
tool-call id, completes the waiting `call()`, and the same generation carries on. That keeps
the framework in charge of the tool loop, so the model sees tool calls and results in the
format it was trained on.

A conversation the bridge does not hold (server restart, eviction, edited history, turns
written by another model) is rebuilt from the request's messages through a transcript.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import apple_fm_sdk as fm
from apple_fm_sdk import _ctypes_bindings as _native

from .messages import Turn, fingerprint, split_prompt, trailing_tool_results, transcript_entries
from .schema import coerce_arguments, tool_schema

log = logging.getLogger("fm_bridge")

# Parallel tool calls reach the bridge within milliseconds of each other; wait this long for
# the rest before handing the batch to the client.
SIBLING_WINDOW = 0.15
MISSING_RESULT = "Error: the client returned no result for this tool call."
CANCELLED_RESULT = "Error: the tool call was cancelled."
_IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/heic": ".heic",
                   "image/gif": ".gif", "image/webp": ".webp", "image/tiff": ".tiff", "image/bmp": ".bmp"}


@dataclass
class PendingCall:
    id: str
    name: str
    arguments: str  # JSON text
    future: concurrent.futures.Future


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCalls:
    calls: list[PendingCall]
    usage: dict | None


@dataclass
class Finished:
    reason: str  # "stop", "length" or "content_filter"
    usage: dict | None


class _ClientTool(fm.Tool):
    """A tool the client runs. `call()` parks the call until the client sends its result."""

    def __init__(self, conversation: Conversation, spec: dict):
        function = spec["function"]
        self.name = function["name"]
        self.description = function.get("description") or ""
        self.parameters = function.get("parameters") or {"type": "object", "properties": {}}
        self._schema = tool_schema(self.name, self.parameters)
        self._conversation = conversation
        super().__init__()

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return self._schema

    async def call(self, args: fm.GeneratedContent) -> str:
        # Runs on a thread the SDK starts for the call; the result arrives from the server's loop.
        arguments = coerce_arguments(json.loads(args.to_json()), self.parameters)
        call = PendingCall(f"call_{secrets.token_hex(12)}", self.name, json.dumps(arguments), concurrent.futures.Future())
        self._conversation.post("tool_call", call)
        return await asyncio.wrap_future(call.future)


class Conversation:
    """One Foundation Models session and the generation running in it.

    States: generating → suspended (tool calls out) → generating … → idle (answered), or closed.
    """

    def __init__(self, bridge: Bridge, key: str, tools: list[dict]):
        self.bridge = bridge
        self.key = key  # instructions and tools
        self.id = secrets.token_hex(3)
        self.loop = asyncio.get_running_loop()
        self.tools = [_ClientTool(self, spec) for spec in tools]
        self.session: fm.LanguageModelSession | None = None
        self.state = "new"
        self.busy = False  # a request is reading this conversation's events
        self.history_key: str | None = None  # the turns the session holds while idle
        self.turns: list[Turn] = []  # the turns of the request being served
        self.pending: dict[str, PendingCall] = {}
        self.touched = time.monotonic()
        self.max_tokens: int | None = None
        self._prompt: Any = None
        self._events: asyncio.Queue = asyncio.Queue()
        self._held: list[tuple[str, Any]] = []
        self._snapshot = ""
        self._text = ""
        self._generation: tuple[asyncio.AbstractEventLoop, asyncio.Task] | None = None
        self._files: Path | None = None

    async def open(self, instructions: str | None, history: list[Turn]) -> None:
        session = fm.LanguageModelSession(instructions=instructions, tools=self.tools)
        if history:
            # Keep the framework's own instructions entry: it renders the tool definitions.
            envelope = await session.transcript.to_dict()
            envelope["transcript"]["entries"] += transcript_entries(history)
            transcript = await fm.Transcript.from_dict(envelope)
            holder = transcript.session_ptr
            session = fm.LanguageModelSession.from_transcript(transcript, tools=self.tools)
            # `Transcript.from_dict` creates a native session that the SDK never releases; the
            # new session copied its transcript, so release it here.
            _native.FMRelease(holder)
        self.session = session

    def prompt(self, turn: Turn | None) -> Any:
        if turn is None:
            return ""  # continue after tool results the bridge did not hand out
        if not turn.images:
            return turn.text
        if self._files is None:
            self._files = Path(tempfile.mkdtemp(prefix="fm-bridge-"))
        parts: list[Any] = [turn.text] if turn.text else []
        for image in turn.images:
            path = self._files / f"{secrets.token_hex(6)}{_IMAGE_SUFFIXES.get(image.mime, '.img')}"
            path.write_bytes(image.data)
            parts.append(fm.ImageAttachment(path))
        return parts

    def start(self, prompt: Any, options: fm.GenerationOptions | None, turns: list[Turn]) -> None:
        self.turns = turns
        self._prompt = prompt
        self.state = "generating"
        self.max_tokens = options.maximum_response_tokens if options else None
        self._text = ""
        threading.Thread(target=self._generate, args=(prompt, options), name=f"fm-{self.id}", daemon=True).start()

    def _generate(self, prompt: Any, options: fm.GenerationOptions | None) -> None:
        # The SDK's stream iterator blocks its event loop while it waits for output, so each
        # generation gets a thread and loop of its own; events reach the server through post().
        async def run() -> None:
            self._generation = (asyncio.get_running_loop(), asyncio.current_task())
            try:
                async for snapshot in self.session.stream_response(prompt, options):
                    self.post("snapshot", snapshot)
                    await asyncio.sleep(0)  # lets close() cancel the stream here
                self.post("done", None)
            except asyncio.CancelledError:
                self.post("cancelled", None)
            except Exception as error:
                self.post("error", error)

        asyncio.run(run())

    def post(self, kind: str, value: Any) -> None:
        """Queue an event for the server's loop. Safe from any thread."""
        try:
            self.loop.call_soon_threadsafe(self._receive, kind, value)
        except RuntimeError:
            pass  # the server is shutting down

    def _receive(self, kind: str, value: Any) -> None:
        if self.state == "closed":
            if kind == "tool_call":
                value.future.set_result(CANCELLED_RESULT)  # nobody will hand this call out
            return
        self._events.put_nowait((kind, value))

    async def events(self) -> AsyncIterator[TextDelta | ToolCalls | Finished]:
        """The current round: text deltas, then ToolCalls or Finished. Errors are raised."""
        while True:
            kind, value = await self._next()
            if kind == "snapshot":
                delta = self._delta(value)
                if delta:
                    yield TextDelta(delta)
            elif kind == "tool_call":
                calls = [value] + await self._siblings()
                self.state = "suspended"
                self.touched = time.monotonic()
                for call in calls:
                    self.pending[call.id] = call
                    self.bridge.calls[call.id] = self
                output = json.dumps([{"name": c.name, "arguments": c.arguments} for c in calls])
                yield ToolCalls(calls, await self._usage(output))
                return
            elif kind == "done":
                self.state = "idle"
                self.touched = time.monotonic()
                self.history_key = fingerprint(self.key, self.turns + [Turn("assistant", self._text)])
                usage = await self._usage(self._text)
                capped = self.max_tokens and usage and usage["completion_tokens"] >= self.max_tokens
                yield Finished("length" if capped else "stop", usage)
                return
            elif kind == "cancelled":
                self.close()
                return
            else:
                error = await self._explain(value)
                self.close()
                reason = _finish_reason(error)
                if reason == "content_filter" or (reason == "length" and self._text):
                    yield Finished(reason, None)
                    return
                raise error

    async def _next(self) -> tuple[str, Any]:
        return self._held.pop(0) if self._held else await self._events.get()

    async def _siblings(self) -> list[PendingCall]:
        calls = []
        while True:
            try:
                kind, value = await asyncio.wait_for(self._events.get(), SIBLING_WINDOW)
            except TimeoutError:
                return calls
            if kind != "tool_call":
                self._held.append((kind, value))
                return calls
            calls.append(value)

    def _delta(self, snapshot: str) -> str:
        # Snapshots hold the response so far. One that does not extend the previous snapshot
        # starts a new response, as after tool results.
        delta = snapshot[len(self._snapshot):] if snapshot.startswith(self._snapshot) else snapshot
        self._snapshot = snapshot
        self._text += delta
        return delta

    async def _explain(self, error: Exception) -> Exception:
        # The SDK reports a context overflow as a bare GenerationError (status 255, no message),
        # so recognize one by size; counting only on failure keeps the normal path fast.
        if type(error) is not fm.GenerationError:
            return error
        model = self.bridge.model
        try:
            used = await model.token_count(self.session.transcript)
            if self._prompt:
                used += await model.token_count(self._prompt)
        except fm.FoundationModelsError:
            return error
        if used >= model.context_size:
            return fm.ExceededContextWindowSizeError(f"{used} tokens for a {model.context_size}-token window")
        return error

    async def _usage(self, output: str) -> dict | None:
        model = self.bridge.model
        try:
            total = await model.token_count(self.session.transcript)
            completion = min(await model.token_count(output), total) if output else 0
        except fm.FoundationModelsError as error:
            log.debug("token count failed: %s", error)
            return None
        return {"prompt_tokens": total - completion, "completion_tokens": completion, "total_tokens": total}

    def resume(self, results: dict[str, str], turns: list[Turn]) -> None:
        self.turns = turns
        self._prompt = "\n".join(results.values())  # the new input, for _explain()
        self.state = "generating"
        self._text = ""
        for call_id, call in self.pending.items():
            call.future.set_result(results.get(call_id, MISSING_RESULT))
            self.bridge.calls.pop(call_id, None)
        self.pending.clear()

    def close(self) -> None:
        if self.state == "closed":
            return
        self.state = "closed"
        queued = [value for kind, value in self._held + _drain(self._events) if kind == "tool_call"]
        for call in [*self.pending.values(), *queued]:
            if not call.future.done():
                call.future.set_result(CANCELLED_RESULT)  # lets the parked generation finish
            self.bridge.calls.pop(call.id, None)
        self.pending.clear()
        self._held.clear()
        if self._generation:
            loop, task = self._generation
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # the generation already ended
        self._remove_files()

    def _remove_files(self) -> None:
        if self._files is not None:
            shutil.rmtree(self._files, ignore_errors=True)
            self._files = None


def _drain(events: asyncio.Queue) -> list[tuple[str, Any]]:
    drained = []
    while not events.empty():
        drained.append(events.get_nowait())
    return drained


def _finish_reason(error: Exception) -> str | None:
    if isinstance(error, (fm.GuardrailViolationError, fm.RefusalError)):
        return "content_filter"
    if isinstance(error, fm.ExceededContextWindowSizeError):
        return "length"
    return None


class Bridge:
    """Routes each request to the conversation it continues, or starts one."""

    def __init__(self, *, tool_timeout: float = 600, idle_ttl: float = 1800, max_idle: int = 8):
        self.model = fm.SystemLanguageModel()
        self.tool_timeout = tool_timeout
        self.idle_ttl = idle_ttl
        self.max_idle = max_idle
        self.conversations: list[Conversation] = []
        self.calls: dict[str, Conversation] = {}  # pending tool-call id → conversation
        self._lock = asyncio.Lock()

    async def begin(
        self,
        instructions: str | None,
        turns: list[Turn],
        tools: list[dict],
        options: fm.GenerationOptions | None,
    ) -> tuple[Conversation, str]:
        """Set the next round going. Returns the conversation and how it was found."""
        async with self._lock:
            results = trailing_tool_results(turns)
            owner = next((self.calls[i] for i in results if i in self.calls), None)
            if owner is not None and owner.state == "suspended" and not owner.busy:
                owner.busy = True
                owner.resume(results, turns)
                return owner, "resume"

            key = fingerprint(instructions, tools)
            history, prompt_turn = split_prompt(turns)
            conversation, route = None, "rebuild" if history else "new"
            if prompt_turn is not None and history:
                wanted = fingerprint(key, history)
                conversation = next(
                    (c for c in self.conversations if c.state == "idle" and not c.busy and c.history_key == wanted),
                    None,
                )
                route = "reuse" if conversation else route
            if conversation is None:
                conversation = Conversation(self, key, tools)
                await conversation.open(instructions, history)
                self.conversations.append(conversation)
            conversation.busy = True
            conversation.start(conversation.prompt(prompt_turn), options, turns)
            return conversation, route

    def release(self, conversation: Conversation) -> None:
        conversation.busy = False
        self.sweep()

    def sweep(self) -> None:
        """Drop conversations whose client went away, and keep the idle cache bounded."""
        now = time.monotonic()
        for c in self.conversations:
            if c.busy:
                continue
            if c.state == "suspended" and now - c.touched > self.tool_timeout:
                log.info("conversation %s: no tool result after %ds, cancelled", c.id, self.tool_timeout)
                c.close()
            elif c.state == "idle" and now - c.touched > self.idle_ttl:
                c.close()
        idle = sorted((c for c in self.conversations if c.state == "idle" and not c.busy), key=lambda c: c.touched)
        for c in idle[: max(0, len(idle) - self.max_idle)]:
            c.close()
        self.conversations = [c for c in self.conversations if c.state != "closed"]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.conversations:
            counts[c.state] = counts.get(c.state, 0) + 1
        return counts
