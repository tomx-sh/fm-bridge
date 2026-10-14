"""OpenAI chat messages as normalized turns, conversation fingerprints and transcript entries."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any


class RequestError(Exception):
    """A request the bridge cannot serve. Reported to the client as a 4xx error."""

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_request", param: str | None = None):
        super().__init__(message)
        self.status, self.code, self.param = status, code, param


@dataclass
class Image:
    mime: str
    data: bytes


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text


@dataclass
class Turn:
    role: str  # "user", "assistant" or "tool"
    text: str = ""
    images: list[Image] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


_DATA_URL = re.compile(r"^data:(?P<mime>image/[\w.+-]+);base64,(?P<data>.*)$", re.S)


def parse_messages(messages: Any) -> tuple[str | None, list[Turn]]:
    """Split OpenAI messages into instructions (all system text) and conversation turns."""
    if not isinstance(messages, list) or not messages:
        raise RequestError("`messages` must be a non-empty array", param="messages")
    instructions: list[str] = []
    turns: list[Turn] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestError(f"messages[{index}] must be an object", param="messages")
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            instructions.append(_text(content, index))
        elif role == "user":
            text, images = _user_content(content, index)
            turns.append(Turn("user", text, images))
        elif role == "assistant":
            calls = []
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                calls.append(ToolCall(call.get("id", ""), function.get("name", ""), function.get("arguments") or "{}"))
            turns.append(Turn("assistant", _text(content, index), tool_calls=calls))
        elif role == "tool":
            turns.append(Turn("tool", _text(content, index), tool_call_id=message.get("tool_call_id")))
        else:
            raise RequestError(f"messages[{index}] has unsupported role {role!r}", param="messages")
    if not turns:
        raise RequestError("`messages` must contain at least one user message", param="messages")
    return "\n\n".join(i for i in instructions if i) or None, turns


def _text(content: Any, index: int) -> str:
    text, images = _user_content(content, index)
    if images:
        raise RequestError(f"messages[{index}]: images are only supported in user messages", param="messages")
    return text


def _user_content(content: Any, index: int) -> tuple[str, list[Image]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise RequestError(f"messages[{index}].content must be a string or an array", param="messages")
    texts, images = [], []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("text", "input_text"):
            texts.append(part.get("text", ""))
        elif kind == "image_url":
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            images.append(_decode_image(url, index))
        else:
            raise RequestError(f"messages[{index}] has unsupported content part {kind!r}", param="messages")
    return "\n".join(texts), images


def _decode_image(url: Any, index: int) -> Image:
    match = _DATA_URL.match(url or "")
    if not match:
        raise RequestError(f"messages[{index}]: images must be base64 data: URLs", param="messages")
    try:
        return Image(match["mime"], base64.b64decode(match["data"], validate=True))
    except binascii.Error as e:
        raise RequestError(f"messages[{index}]: invalid base64 image data", param="messages") from e


def split_prompt(turns: list[Turn]) -> tuple[list[Turn], Turn | None]:
    """The history before the prompt, and the prompt (None when the model continues after tool results)."""
    if turns[-1].role == "user":
        return turns[:-1], turns[-1]
    return turns, None


def trailing_tool_results(turns: list[Turn]) -> dict[str, str]:
    """Tool results that end the conversation, keyed by tool-call id."""
    results: dict[str, str] = {}
    for turn in reversed(turns):
        if turn.role != "tool":
            break
        results[turn.tool_call_id or ""] = turn.text
    return results


def fingerprint(*parts: Any) -> str:
    """A stable hash of instructions, tools and turns, used to recognize a known conversation."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=_canonical).encode()).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Turn):
        return [
            value.role,
            value.text.strip(),
            [hashlib.sha256(i.data).hexdigest() for i in value.images],
            [[c.id, c.name, _canonical_json(c.arguments)] for c in value.tool_calls],
            value.tool_call_id,
        ]
    raise TypeError(f"cannot fingerprint {type(value).__name__}")


def _canonical_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), sort_keys=True)
    except ValueError:
        return text


def transcript_entries(turns: list[Turn]) -> list[dict]:
    """Foundation Models transcript entries for conversation turns (instructions excluded)."""
    entries: list[dict] = []
    tool_names: dict[str, str] = {}
    for turn in turns:
        if turn.role == "user":
            # Earlier images cannot be replayed into a transcript; keep a marker in their place.
            text = "\n".join([turn.text] + ["[image]"] * len(turn.images)).strip()
            entries.append({"id": _uid(), "role": "user", "options": {}, "contents": _contents(text)})
        elif turn.role == "assistant":
            # A response entry holds text or tool calls; the decoder drops text set beside tool calls.
            if turn.text or not turn.tool_calls:
                entries.append({"id": _uid(), "role": "response", "contents": _contents(turn.text)})
            if turn.tool_calls:
                calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls]
                entries.append({"id": _uid(), "role": "response", "toolCalls": calls})
                tool_names.update((c.id, c.name) for c in turn.tool_calls)
        else:
            call_id = turn.tool_call_id or _uid()
            entries.append({
                "id": call_id,
                "role": "tool",
                "toolCallID": call_id,
                "toolName": tool_names.get(call_id, "tool"),
                "contents": _contents(turn.text),
            })
    return entries


def _contents(text: str) -> list[dict]:
    return [{"id": _uid(), "type": "text", "text": text}]


def _uid() -> str:
    return str(uuid.uuid4()).upper()
