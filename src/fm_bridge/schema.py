"""Translate OpenAI tool parameters (JSON Schema) into Foundation Models generation schemas.

Foundation Models accepts a subset of JSON Schema: objects with typed properties holding
strings, integers, numbers, booleans, arrays of strings/integers/numbers/objects, and nested
objects, plus a few guides (enum, numeric bounds, item counts). Anything outside that subset
is mapped to the nearest supported shape, usually a string that holds JSON, and
`coerce_arguments` turns the generated arguments back into what the original schema expects.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

import apple_fm_sdk as fm
from apple_fm_sdk.generation_property import Property

_PRIMITIVES = {"string": str, "integer": int, "number": float}
_JSON_OBJECT = "Write the value as a JSON object."
_JSON_ARRAY = "Write the value as a JSON array."


def tool_schema(tool_name: str, parameters: dict | None) -> fm.GenerationSchema:
    """Build the arguments schema for one tool."""
    return _Builder(tool_name, parameters or {}).build()


def coerce_arguments(arguments: Any, parameters: dict | None) -> Any:
    """Convert generated arguments back to the shapes the original JSON Schema declares."""
    parameters = parameters or {}
    return _coerce(arguments, parameters, _definitions(parameters))


def _definitions(root: dict) -> dict:
    return {**root.get("definitions", {}), **root.get("$defs", {})}


def _normalize(node: Any, defs: dict, seen: frozenset = frozenset()) -> tuple[dict, bool]:
    """Resolve references and unions to one concrete schema. Returns (schema, nullable)."""
    if not isinstance(node, dict):
        return {}, False
    nullable = False
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        name = ref.rsplit("/", 1)[-1]
        if name in seen or name not in defs:
            return {"type": "string"}, False  # recursive or unknown: no finite form
        resolved, nullable = _normalize(defs[name], defs, seen | {name})
        extra = {k: v for k, v in node.items() if k != "$ref"}
        return {**resolved, **extra}, nullable
    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches:
            options = [_normalize(b, defs, seen)[0] for b in branches]
            concrete = [o for o in options if o.get("type") != "null"]
            nullable = len(concrete) < len(options)
            if all(isinstance(o.get("const", o.get("enum")), (str, list)) for o in concrete):
                values = [v for o in concrete for v in (o["enum"] if "enum" in o else [o["const"]])]
                if values and all(isinstance(v, str) for v in values):
                    return {**_rest(node, key), "type": "string", "enum": values}, nullable
            base = concrete[0] if concrete else {"type": "string"}
            return {**base, **_rest(node, key)}, nullable
    all_of = node.get("allOf")
    if isinstance(all_of, list) and all_of:
        merged: dict = {"type": "object", "properties": {}, "required": []}
        for part in all_of:
            part, _ = _normalize(part, defs, seen)
            merged["properties"].update(part.get("properties", {}))
            merged["required"] += part.get("required", [])
            if part.get("type") not in (None, "object"):
                merged = part
                break
        return {**merged, **_rest(node, "allOf")}, False
    types = node.get("type")
    if isinstance(types, list):
        concrete = [t for t in types if t != "null"]
        nullable = len(concrete) < len(types)
        node = {**node, "type": concrete[0] if concrete else "string"}
    if "type" not in node:
        if "properties" in node:
            node = {**node, "type": "object"}
        elif "items" in node:
            node = {**node, "type": "array"}
        elif "enum" in node or "const" in node:
            values = node.get("enum", [node.get("const")])
            node = {**node, "type": "string" if all(isinstance(v, str) for v in values) else "number"}
    return node, nullable


def _rest(node: dict, key: str) -> dict:
    return {k: v for k, v in node.items() if k != key}


def _identifier(text: str) -> str:
    name = re.sub(r"\W", "_", text) or "Tool"
    return name if name[0].isalpha() else f"T{name}"


def _describe(*parts: str | None) -> str | None:
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return text or None


class _Builder:
    def __init__(self, tool_name: str, root: dict):
        self.root = root
        self.defs = _definitions(root)
        self.prefix = _identifier(tool_name)
        self.nested: list[fm.GenerationSchema] = []
        self.names: set[str] = set()

    def build(self) -> fm.GenerationSchema:
        node, _ = _normalize(self.root, self.defs)
        properties = self._properties(node, self.prefix, frozenset())
        cls = type(self._unique(f"{self.prefix}_Arguments"), (), {})
        # Every nested schema hangs off the root: the Swift side builds only the root's references.
        return fm.GenerationSchema(cls, node.get("description"), properties, self.nested)

    def _unique(self, name: str) -> str:
        candidate, n = name, 2
        while candidate in self.names:
            candidate, n = f"{name}{n}", n + 1
        self.names.add(candidate)
        return candidate

    def _properties(self, node: dict, path: str, active: frozenset) -> list[Property]:
        required = set(node.get("required") or [])
        result = []
        for name, sub in (node.get("properties") or {}).items():
            typ, guides, note = self._field(sub, f"{path}_{_identifier(name)}", active)
            if name not in required:
                typ = Optional[typ]
            description = _describe(sub.get("description") if isinstance(sub, dict) else None, note)
            result.append(Property(name, typ, description, guides))
        return result

    def _object(self, node: dict, path: str, active: frozenset) -> type:
        cls = type(self._unique(path), (), {})
        properties = self._properties(node, path, active)
        self.nested.append(fm.GenerationSchema(cls, node.get("description"), properties, []))
        return cls

    def _field(self, sub: Any, path: str, active: frozenset) -> tuple[Any, list, str | None]:
        # A definition that contains itself has no finite form: send that level as JSON text.
        ref = sub.get("$ref") if isinstance(sub, dict) else None
        if isinstance(ref, str):
            if ref in active:
                return str, [], _JSON_OBJECT
            active = active | {ref}
        node, _ = _normalize(sub, self.defs)
        kind = node.get("type", "string")
        if kind == "string":
            values = node.get("enum") or ([node["const"]] if "const" in node else None)
            guides = [fm.GenerationGuide.anyOf(values)] if values and all(isinstance(v, str) for v in values) else []
            return str, guides, None
        if kind in ("integer", "number"):
            return _PRIMITIVES[kind], _bounds(node, kind), None
        if kind == "boolean":
            return bool, [], None
        if kind == "array":
            items, _ = _normalize(node.get("items", {}), self.defs)
            item_kind = items.get("type", "string")
            if item_kind in _PRIMITIVES and not (item_kind == "string" and items.get("enum")):
                element = _PRIMITIVES[item_kind]
            elif item_kind == "object" and items.get("properties"):
                item_ref = node.get("items", {}).get("$ref")
                if item_ref in active:
                    return str, [], _JSON_ARRAY
                element = self._object(items, f"{path}_Item", active | ({item_ref} if item_ref else set()))
            else:  # arrays of arrays, booleans, enums or free-form values
                return str, [], _JSON_ARRAY
            return List[element], _counts(node), None
        if kind == "object":
            if node.get("properties"):
                return self._object(node, path, active), [], None
            return str, [], _JSON_OBJECT
        return str, [], None


def _bounds(node: dict, kind: str) -> list:
    low, high = node.get("minimum"), node.get("maximum")
    step = 1 if kind == "integer" else 0
    if node.get("exclusiveMinimum") is not None and not isinstance(node["exclusiveMinimum"], bool):
        low = node["exclusiveMinimum"] + step
    if node.get("exclusiveMaximum") is not None and not isinstance(node["exclusiveMaximum"], bool):
        high = node["exclusiveMaximum"] - step
    if kind == "integer":
        low = None if low is None else int(low)
        high = None if high is None else int(high)
    if low is not None and high is not None:
        return [fm.GenerationGuide.range((low, high))]
    if low is not None:
        return [fm.GenerationGuide.minimum(low)]
    if high is not None:
        return [fm.GenerationGuide.maximum(high)]
    return []


def _counts(node: dict) -> list:
    guides = []
    if isinstance(node.get("minItems"), int):
        guides.append(fm.GenerationGuide.min_items(node["minItems"]))
    if isinstance(node.get("maxItems"), int):
        guides.append(fm.GenerationGuide.max_items(node["maxItems"]))
    return guides


def _coerce(value: Any, node: Any, defs: dict) -> Any:
    node, _ = _normalize(node, defs)
    kind = node.get("type")
    if value is None:
        return None
    if kind in ("object", "array") and isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if kind == "object" and isinstance(value, dict):
        properties = node.get("properties") or {}
        result = {}
        for key, item in value.items():
            schema = properties.get(key)
            if item is None and not _nullable(schema, defs):
                continue  # an omitted optional property comes back as null
            result[key] = _coerce(item, schema, defs) if schema is not None else item
        return result
    if kind == "array" and isinstance(value, list):
        return [_coerce(item, node.get("items", {}), defs) for item in value]
    if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if kind == "integer":
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
    if kind == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _nullable(schema: Any, defs: dict) -> bool:
    return schema is not None and _normalize(schema, defs)[1]
