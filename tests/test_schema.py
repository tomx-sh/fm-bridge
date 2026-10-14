import json
from pathlib import Path

import apple_fm_sdk as fm
import pytest

from fm_bridge.schema import coerce_arguments, tool_schema

PI_TOOLS = json.loads((Path(__file__).parent / "pi_tools.json").read_text())

EDGE_CASES = {
    "type": "object",
    "$defs": {
        "Point": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]},
        "Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}},
    },
    "properties": {
        "mode": {"type": "string", "enum": ["fast", "slow"]},
        "maybe": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
        "flags": {"type": "array", "items": {"type": "boolean"}},
        "grid": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "meta": {"type": "object", "additionalProperties": True},
        "points": {"type": "array", "items": {"$ref": "#/$defs/Point"}, "minItems": 1},
        "tree": {"$ref": "#/$defs/Node"},
        "choice": {"anyOf": [{"const": "a"}, {"const": "b"}]},
    },
    "required": ["mode", "points"],
}


class _Tool(fm.Tool):
    def __init__(self, name: str, parameters: dict):
        self.name, self.description = name, "test"
        self._schema = tool_schema(name, parameters)
        super().__init__()  # the framework validates the schema here

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return self._schema

    async def call(self, args: fm.GeneratedContent) -> str:
        return ""


@pytest.mark.parametrize("spec", PI_TOOLS, ids=lambda t: t["function"]["name"])
def test_pi_tools_build(spec):
    function = spec["function"]
    schema = _Tool(function["name"], function["parameters"])._schema.to_dict()
    assert set(schema["properties"]) == set(function["parameters"]["properties"])
    assert set(schema["required"]) == set(function["parameters"].get("required", []))


def test_edit_tool_keeps_array_of_objects():
    edit = next(t["function"] for t in PI_TOOLS if t["function"]["name"] == "edit")
    schema = _Tool("edit", edit["parameters"])._schema.to_dict()
    items = schema["properties"]["edits"]["items"]["$ref"].rsplit("/", 1)[-1]
    assert set(schema["$defs"][items]["properties"]) == {"oldText", "newText"}


def test_edge_cases_map_to_supported_shapes():
    schema = _Tool("weird", EDGE_CASES)._schema.to_dict()
    props = schema["properties"]
    assert props["mode"]["enum"] == ["fast", "slow"]
    assert props["choice"]["enum"] == ["a", "b"]
    assert (props["maybe"]["minimum"], props["maybe"]["maximum"]) == (1, 5)
    assert "maybe" not in schema["required"]
    for lossy in ("flags", "grid", "meta"):
        assert props[lossy]["type"] == "string"
    tree = schema["$defs"][props["tree"]["$ref"].rsplit("/", 1)[-1]]
    assert tree["properties"]["child"]["type"] == "string"  # recursion stops at one level


def test_coerce_restores_lossy_values():
    generated = {
        "mode": "fast", "maybe": None, "flags": "[true, false]", "grid": "[[1, 2], [3]]",
        "meta": '{"a": 1}', "points": [{"x": 1.0, "y": 2.5}], "tree": {"child": '{"child": null}'},
    }
    assert coerce_arguments(generated, EDGE_CASES) == {
        "mode": "fast", "maybe": None, "flags": [True, False], "grid": [[1, 2], [3]],
        "meta": {"a": 1}, "points": [{"x": 1.0, "y": 2.5}], "tree": {"child": {}},
    }


def test_coerce_drops_null_optionals_and_fixes_integers():
    read = next(t["function"] for t in PI_TOOLS if t["function"]["name"] == "read")
    assert coerce_arguments({"path": "a.txt", "offset": None, "limit": 10.0}, read["parameters"]) == {"path": "a.txt", "limit": 10.0}
    integer = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert coerce_arguments({"n": 3.0}, integer) == {"n": 3}


def test_tool_without_parameters():
    assert _Tool("ping", {})._schema.to_dict()["properties"] == {}
