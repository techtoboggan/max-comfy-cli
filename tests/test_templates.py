"""Unit tests for the template engine."""

from __future__ import annotations

import pytest

from max_comfy.exceptions import MissingParamError
from max_comfy.templates import find_placeholders, render


def test_simple_substitution_preserves_string():
    out = render("hello {{name}}", {"name": "world"})
    assert out == "hello world"


def test_whole_string_placeholder_preserves_int():
    out = render("{{seed}}", {"seed": 42})
    assert out == 42
    assert isinstance(out, int)


def test_whole_string_placeholder_preserves_float():
    out = render("{{cfg}}", {"cfg": 6.5})
    assert out == 6.5
    assert isinstance(out, float)


def test_whole_string_placeholder_preserves_list():
    out = render("{{loras}}", {"loras": [{"name": "a"}]})
    assert out == [{"name": "a"}]


def test_default_value_int():
    assert render("{{steps:30}}", {}) == 30


def test_default_value_float():
    assert render("{{cfg:6.5}}", {}) == 6.5


def test_default_value_string():
    assert render("{{name:default}}", {}) == "default"


def test_default_value_quoted_string():
    assert render('{{name:"default"}}', {}) == "default"


def test_default_value_bool():
    assert render("{{enabled:true}}", {}) is True
    assert render("{{enabled:false}}", {}) is False


def test_default_empty_string():
    assert render("{{negative:}}", {}) == ""


def test_param_overrides_default():
    assert render("{{steps:30}}", {"steps": 50}) == 50


def test_dotted_path():
    assert render("{{job.prompt}}", {"job": {"prompt": "hi"}}) == "hi"


def test_dotted_missing_uses_default():
    assert render("{{job.steps:25}}", {"job": {}}) == 25


def test_missing_required_raises():
    with pytest.raises(MissingParamError) as exc:
        render("{{required}}", {})
    assert "required" in str(exc.value)


def test_recurses_into_dict():
    template = {"a": "{{x}}", "b": {"c": "{{y}}"}}
    out = render(template, {"x": 1, "y": 2})
    assert out == {"a": 1, "b": {"c": 2}}


def test_recurses_into_list():
    template = ["a", "{{x}}", {"y": "{{y}}"}]
    out = render(template, {"x": 5, "y": "hello"})
    assert out == ["a", 5, {"y": "hello"}]


def test_does_not_mutate_input():
    template = {"a": "{{x}}"}
    render(template, {"x": 1})
    assert template == {"a": "{{x}}"}


def test_mixed_string_interpolates():
    out = render("file_{{name}}_{{i}}.png", {"name": "foo", "i": 3})
    assert out == "file_foo_3.png"


def test_find_placeholders_collects_all():
    template = {"a": "{{x}}", "b": ["{{y}}", "{{z:5}}", "literal"]}
    assert find_placeholders(template) == {"x", "y", "z"}


def test_find_placeholders_empty():
    assert find_placeholders({"a": "literal"}) == set()


def test_node_reference_lists_passthrough():
    """ComfyUI node refs are [node_id, slot] lists — they must pass through unchanged."""
    template = {
        "1": {"class_type": "Foo", "inputs": {"model": ["2", 0], "seed": "{{seed}}"}}
    }
    out = render(template, {"seed": 42})
    assert out == {
        "1": {"class_type": "Foo", "inputs": {"model": ["2", 0], "seed": 42}}
    }
