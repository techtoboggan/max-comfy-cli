"""Template engine for ComfyUI workflow JSON.

Resolves `{{var}}` placeholders inside a workflow graph. Supports:

* Whole-string placeholders preserve the original parameter type, so
  `"seed": "{{seed}}"` becomes `"seed": 42` (int), not `"42"` (str).
* Mixed strings interpolate as text: `"prefix_{{name}}.png"` -> `"prefix_foo.png"`.
* Dotted paths into nested dicts: `{{job.prompt}}`.
* Optional defaults: `{{steps:30}}` returns 30 if `steps` is unset.
* Recursion through dicts and lists.
"""

from __future__ import annotations

import re
from typing import Any

from .exceptions import MissingParamError

# {{name}}, {{name:default}}, {{a.b.c}}, {{a.b:fallback}}
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*(?::([^}]*))?\s*\}\}")


def render(template: Any, params: dict[str, Any]) -> Any:
    """Recursively render a template against a params dict.

    Returns a new structure with placeholders replaced. The input is not mutated.
    Raises MissingParamError when a placeholder has no value and no default.
    """
    if isinstance(template, dict):
        return {k: render(v, params) for k, v in template.items()}
    if isinstance(template, list):
        return [render(item, params) for item in template]
    if isinstance(template, str):
        return _render_string(template, params)
    return template


def _render_string(text: str, params: dict[str, Any]) -> Any:
    matches = list(_PLACEHOLDER_RE.finditer(text))
    if not matches:
        return text

    # Whole-string placeholder: preserve the original Python type.
    if len(matches) == 1 and matches[0].group(0) == text.strip():
        name, default = matches[0].group(1), matches[0].group(2)
        return _resolve(name, default, params)

    # Mixed string: interpolate everything as strings.
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = _resolve(name, default, params)
        return str(value)

    return _PLACEHOLDER_RE.sub(replace, text)


def _resolve(name: str, default: str | None, params: dict[str, Any]) -> Any:
    parts = name.split(".")
    current: Any = params
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            current = _MISSING
            break

    if current is _MISSING:
        if default is not None:
            return _coerce_default(default)
        raise MissingParamError(name, available=list(params.keys()))
    return current


def _coerce_default(text: str) -> Any:
    """Try to parse a default value as int, float, bool, or str."""
    text = text.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    # Strip optional surrounding quotes.
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


def find_placeholders(template: Any) -> set[str]:
    """Walk a template and return the set of placeholder names it references."""
    found: set[str] = set()
    _collect(template, found)
    return found


def _collect(node: Any, found: set[str]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _collect(v, found)
    elif isinstance(node, list):
        for v in node:
            _collect(v, found)
    elif isinstance(node, str):
        for match in _PLACEHOLDER_RE.finditer(node):
            found.add(match.group(1))


class _Missing:
    """Sentinel for missing values."""

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _Missing()
