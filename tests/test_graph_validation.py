"""Tests for the static graph validator."""

from __future__ import annotations

from max_comfy.workflows import validate_graph


def test_valid_graph_returns_no_issues():
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
        "2": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": 42,
                "steps": 30,
                "cfg": 7.0,
            },
        },
    }
    assert validate_graph(graph) == []


def test_unknown_node_reference_is_caught():
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {
            "class_type": "KSampler",
            "inputs": {"model": ["999", 0]},  # node 999 doesn't exist
        },
    }
    issues = validate_graph(graph)
    assert len(issues) == 1
    assert "unknown node '999'" in issues[0]


def test_missing_class_type_is_caught():
    graph = {"1": {"inputs": {"seed": 0}}}
    issues = validate_graph(graph)
    assert any("missing 'class_type'" in i for i in issues)


def test_empty_class_type_is_caught():
    graph = {"1": {"class_type": "", "inputs": {}}}
    issues = validate_graph(graph)
    assert any("non-empty string" in i for i in issues)


def test_inputs_must_be_dict():
    graph = {"1": {"class_type": "X", "inputs": [1, 2, 3]}}
    issues = validate_graph(graph)
    assert any("must be an object" in i for i in issues)


def test_unresolved_placeholder_is_caught():
    """A graph that still has {{var}} means template rendering was skipped or incomplete."""
    graph = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "{{prompt}}", "clip": ["2", 1]},
        },
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}},
    }
    issues = validate_graph(graph)
    assert any("unresolved placeholder" in i for i in issues)


def test_negative_slot_is_caught():
    graph = {
        "1": {"class_type": "X", "inputs": {}},
        "2": {"class_type": "Y", "inputs": {"in": ["1", -1]}},
    }
    issues = validate_graph(graph)
    assert any("negative slot" in i for i in issues)


def test_nested_references_are_validated():
    """References can appear inside nested lists or dicts (rare but legal)."""
    graph = {
        "1": {"class_type": "X", "inputs": {"sub": {"deep": ["999", 0]}}},
    }
    issues = validate_graph(graph)
    assert any("unknown node '999'" in i for i in issues)


def test_non_dict_root_returns_single_issue():
    assert validate_graph([]) == ["Graph must be a dict, got list"]  # type: ignore[arg-type]
    assert validate_graph(None) == ["Graph must be a dict, got NoneType"]  # type: ignore[arg-type]


def test_empty_graph():
    assert validate_graph({}) == ["Graph is empty"]


def test_real_sdxl_template_renders_to_valid_graph(tmp_path):
    """Round-trip: load the example sdxl_txt2img template, render, validate."""
    import json
    from pathlib import Path

    from max_comfy.workflows import TemplateWorkflow

    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "examples" / "workflows" / "sdxl_txt2img.json"
    if not template.is_file():  # pragma: no cover
        return  # skip if running outside the repo

    wf = TemplateWorkflow(template, defaults={
        "negative_prompt": "blurry",
        "width": 1024,
        "height": 1024,
        "batch_size": 1,
        "steps": 30,
        "cfg": 6.5,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "seed": 42,
        "output_prefix": "test",
    })
    graph = wf.build({"prompt": "a sunset", "checkpoint": "x.safetensors", **wf.defaults})
    # Ensure rendered graph is valid JSON-serialisable too.
    json.dumps(graph)
    assert validate_graph(graph) == []


def test_real_sdxl_loras_python_workflow_produces_valid_graph():
    """The programmatic SDXL+LoRAs example builds correctly with no LoRAs."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    plugin = repo_root / "examples" / "workflows" / "sdxl_loras.py"
    if not plugin.is_file():  # pragma: no cover
        return

    # Import the example module directly (it auto-registers).
    sys.path.insert(0, str(plugin.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("sdxl_loras_example", plugin)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        wf = module.SDXLWithLoras()
        params = {**wf.defaults, "prompt": "a forest", "checkpoint": "x.safetensors"}
        graph = wf.build(params)
        assert validate_graph(graph) == []
        # Now with two LoRAs:
        params["loras"] = [
            {"name": "a.safetensors", "weight": 0.8},
            {"name": "b.safetensors", "weight": 0.5},
        ]
        graph2 = wf.build(params)
        assert validate_graph(graph2) == []
    finally:
        sys.path.pop(0)
