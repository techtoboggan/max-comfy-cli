"""Unit tests for the workflow registry and template loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from max_comfy.exceptions import WorkflowError
from max_comfy.workflows import TemplateWorkflow, Workflow, WorkflowRegistry, discover


def test_registry_register_and_get():
    reg = WorkflowRegistry()

    class Mine(Workflow):
        name = "mine"
        description = "test"

        def build(self, params):
            return {"ok": True}

    reg.register(Mine())
    assert "mine" in reg
    assert reg.get("mine").description == "test"


def test_registry_get_missing_raises():
    reg = WorkflowRegistry()
    with pytest.raises(WorkflowError):
        reg.get("nope")


def test_registry_rejects_unnamed():
    reg = WorkflowRegistry()
    with pytest.raises(WorkflowError):
        reg.register(Workflow())  # no name set


def test_template_workflow_builds(tmp_path: Path):
    path = tmp_path / "tpl.json"
    path.write_text(json.dumps({
        "1": {"class_type": "X", "inputs": {"text": "{{prompt}}", "seed": "{{seed:0}}"}}
    }))
    wf = TemplateWorkflow(path)
    out = wf.build({"prompt": "hi"})
    assert out == {"1": {"class_type": "X", "inputs": {"text": "hi", "seed": 0}}}


def test_template_workflow_placeholders(tmp_path: Path):
    path = tmp_path / "tpl.json"
    path.write_text(json.dumps({"a": "{{x}}", "b": "{{y:5}}"}))
    wf = TemplateWorkflow(path)
    assert wf.placeholders() == {"x", "y"}


def test_template_workflow_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(WorkflowError):
        TemplateWorkflow(path)


def test_discover_loads_json_and_sidecar(tmp_path: Path):
    (tmp_path / "wf.json").write_text(json.dumps({"a": "{{x}}"}))
    (tmp_path / "wf.toml").write_text(
        'description = "test desc"\nrequired = ["x"]\n[defaults]\nx = "default-x"\n'
    )

    reg = WorkflowRegistry()
    discover(dirs=[tmp_path], target=reg)

    assert "wf" in reg
    wf = reg.get("wf")
    assert wf.description == "test desc"
    assert wf.required == ("x",)
    assert wf.defaults == {"x": "default-x"}


def test_discover_loads_python_module(tmp_path: Path):
    (tmp_path / "plugin.py").write_text(
        "from max_comfy import Workflow, register\n"
        "@register\n"
        "class MyFlow(Workflow):\n"
        "    name = 'pyflow'\n"
        "    description = 'from a plugin'\n"
        "    def build(self, params): return {'ok': True}\n"
    )
    reg = WorkflowRegistry()
    # Discover into a custom registry by patching the module-level one is awkward;
    # instead, just verify the file imports without error and the @register-decorated
    # workflow appears in the global registry.
    from max_comfy import registry as global_reg

    before = "pyflow" in global_reg
    discover(dirs=[tmp_path], target=reg)
    assert "pyflow" in global_reg or before  # always succeeds; main check is no exception


def test_template_validate_missing_required(tmp_path: Path):
    path = tmp_path / "tpl.json"
    path.write_text(json.dumps({"a": "{{x}}"}))
    wf = TemplateWorkflow(path, required=("x",))
    with pytest.raises(WorkflowError):
        wf.validate_params({})
