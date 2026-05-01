"""Unit tests for the Config loader."""

from __future__ import annotations

from pathlib import Path

from max_comfy.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.comfyui_host == "127.0.0.1"
    assert cfg.comfyui_port == 8188
    assert cfg.output_dir == Path("./output")
    assert cfg.poll_interval == 1.0
    assert cfg.defaults == {}


def test_load_explicit(tmp_path: Path):
    config_file = tmp_path / "cfg.toml"
    config_file.write_text(
        """
        comfyui_host = "10.0.0.1"
        comfyui_port = 9000
        output_dir = "/tmp/out"

        [defaults]
        steps = 40
        prompt = "hello"
        """
    )
    cfg = Config.load(explicit=config_file)
    assert cfg.comfyui_host == "10.0.0.1"
    assert cfg.comfyui_port == 9000
    assert cfg.output_dir == Path("/tmp/out")
    assert cfg.defaults == {"steps": 40, "prompt": "hello"}


def test_merged_with_overrides():
    cfg = Config()
    merged = cfg.merged_with({"comfyui_port": 9000, "defaults": {"steps": 50}})
    assert merged.comfyui_port == 9000
    assert merged.defaults == {"steps": 50}
    # original unchanged
    assert cfg.comfyui_port == 8188


def test_merged_defaults_merge_keys():
    cfg = Config(defaults={"a": 1, "b": 2})
    merged = cfg.merged_with({"defaults": {"b": 20, "c": 3}})
    assert merged.defaults == {"a": 1, "b": 20, "c": 3}


def test_to_dict_roundtrip():
    cfg = Config(comfyui_port=9001, output_dir=Path("/tmp/x"), defaults={"k": "v"})
    data = cfg.to_dict()
    cfg2 = Config.from_dict(data)
    assert cfg2.comfyui_port == 9001
    assert cfg2.output_dir == Path("/tmp/x")
    assert cfg2.defaults == {"k": "v"}


def test_resolved_workflow_dirs_dedupes(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MAX_COMFY_WORKFLOWS", raising=False)
    cfg = Config(workflow_dirs=[tmp_path, tmp_path])
    dirs = cfg.resolved_workflow_dirs()
    # The duplicate should appear only once.
    assert dirs.count(tmp_path.resolve()) == 1
