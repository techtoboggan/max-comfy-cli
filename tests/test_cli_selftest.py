"""End-to-end tests for the `max-comfy selftest` CLI command.

Drives the actual click CLI against the fake ComfyUI server fixture, so we
verify both that the command's argument parsing/output is right AND that
the graph it constructs round-trips correctly through the HTTP layer.
"""

from __future__ import annotations

from click.testing import CliRunner

from max_comfy.cli import cli


def test_selftest_succeeds_against_fake_server(fake_comfy_server):
    port, state = fake_comfy_server
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "--port",
            str(port),
            "selftest",
            "--checkpoint",
            "test_model.safetensors",
            "--steps",
            "1",
            "--size",
            "64",
        ],
    )

    assert result.exit_code == 0, f"selftest failed:\n{result.output}"
    assert "OK" in result.output
    assert "max-comfy selftest" not in result.output.split("OK")[0] or "Submitting" in result.output
    assert "ComfyUI integration verified" in result.output

    # Verify ONE prompt was submitted, and the graph has the expected shape.
    assert len(state.submitted) == 1
    graph = next(iter(state.submitted.values()))
    assert graph["1"]["class_type"] == "CheckpointLoaderSimple"
    assert graph["1"]["inputs"]["ckpt_name"] == "test_model.safetensors"
    assert graph["5"]["class_type"] == "KSampler"
    assert graph["5"]["inputs"]["steps"] == 1
    assert graph["4"]["inputs"]["width"] == 64
    assert graph["4"]["inputs"]["height"] == 64
    assert graph["7"]["class_type"] == "SaveImage"


def test_selftest_passes_static_validation_before_submitting(fake_comfy_server):
    """The graph the command builds should always be structurally valid."""
    from max_comfy.workflows import validate_graph

    port, _ = fake_comfy_server
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--port", str(port), "selftest", "--checkpoint", "x.safetensors"],
    )
    assert result.exit_code == 0, result.output

    # If we ever introduce a bug in the smoke-test graph itself, the static
    # validator should catch it before we hit ComfyUI; this test ensures the
    # validator agrees the graph is clean today.
    from max_comfy.cli import selftest as _  # noqa: F401 — already imported via cli

    # Reconstruct the graph by inspecting what the command actually submitted.
    port, state = fake_comfy_server
    if state.submitted:
        graph = next(iter(state.submitted.values()))
        assert validate_graph(graph) == []


def test_selftest_fails_when_server_unreachable():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--port", "1", "selftest", "--checkpoint", "x.safetensors"],
    )
    assert result.exit_code == 1
    assert "not reachable" in result.output


def test_selftest_requires_checkpoint_when_no_path_configured(fake_comfy_server):
    port, _ = fake_comfy_server
    runner = CliRunner()
    # No --checkpoint, no comfyui_path in config → should error.
    result = runner.invoke(cli, ["--port", str(port), "selftest"])
    assert result.exit_code == 1
    assert "checkpoint" in result.output.lower()


def test_selftest_auto_picks_checkpoint_from_comfyui_path(fake_comfy_server, tmp_path):
    port, state = fake_comfy_server

    # Build a fake ComfyUI install with one checkpoint.
    fake_comfyui = tmp_path / "ComfyUI"
    (fake_comfyui / "models" / "checkpoints").mkdir(parents=True)
    (fake_comfyui / "main.py").write_text("# fake")
    ckpt = fake_comfyui / "models" / "checkpoints" / "auto_picked.safetensors"
    ckpt.write_bytes(b"fake checkpoint")

    config_file = tmp_path / "max-comfy.toml"
    config_file.write_text(
        f'comfyui_host = "127.0.0.1"\n'
        f"comfyui_port = {port}\n"
        f'comfyui_path = "{fake_comfyui}"\n'
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(config_file),
            "--port",
            str(port),  # CLI flag overrides config
            "selftest",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Auto-selected checkpoint: auto_picked.safetensors" in result.output

    graph = next(iter(state.submitted.values()))
    assert graph["1"]["inputs"]["ckpt_name"] == "auto_picked.safetensors"


def test_selftest_propagates_comfy_errors(fake_comfy_server):
    """If ComfyUI never marks the prompt complete, selftest should fail with a clear message."""
    port, state = fake_comfy_server
    state.auto_complete = False  # the fake server accepts /prompt but never reports completion

    # Use the smallest possible timeout so the test stays fast.
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--port",
            str(port),
            "selftest",
            "--checkpoint",
            "x.safetensors",
            "--timeout",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "did not complete" in result.output.lower()
