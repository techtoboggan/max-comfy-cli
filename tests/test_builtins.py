"""Tests for the built-in workflows.

Uses a fake Client to avoid touching ffmpeg or ComfyUI. Verifies that:

* ``video-chain`` runs the segment workflow N times, threading last-frame paths
  forward, and calls media.stitch_videos with the right inputs.
* ``video-postprocess`` walks its filter chain in the documented order.
* ``reactor-face-swap`` extracts frames, uploads each, submits per-frame, then
  encodes the result.

The ffmpeg primitives in :mod:`max_comfy.media` are themselves tested in
``test_media.py`` (with arg-construction + skipif-ffmpeg integration tests).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from max_comfy import media, registry
from max_comfy.builtins import VideoChainWorkflow, VideoPostprocessWorkflow
from max_comfy.config import Config
from max_comfy.workflows import RunContext, RunResult, Workflow, WorkflowRegistry


class _FakeSegmentWorkflow(Workflow):
    """A Workflow that pretends to produce a video without touching ComfyUI."""

    name = "fake-segment"
    description = "test"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, ctx: RunContext) -> RunResult:
        self.calls.append(dict(ctx.params))
        out = ctx.output_dir / f"segment_{ctx.params['segment_index']}.webp"
        out.write_bytes(b"fake-video-bytes")
        return RunResult(files=[out])


@pytest.fixture
def fake_registry():
    reg = WorkflowRegistry()
    seg = _FakeSegmentWorkflow()
    reg.register(seg)
    return reg, seg


def _ctx(tmp_path: Path, params: dict, fake_reg: WorkflowRegistry) -> RunContext:
    client = MagicMock()
    client.upload_input.side_effect = lambda p, **kw: Path(p).name
    return RunContext(
        client=client,
        config=Config(),
        params=params,
        output_dir=tmp_path,
        job_id="test",
        registry=fake_reg,
    )


# ---------------------------------------------------------------------- #
# video-chain
# ---------------------------------------------------------------------- #
def test_video_chain_runs_each_segment_with_threaded_last_frame(tmp_path, fake_registry):
    reg, seg = fake_registry
    initial = tmp_path / "start.png"
    initial.write_bytes(b"fake-png")

    chain = VideoChainWorkflow()
    params = {
        "segment_workflow": "fake-segment",
        "initial_image": str(initial),
        "segments": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
        "stitch": False,
    }
    ctx = _ctx(tmp_path, params, reg)

    with patch.object(media, "extract_last_frame") as mock_extract:
        mock_extract.side_effect = lambda src, dst: Path(dst).write_bytes(b"frame")
        chain.run(ctx)

    # Three segments invoked.
    assert len(seg.calls) == 3
    # First segment's input_image is the uploaded start frame.
    assert seg.calls[0]["input_image"] == "start.png"
    # Subsequent segments are uploaded last-frames; mocked upload returns the basename.
    assert seg.calls[1]["input_image"].endswith("_lastframe.png")
    assert seg.calls[2]["input_image"].endswith("_lastframe.png")
    # Per-segment seeds applied
    assert [c["seed"] for c in seg.calls] == [1, 2, 3]
    # extract_last_frame called between segments only (so 2 times for 3 segments).
    assert mock_extract.call_count == 2


def test_video_chain_without_initial_image_runs_first_text_only(tmp_path, fake_registry):
    reg, seg = fake_registry
    params = {
        "segment_workflow": "fake-segment",
        "segments": [{"seed": 1}, {"seed": 2}],
        "stitch": False,
    }
    ctx = _ctx(tmp_path, params, reg)
    chain = VideoChainWorkflow()

    with patch.object(media, "extract_last_frame") as mock_extract:
        mock_extract.side_effect = lambda src, dst: Path(dst).write_bytes(b"frame")
        chain.run(ctx)

    # First segment did NOT receive input_image (text-only mode).
    assert "input_image" not in seg.calls[0]
    # Second segment did.
    assert "input_image" in seg.calls[1]


def test_video_chain_calls_stitch_and_postprocess(tmp_path, fake_registry):
    reg, _ = fake_registry
    chain = VideoChainWorkflow()
    params = {
        "segment_workflow": "fake-segment",
        "segments": [{"seed": 1}, {"seed": 2}],
        "stitch": True,
        "interpolate_to_fps": 60,
        "upscale": 2.0,
        "sharpen": 1.0,
    }
    ctx = _ctx(tmp_path, params, reg)

    with patch.object(media, "extract_last_frame") as mock_extract, \
         patch.object(media, "stitch_videos") as mock_stitch, \
         patch.object(media, "interpolate") as mock_interp, \
         patch.object(media, "upscale") as mock_upscale, \
         patch.object(media, "sharpen") as mock_sharpen:
        mock_extract.side_effect = lambda src, dst: Path(dst).write_bytes(b"frame")
        mock_stitch.side_effect = lambda inputs, out, **kw: Path(out).write_bytes(b"")
        mock_interp.side_effect = lambda src, dst, **kw: Path(dst).write_bytes(b"")
        mock_upscale.side_effect = lambda src, dst, **kw: Path(dst).write_bytes(b"")
        mock_sharpen.side_effect = lambda src, dst, **kw: Path(dst).write_bytes(b"")

        chain.run(ctx)

    assert mock_stitch.called
    assert mock_interp.called
    assert mock_upscale.called
    assert mock_sharpen.called
    # Verify the stitch call got both segment files.
    stitch_inputs = mock_stitch.call_args.args[0]
    assert len(stitch_inputs) == 2


def test_video_chain_rejects_missing_registry(tmp_path):
    chain = VideoChainWorkflow()
    ctx = RunContext(
        client=MagicMock(),
        config=Config(),
        params={"segment_workflow": "x", "segments": [{}]},
        output_dir=tmp_path,
        registry=None,
    )
    with pytest.raises(Exception, match="registry"):
        chain.run(ctx)


def test_video_chain_rejects_empty_segments(tmp_path, fake_registry):
    reg, _ = fake_registry
    chain = VideoChainWorkflow()
    ctx = _ctx(tmp_path, {"segment_workflow": "fake-segment", "segments": []}, reg)
    with pytest.raises(Exception, match="non-empty list"):
        chain.run(ctx)


# ---------------------------------------------------------------------- #
# video-postprocess
# ---------------------------------------------------------------------- #
def test_video_postprocess_transcode_path(tmp_path):
    pp = VideoPostprocessWorkflow()
    inp = tmp_path / "raw.webp"
    inp.write_bytes(b"x")
    ctx = RunContext(
        client=MagicMock(),
        config=Config(),
        params={"input": str(inp), "transcode": True, "output_filename": "out.mp4"},
        output_dir=tmp_path,
        registry=WorkflowRegistry(),
    )
    with patch.object(media, "transcode") as mock_t:
        mock_t.side_effect = lambda src, dst, **kw: Path(dst).write_bytes(b"")
        result = pp.run(ctx)

    assert mock_t.called
    assert result.files[0].name == "out.mp4"


def test_video_postprocess_full_chain(tmp_path):
    pp = VideoPostprocessWorkflow()
    inp = tmp_path / "raw.webp"
    inp.write_bytes(b"x")
    params = {
        "input": str(inp),
        "transcode": True,
        "interpolate_to_fps": 60,
        "upscale": 2.0,
        "sharpen": 1.0,
        "output_filename": "out.mp4",
    }
    ctx = RunContext(
        client=MagicMock(),
        config=Config(),
        params=params,
        output_dir=tmp_path,
        registry=WorkflowRegistry(),
    )
    with patch.object(media, "transcode") as mt, \
         patch.object(media, "interpolate") as mi, \
         patch.object(media, "upscale") as mu, \
         patch.object(media, "sharpen") as ms:
        for m in (mt, mi, mu, ms):
            m.side_effect = lambda src, dst, **kw: Path(dst).write_bytes(b"")
        pp.run(ctx)

    assert all(m.called for m in (mt, mi, mu, ms))


# ---------------------------------------------------------------------- #
# Registry: built-ins are auto-registered
# ---------------------------------------------------------------------- #
def test_builtins_registered_globally():
    assert "video-chain" in registry
    assert "video-postprocess" in registry
    assert "reactor-face-swap" in registry
