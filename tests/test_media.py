"""Unit tests for the media module.

Covers the pure-logic bits (argument construction, encoder line parsing) that
don't require running ffmpeg. Anything that does require ffmpeg is tested
behind a ``ffmpeg`` skip mark so the suite still passes on minimal CI.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from max_comfy import media
from max_comfy.media import MediaError


# ---------------------------------------------------------------------- #
# Encoder detection
# ---------------------------------------------------------------------- #
def test_detect_encoder_returns_first_match():
    media.detect_encoder.cache_clear()
    fake_output = (
        "Encoders:\n"
        " V..... = Video\n"
        " ------\n"
        " V....D libx264              libx264 H.264 / AVC\n"
        " V..... mpeg4                MPEG-4 part 2\n"
    )
    with patch.object(
        subprocess, "run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=fake_output, stderr=""),
    ), patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
        result = media.detect_encoder(prefer=("libx264", "mpeg4"))
    assert result == "libx264"
    media.detect_encoder.cache_clear()


def test_detect_encoder_falls_back_through_list():
    media.detect_encoder.cache_clear()
    fake_output = (
        "Encoders:\n"
        " ------\n"
        " V..... mpeg4                MPEG-4 part 2\n"
    )
    with patch.object(
        subprocess, "run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=fake_output, stderr=""),
    ), patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
        result = media.detect_encoder(prefer=("libx264", "libsvtav1", "mpeg4"))
    assert result == "mpeg4"
    media.detect_encoder.cache_clear()


def test_detect_encoder_raises_when_none_found():
    media.detect_encoder.cache_clear()
    fake_output = "Encoders:\n ------\n V..... rawvideo  raw\n"
    with patch.object(
        subprocess, "run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=fake_output, stderr=""),
    ), patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"), pytest.raises(
        MediaError, match="None of the preferred encoders"
    ):
        media.detect_encoder(prefer=("libx264", "libsvtav1"))
    media.detect_encoder.cache_clear()


def test_require_ffmpeg_raises_when_missing():
    with patch.object(shutil, "which", return_value=None), pytest.raises(
        MediaError, match="ffmpeg not found"
    ):
        media.require_ffmpeg()


# ---------------------------------------------------------------------- #
# Stitch arg construction
# ---------------------------------------------------------------------- #
def test_build_stitch_args_simple_concat():
    args = media.build_stitch_args(
        inputs=[Path("a.mp4"), Path("b.mp4"), Path("c.mp4")],
        output_path=Path("out.mp4"),
        drop_first_frame=False,
        crossfade_frames=0,
        fps=24,
        encoder="libx264",
        crf=18,
    )
    # Three inputs and a concat filter
    joined = " ".join(args)
    assert args[0] == "ffmpeg"
    assert args.count("-i") == 3
    assert "concat=n=3:v=1:a=0[outv]" in joined
    assert "[v0][v1][v2]" in joined
    assert "-c:v" in args and "libx264" in args
    assert "-r" in args and "24" in args
    assert args[-1] == "out.mp4"


def test_build_stitch_args_drop_first_frame():
    args = media.build_stitch_args(
        inputs=[Path("a.mp4"), Path("b.mp4")],
        output_path=Path("out.mp4"),
        drop_first_frame=True,
        crossfade_frames=0,
        fps=None,
        encoder="libx264",
        crf=18,
    )
    joined = " ".join(args)
    # First input keeps frame 0, subsequent inputs drop it.
    assert "[0:v]setpts=N/FRAME_RATE/TB[v0]" in joined
    assert "[1:v]select='gte(n,1)'" in joined
    assert "concat=n=2:v=1:a=0[outv]" in joined


def test_build_stitch_args_crossfade_requires_fps():
    with pytest.raises(MediaError, match="requires fps"):
        media.build_stitch_args(
            inputs=[Path("a.mp4"), Path("b.mp4")],
            output_path=Path("out.mp4"),
            drop_first_frame=False,
            crossfade_frames=8,
            fps=None,
            encoder="libx264",
            crf=18,
        )


def test_build_stitch_args_crossfade_chains_xfades():
    # Three inputs with crossfade should produce two xfade filter steps.
    info = media.VideoInfo(
        path=Path("x.mp4"), width=1024, height=576, fps=24, duration_s=2.0,
        frame_count=48, codec="h264", pixel_format="yuv420p",
    )
    with patch.object(media, "get_video_info", return_value=info):
        args = media.build_stitch_args(
            inputs=[Path("a.mp4"), Path("b.mp4"), Path("c.mp4")],
            output_path=Path("out.mp4"),
            drop_first_frame=False,
            crossfade_frames=8,
            fps=24,
            encoder="libx264",
            crf=18,
        )
    joined = " ".join(args)
    assert joined.count("xfade=") == 2  # two transitions for three inputs
    assert "[outv]" in joined


# ---------------------------------------------------------------------- #
# Filter helpers
# ---------------------------------------------------------------------- #
def test_interpolate_unknown_method_raises():
    with pytest.raises(MediaError, match="Unknown interpolation method"):
        media.interpolate(Path("in.mp4"), Path("out.mp4"), target_fps=60, method="nope")


# ---------------------------------------------------------------------- #
# Real-ffmpeg integration (skipped if ffmpeg unavailable)
# ---------------------------------------------------------------------- #
ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available on PATH",
)


@ffmpeg_required
def test_get_video_info_on_generated_clip(tmp_path: Path):
    # Generate a tiny 1-second test video using ffmpeg's test source.
    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=1:size=64x64:rate=10",
            "-c:v", "mpeg4", str(src),
        ],
        check=True, capture_output=True,
    )
    info = media.get_video_info(src)
    assert info.width == 64
    assert info.height == 64
    assert 9 <= info.fps <= 11
    assert 0.9 <= info.duration_s <= 1.2


@ffmpeg_required
def test_extract_last_frame_writes_a_png(tmp_path: Path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=1:size=64x64:rate=10",
            "-c:v", "mpeg4", str(src),
        ],
        check=True, capture_output=True,
    )
    out = tmp_path / "last.png"
    media.extract_last_frame(src, out)
    assert out.is_file()
    assert out.stat().st_size > 0
