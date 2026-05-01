"""FFmpeg/ffprobe-based media primitives.

First-class video tooling for ComfyUI pipelines. Both ``ffmpeg`` and
``ffprobe`` must be available on PATH (any standard ffmpeg install satisfies
both). All functions are subprocess wrappers — there is no Python video
library dependency.

The functions below cover everything the original generation scripts needed:

* Frame extraction (one frame, last frame, all frames).
* Encoding from a directory of frames.
* Stitching with last-of-prev / first-of-next deduplication and optional crossfade.
* Frame interpolation (motion-compensated and blended modes).
* Spatial upscaling.
* Unsharp-mask sharpening.
* Generic filter chain — escape hatch for anything ffmpeg can do.
* Encoder auto-detection (libsvtav1 / libx264 / mpeg4).
* Probe a video for fps/dimensions/duration/codec.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .exceptions import MaxComfyError

log = logging.getLogger(__name__)


class MediaError(MaxComfyError):
    """An ffmpeg/ffprobe operation failed or a media tool is unavailable."""


# ---------------------------------------------------------------------- #
# Probe / detection
# ---------------------------------------------------------------------- #
@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int
    codec: str
    pixel_format: str


def require_ffmpeg() -> None:
    """Raise :class:`MediaError` if ffmpeg or ffprobe are not on PATH."""
    if shutil.which("ffmpeg") is None:
        raise MediaError("ffmpeg not found on PATH. Install ffmpeg to use media operations.")
    if shutil.which("ffprobe") is None:
        raise MediaError("ffprobe not found on PATH (it ships with ffmpeg).")


@lru_cache(maxsize=1)
def detect_encoder(prefer: tuple[str, ...] = ("libx264", "libsvtav1", "mpeg4")) -> str:
    """Return the first encoder in ``prefer`` that ffmpeg has compiled in."""
    require_ffmpeg()
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    encoder_names = set()
    for line in out.splitlines():
        # Lines look like: " V..... libx264              libx264 H.264 / AVC ..."
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            encoder_names.add(parts[1])
    for enc in prefer:
        if enc in encoder_names:
            return enc
    raise MediaError(
        f"None of the preferred encoders are available. Tried: {', '.join(prefer)}. "
        f"Install an ffmpeg build that includes at least one of them."
    )


def get_video_info(path: Path | str) -> VideoInfo:
    """Probe a video for its width, height, fps, duration, frame count, codec."""
    require_ffmpeg()
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"Video not found: {p}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt:format=duration",
        "-of",
        "json",
        str(p),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed for {p}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise MediaError(f"No video stream in {p}")
    stream = streams[0]
    fmt = data.get("format") or {}

    fps_str = stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/", 1)
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return VideoInfo(
        path=p,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=fps,
        duration_s=float(fmt.get("duration") or 0.0),
        frame_count=int(stream.get("nb_frames") or 0),
        codec=str(stream.get("codec_name") or ""),
        pixel_format=str(stream.get("pix_fmt") or ""),
    )


# ---------------------------------------------------------------------- #
# Frame I/O
# ---------------------------------------------------------------------- #
def extract_frames(
    video_path: Path | str,
    output_dir: Path | str,
    pattern: str = "frame_%05d.png",
    fps: float | None = None,
) -> list[Path]:
    """Extract frames as PNGs into ``output_dir``. Returns the sorted list."""
    require_ffmpeg()
    video = Path(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-i", str(video)]
    if fps:
        args += ["-vf", f"fps={fps}"]
    args += [str(out / pattern)]
    _run(args)
    glob_pattern = pattern.replace("%05d", "*").replace("%04d", "*").replace("%03d", "*")
    return sorted(out.glob(glob_pattern))


def extract_last_frame(video_path: Path | str, output_path: Path | str) -> Path:
    """Save the last frame of ``video_path`` as a PNG at ``output_path``."""
    require_ffmpeg()
    video = Path(video_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-1",
        "-i",
        str(video),
        "-update",
        "1",
        "-frames:v",
        "1",
        "-q:v",
        "1",
        str(out),
    ]
    _run(args)
    if not out.is_file():
        # ``-sseof`` can fail on tiny videos; fall back to last-by-index.
        info = get_video_info(video)
        idx = max(info.frame_count - 1, 0)
        args = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select='eq(n,{idx})'",
            "-vframes",
            "1",
            "-q:v",
            "1",
            str(out),
        ]
        _run(args)
    return out


def extract_frame_at(
    video_path: Path | str,
    output_path: Path | str,
    frame_index: int,
) -> Path:
    """Save frame ``frame_index`` of the video as a PNG."""
    require_ffmpeg()
    video = Path(video_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select='eq(n,{frame_index})'",
        "-vframes",
        "1",
        "-q:v",
        "1",
        str(out),
    ]
    _run(args)
    return out


def encode_from_frames(
    frames_dir: Path | str,
    output_path: Path | str,
    fps: int = 24,
    pattern: str = "frame_%05d.png",
    encoder: str | None = None,
    crf: int = 18,
    pixel_format: str = "yuv420p",
    extra_args: list[str] | None = None,
) -> Path:
    """Encode a directory of numbered frames into a video file."""
    require_ffmpeg()
    frames = Path(frames_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = encoder or detect_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames / pattern),
        "-c:v",
        enc,
        "-crf",
        str(crf),
        "-pix_fmt",
        pixel_format,
    ]
    if extra_args:
        args += extra_args
    args += [str(out)]
    _run(args)
    return out


def encode_from_frame_list(
    frame_paths: list[Path | str],
    output_path: Path | str,
    fps: int = 24,
    encoder: str | None = None,
    crf: int = 18,
    pixel_format: str = "yuv420p",
) -> Path:
    """Encode an arbitrary list of frame paths into a video.

    Useful when frames don't follow a numeric pattern. Internally writes a
    concat-demuxer manifest and feeds it to ffmpeg.
    """
    require_ffmpeg()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frame_paths:
        raise MediaError("encode_from_frame_list requires at least one frame")
    enc = encoder or detect_encoder()
    manifest = out.with_suffix(out.suffix + ".concat.txt")
    duration = 1.0 / fps
    with manifest.open("w") as f:
        for p in frame_paths:
            f.write(f"file '{Path(p).resolve()}'\nduration {duration}\n")
        # ffmpeg quirk: repeat the last frame to anchor its duration.
        f.write(f"file '{Path(frame_paths[-1]).resolve()}'\n")
    try:
        args = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c:v",
            enc,
            "-crf",
            str(crf),
            "-pix_fmt",
            pixel_format,
            "-r",
            str(fps),
            str(out),
        ]
        _run(args)
    finally:
        manifest.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------- #
# Stitch
# ---------------------------------------------------------------------- #
def build_stitch_args(
    inputs: list[Path],
    output_path: Path,
    drop_first_frame: bool,
    crossfade_frames: int,
    fps: int | None,
    encoder: str,
    crf: int,
) -> list[str]:
    """Return the ffmpeg argv that :func:`stitch_videos` would run.

    Exposed separately for testing — you can verify the constructed command
    without actually running ffmpeg.
    """
    args: list[str] = ["ffmpeg", "-y"]
    for p in inputs:
        args += ["-i", str(p)]

    filter_parts: list[str] = []
    labels: list[str] = []
    for i, _ in enumerate(inputs):
        src = f"[{i}:v]"
        if drop_first_frame and i > 0:
            filter_parts.append(
                f"{src}select='gte(n,1)',setpts=N/FRAME_RATE/TB[v{i}]"
            )
        else:
            filter_parts.append(f"{src}setpts=N/FRAME_RATE/TB[v{i}]")
        labels.append(f"[v{i}]")

    if crossfade_frames > 0 and len(inputs) > 1:
        if fps is None:
            raise MediaError("crossfade_frames requires fps to be specified")
        # Chain xfade between adjacent segments.
        durations = [get_video_info(p).duration_s for p in inputs]
        xfade_dur = crossfade_frames / fps
        prev_label = labels[0]
        offset = durations[0] - xfade_dur
        for i in range(1, len(inputs)):
            cur = labels[i]
            tag = "[outv]" if i == len(inputs) - 1 else f"[xf{i}]"
            filter_parts.append(
                f"{prev_label}{cur}xfade=transition=fade:duration={xfade_dur:.4f}:offset={offset:.4f}{tag}"
            )
            prev_label = tag
            if i < len(inputs) - 1:
                offset += durations[i] - xfade_dur
    else:
        filter_parts.append(
            "".join(labels) + f"concat=n={len(inputs)}:v=1:a=0[outv]"
        )

    args += [
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[outv]",
        "-c:v",
        encoder,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if fps:
        args += ["-r", str(fps)]
    args += [str(output_path)]
    return args


def stitch_videos(
    inputs: list[Path | str],
    output_path: Path | str,
    drop_first_frame: bool = False,
    crossfade_frames: int = 0,
    fps: int | None = None,
    encoder: str | None = None,
    crf: int = 18,
) -> Path:
    """Concatenate videos into one.

    When ``drop_first_frame`` is True, the first frame of every segment after
    the first is skipped — useful for chained video where each segment starts
    with the previous segment's last frame.

    When ``crossfade_frames > 0``, adjacent segments are crossfaded over that
    many frames using ffmpeg's ``xfade`` filter (requires ``fps``).
    """
    require_ffmpeg()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not inputs:
        raise MediaError("stitch_videos requires at least one input")
    paths = [Path(p) for p in inputs]
    if len(paths) == 1 and not drop_first_frame and crossfade_frames == 0:
        shutil.copyfile(paths[0], out)
        return out
    enc = encoder or detect_encoder()
    args = build_stitch_args(
        inputs=paths,
        output_path=out,
        drop_first_frame=drop_first_frame,
        crossfade_frames=crossfade_frames,
        fps=fps,
        encoder=enc,
        crf=crf,
    )
    _run(args)
    return out


# ---------------------------------------------------------------------- #
# Filters
# ---------------------------------------------------------------------- #
def apply_filter_chain(
    input_path: Path | str,
    output_path: Path | str,
    filters: list[str],
    fps: int | None = None,
    encoder: str | None = None,
    crf: int = 18,
    pixel_format: str = "yuv420p",
) -> Path:
    """Apply an arbitrary ffmpeg ``-vf`` filter chain.

    Example: ``apply_filter_chain(in, out, ["unsharp=5:5:1.0", "scale=iw*2:ih*2"])``.
    Use this for anything beyond what the dedicated helpers below cover.
    """
    require_ffmpeg()
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = encoder or detect_encoder()
    args = ["ffmpeg", "-y", "-i", str(inp)]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-c:v", enc, "-crf", str(crf), "-pix_fmt", pixel_format]
    if fps:
        args += ["-r", str(fps)]
    args += [str(out)]
    _run(args)
    return out


def interpolate(
    input_path: Path | str,
    output_path: Path | str,
    target_fps: int,
    method: str = "minterpolate",
    encoder: str | None = None,
    crf: int = 18,
) -> Path:
    """Increase the framerate of a video using motion-compensated interpolation.

    ``method`` is one of:

    * ``"minterpolate"`` — high-quality motion-compensated interpolation (slow).
    * ``"blend"`` — simple blended interpolation via ``framerate=`` filter (fast).
    """
    if method == "minterpolate":
        f = (
            f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:"
            f"me_mode=bidir:vsbmc=1"
        )
    elif method == "blend":
        f = f"framerate=fps={target_fps}:flags=scene_change_detect+bilinear"
    else:
        raise MediaError(f"Unknown interpolation method: {method!r}")
    return apply_filter_chain(
        input_path,
        output_path,
        filters=[f],
        fps=target_fps,
        encoder=encoder,
        crf=crf,
    )


def upscale(
    input_path: Path | str,
    output_path: Path | str,
    scale: float = 2.0,
    algorithm: str = "lanczos",
    encoder: str | None = None,
    crf: int = 18,
) -> Path:
    """Spatially upscale a video by a factor of ``scale``.

    ``algorithm`` is the ffmpeg ``scale`` filter's ``flags`` parameter —
    ``lanczos`` (default), ``bicubic``, ``spline``, or ``neighbor`` etc.
    For neural upscalers (RealESRGAN, etc.) wire your own pipeline using the
    Python API.
    """
    f = f"scale=iw*{scale}:ih*{scale}:flags={algorithm}"
    return apply_filter_chain(
        input_path,
        output_path,
        filters=[f],
        encoder=encoder,
        crf=crf,
    )


def sharpen(
    input_path: Path | str,
    output_path: Path | str,
    amount: float = 1.0,
    luma_size: int = 5,
    chroma_size: int = 5,
    encoder: str | None = None,
    crf: int = 18,
) -> Path:
    """Apply unsharp-mask sharpening.

    ``amount`` is the strength multiplier (typical 0.5–1.5).
    """
    f = f"unsharp={luma_size}:{luma_size}:{amount}:{chroma_size}:{chroma_size}:{amount}"
    return apply_filter_chain(
        input_path,
        output_path,
        filters=[f],
        encoder=encoder,
        crf=crf,
    )


def crop(
    input_path: Path | str,
    output_path: Path | str,
    width: int,
    height: int,
    x: int = 0,
    y: int = 0,
    encoder: str | None = None,
    crf: int = 18,
) -> Path:
    """Crop a video to ``width x height`` starting at ``(x, y)``."""
    return apply_filter_chain(
        input_path,
        output_path,
        filters=[f"crop={width}:{height}:{x}:{y}"],
        encoder=encoder,
        crf=crf,
    )


def transcode(
    input_path: Path | str,
    output_path: Path | str,
    encoder: str | None = None,
    crf: int = 18,
    pixel_format: str = "yuv420p",
) -> Path:
    """Re-encode a video without applying any filters (e.g. webp -> mp4)."""
    return apply_filter_chain(
        input_path,
        output_path,
        filters=[],
        encoder=encoder,
        crf=crf,
        pixel_format=pixel_format,
    )


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #
def _run(args: list[str], timeout: float | None = None) -> None:
    log.debug("ffmpeg cmd: %s", " ".join(args))
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffmpeg timed out: {exc}") from exc
    if result.returncode != 0:
        # ffmpeg writes useful info to stderr even on success; only log on error.
        tail = "\n".join(result.stderr.splitlines()[-30:])
        raise MediaError(
            f"ffmpeg exited with code {result.returncode}.\nLast lines of stderr:\n{tail}"
        )


__all__ = [
    "MediaError",
    "VideoInfo",
    "apply_filter_chain",
    "build_stitch_args",
    "crop",
    "detect_encoder",
    "encode_from_frame_list",
    "encode_from_frames",
    "extract_frame_at",
    "extract_frames",
    "extract_last_frame",
    "get_video_info",
    "interpolate",
    "require_ffmpeg",
    "sharpen",
    "stitch_videos",
    "transcode",
    "upscale",
]
