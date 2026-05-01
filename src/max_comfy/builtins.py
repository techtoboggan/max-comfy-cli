"""Built-in workflows that ship with max-comfy-cli.

These are registered automatically when ``max_comfy`` is imported. They cover
the cross-cutting patterns that almost every video pipeline needs:

* :class:`VideoChainWorkflow` — multi-segment video with last-frame chaining
  and stitch. Drives any per-segment "I2V"-style workflow you provide.
* :class:`VideoPostprocessWorkflow` — pure-ffmpeg pipeline of stitch /
  interpolate / upscale / sharpen / transcode. No ComfyUI needed.
* :class:`ReactorFaceSwapWorkflow` — frame-by-frame face swap on a finished
  video using ComfyUI's ReActor + (optional) CodeFormer custom nodes.

All three use the regular :class:`Workflow` API and can be invoked from the
CLI (``max-comfy run video-chain ...``), used programmatically, or composed
into your own workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import media
from .exceptions import WorkflowError
from .workflows import RunContext, RunResult, Workflow, register

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# video-chain
# ---------------------------------------------------------------------- #
@register
class VideoChainWorkflow(Workflow):
    """Multi-segment video with last-frame chaining and stitching.

    Runs a per-segment workflow N times. After each segment, the last frame of
    the produced video is fed back into the next segment as ``input_image``.
    The full sequence is then stitched (with optional first-frame deduplication
    and crossfade), and optionally interpolated, upscaled, and sharpened.

    **Parameters**

    * ``segment_workflow`` (required): name of the per-segment workflow. Must
      accept ``input_image`` (str, an uploaded ComfyUI input filename) as a
      param. Common params (prompt, seed, fps, etc.) flow through unchanged.
    * ``segments`` (required): list of dicts; one element per segment. Each
      dict is merged onto the common params (``defaults`` + top-level run
      params, minus ``segments`` and ``segment_workflow``). Use this to
      override per-segment ``prompt``, ``seed``, ``num_frames``, etc.
    * ``initial_image`` (optional): path to an image used as the first
      segment's input. If omitted, the first segment runs without
      ``input_image`` (text-to-video mode, if the segment workflow supports it).
    * ``stitch`` (default true): combine the segments into a single output.
    * ``drop_first_frame`` (default true): when stitching, drop frame 0 of every
      segment after the first (each segment's frame 0 = previous segment's
      last frame, so it would be a duplicate).
    * ``crossfade_frames`` (default 0): crossfade adjacent segments over this
      many frames. Requires ``fps``.
    * ``fps`` (default 24): assumed framerate of the segments + output.
    * ``interpolate_to_fps``, ``upscale``, ``sharpen``: optional post-processing
      applied to the stitched output. See :func:`max_comfy.media.interpolate`,
      :func:`~max_comfy.media.upscale`, :func:`~max_comfy.media.sharpen`.
    * ``output_filename`` (default ``"chained.mp4"``): name of the final
      stitched/post-processed file inside ``output_dir``.

    **Per-segment outputs**

    Each segment runs in its own subdirectory ``segment_NN/`` of the
    workflow's output directory. The segment's video is identified by its
    extension (``.webp`` / ``.mp4`` / ``.gif``) — the largest produced video
    file is treated as the segment output. Override that detection by setting
    ``segment_video_extensions`` (list of suffixes, ordered by preference).
    """

    name = "video-chain"
    description = "Multi-segment video: last-frame chaining + stitch + post-process."
    required = ("segment_workflow", "segments")
    defaults = {
        "stitch": True,
        "drop_first_frame": True,
        "crossfade_frames": 0,
        "fps": 24,
        "interpolate_to_fps": None,
        "upscale": None,
        "sharpen": None,
        "output_filename": "chained.mp4",
        "segment_video_extensions": [".webp", ".mp4", ".gif"],
        "encoder": None,
        "crf": 18,
        "initial_image": None,
        "upload_initial": True,
    }

    def run(self, ctx: RunContext) -> RunResult:
        if ctx.registry is None:
            raise WorkflowError(
                "video-chain requires a registry on the run context. "
                "Did you bypass the JobRunner?"
            )

        # Apply our own defaults so direct callers don't have to.
        params = {**self.defaults, **ctx.params}
        self.validate_params(params)

        segments = params["segments"]
        if not isinstance(segments, list) or not segments:
            raise WorkflowError("`segments` must be a non-empty list of dicts")

        seg_wf = ctx.registry.get(params["segment_workflow"])

        chain_only_keys = {
            "segments",
            "segment_workflow",
            "stitch",
            "drop_first_frame",
            "crossfade_frames",
            "interpolate_to_fps",
            "upscale",
            "sharpen",
            "output_filename",
            "segment_video_extensions",
            "encoder",
            "crf",
            "initial_image",
            "upload_initial",
        }
        common = {k: v for k, v in params.items() if k not in chain_only_keys}

        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        segment_videos: list[Path] = []
        last_frame_path: Path | None = None
        initial = params.get("initial_image")
        if initial:
            last_frame_path = Path(initial).expanduser()

        for i, overrides in enumerate(segments):
            if not isinstance(overrides, dict):
                raise WorkflowError(f"segments[{i}] must be a dict, got {type(overrides).__name__}")

            seg_dir = ctx.output_dir / f"segment_{i:02d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            seg_params: dict[str, Any] = {**common, **overrides, "segment_index": i}
            if last_frame_path is not None:
                if params.get("upload_initial", True):
                    name = ctx.client.upload_input(last_frame_path)
                    seg_params["input_image"] = name
                else:
                    seg_params["input_image"] = str(last_frame_path)

            seg_ctx = RunContext(
                client=ctx.client,
                config=ctx.config,
                params=seg_params,
                output_dir=seg_dir,
                job_id=f"{ctx.job_id}_seg{i:02d}" if ctx.job_id else f"seg{i:02d}",
                registry=ctx.registry,
            )

            log.info("video-chain: running segment %d/%d", i + 1, len(segments))
            seg_result = seg_wf.run(seg_ctx)

            video = self._pick_video(seg_result.files, params["segment_video_extensions"])
            if video is None:
                raise WorkflowError(
                    f"Segment {i} ({seg_wf.name!r}) produced no video file in {seg_dir}. "
                    f"Files: {[str(p) for p in seg_result.files]}"
                )
            segment_videos.append(video)

            # Extract the last frame for the next segment.
            if i < len(segments) - 1:
                last_frame_path = seg_dir / f"{video.stem}_lastframe.png"
                media.extract_last_frame(video, last_frame_path)

        # Build the final output.
        produced: list[Path] = list(segment_videos)
        if params["stitch"] and len(segment_videos) > 0:
            stitched = ctx.output_dir / params["output_filename"]
            if len(segment_videos) == 1:
                media.transcode(
                    segment_videos[0],
                    stitched,
                    encoder=params["encoder"],
                    crf=params["crf"],
                )
            else:
                media.stitch_videos(
                    segment_videos,
                    stitched,
                    drop_first_frame=params["drop_first_frame"],
                    crossfade_frames=params["crossfade_frames"],
                    fps=params["fps"],
                    encoder=params["encoder"],
                    crf=params["crf"],
                )
            current = stitched

            if params["interpolate_to_fps"]:
                interp = current.with_name(current.stem + "_interp" + current.suffix)
                media.interpolate(
                    current,
                    interp,
                    target_fps=int(params["interpolate_to_fps"]),
                    encoder=params["encoder"],
                    crf=params["crf"],
                )
                current = interp

            if params["upscale"]:
                ups = current.with_name(current.stem + "_up" + current.suffix)
                media.upscale(
                    current,
                    ups,
                    scale=float(params["upscale"]),
                    encoder=params["encoder"],
                    crf=params["crf"],
                )
                current = ups

            if params["sharpen"]:
                shp = current.with_name(current.stem + "_sharp" + current.suffix)
                media.sharpen(
                    current,
                    shp,
                    amount=float(params["sharpen"]),
                    encoder=params["encoder"],
                    crf=params["crf"],
                )
                current = shp

            produced.append(current)

        return RunResult(
            files=produced,
            metadata={
                "segments": len(segment_videos),
                "stitched": ctx.params["stitch"],
            },
        )

    @staticmethod
    def _pick_video(files: list[Path], extensions: list[str]) -> Path | None:
        for ext in extensions:
            ext = ext.lower()
            for f in files:
                if f.suffix.lower() == ext:
                    return f
        return None


# ---------------------------------------------------------------------- #
# video-postprocess
# ---------------------------------------------------------------------- #
@register
class VideoPostprocessWorkflow(Workflow):
    """Pure-ffmpeg post-processing pipeline (no ComfyUI submission).

    Use this when you already have a finished video and want to upscale,
    interpolate, sharpen, transcode, or stitch with another video. Useful as
    the last stage of a batch, or as a standalone tool:

    .. code-block:: bash

        max-comfy run video-postprocess \\
            -p input=in.mp4 \\
            -p output_filename=final.mp4 \\
            -p interpolate_to_fps=60 \\
            -p upscale=2 \\
            -p sharpen=1.0
    """

    name = "video-postprocess"
    description = "FFmpeg-only: stitch/interpolate/upscale/sharpen/transcode. No ComfyUI."
    required = ("input",)
    defaults = {
        "extra_inputs": [],  # additional inputs for stitching
        "stitch": False,
        "drop_first_frame": False,
        "crossfade_frames": 0,
        "fps": None,
        "interpolate_to_fps": None,
        "upscale": None,
        "sharpen": None,
        "transcode": False,
        "output_filename": "post.mp4",
        "encoder": None,
        "crf": 18,
    }

    def run(self, ctx: RunContext) -> RunResult:
        params = {**self.defaults, **ctx.params}
        self.validate_params(params)

        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        inputs = [Path(params["input"]).expanduser()]
        for extra in params.get("extra_inputs") or []:
            inputs.append(Path(extra).expanduser())

        out = ctx.output_dir / params["output_filename"]

        if params["stitch"] and len(inputs) > 1:
            media.stitch_videos(
                inputs,
                out,
                drop_first_frame=params["drop_first_frame"],
                crossfade_frames=params["crossfade_frames"],
                fps=params["fps"],
                encoder=params["encoder"],
                crf=params["crf"],
            )
            current = out
        elif params["transcode"] or len(inputs) == 1:
            media.transcode(
                inputs[0],
                out,
                encoder=params["encoder"],
                crf=params["crf"],
            )
            current = out
        else:
            current = inputs[0]

        if params["interpolate_to_fps"]:
            interp = current.with_name(current.stem + "_interp" + current.suffix)
            media.interpolate(
                current,
                interp,
                target_fps=int(params["interpolate_to_fps"]),
                encoder=params["encoder"],
                crf=params["crf"],
            )
            current = interp

        if params["upscale"]:
            ups = current.with_name(current.stem + "_up" + current.suffix)
            media.upscale(
                current,
                ups,
                scale=float(params["upscale"]),
                encoder=params["encoder"],
                crf=params["crf"],
            )
            current = ups

        if params["sharpen"]:
            shp = current.with_name(current.stem + "_sharp" + current.suffix)
            media.sharpen(
                current,
                shp,
                amount=float(params["sharpen"]),
                encoder=params["encoder"],
                crf=params["crf"],
            )
            current = shp

        return RunResult(files=[current])


# ---------------------------------------------------------------------- #
# reactor-face-swap
# ---------------------------------------------------------------------- #
@register
class ReactorFaceSwapWorkflow(Workflow):
    """Apply ReActor face-swap (and optional CodeFormer restore) to a video.

    Requires the **ReActor** custom node to be installed in your ComfyUI
    (``ComfyUI/custom_nodes/comfyui-reactor-node``). For best quality, also
    enable CodeFormer face restoration via ``restore_face=true``.

    The pipeline:

    1. Extract every frame of ``input_video`` to a working directory.
    2. Upload the reference face image to ComfyUI.
    3. For each frame, submit a graph: ``LoadImage(frame)`` ->
       ``ReActorFaceSwap(source=ref)`` -> optional ``ReActorRestoreFace`` ->
       ``SaveImage``. Frames are downloaded back into ``output_dir/enhanced/``.
    4. Re-encode the enhanced frames into a new video at the original fps
       (or ``fps`` if set).

    **Parameters**

    * ``input_video`` (required): path to a video file.
    * ``reference_face`` (required): path to a face image.
    * ``restore_face`` (default true): also run ReActorRestoreFace (CodeFormer).
    * ``codeformer_weight`` (default 0.5): CodeFormer fidelity ↔ quality.
    * ``face_index_input`` / ``face_index_source`` (default ``"0"``): comma-separated
      face indexes within input/source images (ReActor convention).
    * ``fps`` (optional): output framerate. Defaults to the input's fps.
    * ``encoder`` / ``crf``: forwarded to the encode step.
    * ``output_filename`` (default ``"reactor.mp4"``).
    * ``frames_subdir`` (default ``"frames"``): where extracted frames live.
    * ``enhanced_subdir`` (default ``"enhanced"``): where enhanced frames live.
    """

    name = "reactor-face-swap"
    description = "Per-frame ReActor face swap on a video (requires ReActor custom node)."
    required = ("input_video", "reference_face")
    defaults = {
        "restore_face": True,
        "codeformer_weight": 0.5,
        "face_index_input": "0",
        "face_index_source": "0",
        "face_restore_model": "codeformer-v0.1.0.pth",
        "face_restore_visibility": 1.0,
        "swap_model": "inswapper_128.onnx",
        "fps": None,
        "encoder": None,
        "crf": 18,
        "output_filename": "reactor.mp4",
        "frames_subdir": "frames",
        "enhanced_subdir": "enhanced",
        "save_prefix": "reactor_frame",
    }

    def run(self, ctx: RunContext) -> RunResult:
        params = {**self.defaults, **ctx.params}
        self.validate_params(params)

        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        input_video = Path(params["input_video"]).expanduser()
        ref_face = Path(params["reference_face"]).expanduser()
        if not input_video.is_file():
            raise WorkflowError(f"input_video not found: {input_video}")
        if not ref_face.is_file():
            raise WorkflowError(f"reference_face not found: {ref_face}")

        info = media.get_video_info(input_video)
        fps = params.get("fps") or int(round(info.fps)) or 24

        frames_dir = ctx.output_dir / params["frames_subdir"]
        enhanced_dir = ctx.output_dir / params["enhanced_subdir"]
        frames_dir.mkdir(parents=True, exist_ok=True)
        enhanced_dir.mkdir(parents=True, exist_ok=True)

        log.info("reactor: extracting frames from %s", input_video)
        frames = media.extract_frames(input_video, frames_dir)
        log.info("reactor: %d frames extracted", len(frames))

        ref_uploaded = ctx.client.upload_input(ref_face)
        log.info("reactor: reference face uploaded as %s", ref_uploaded)

        # Process frame-by-frame. ComfyUI runs the work; we collect outputs.
        save_prefix = params["save_prefix"]
        for i, frame in enumerate(frames):
            existing = sorted(enhanced_dir.glob(f"{save_prefix}_{i:05d}_*.png"))
            if existing:
                continue  # resume support: skip already-processed frames
            uploaded = ctx.client.upload_input(frame)
            graph = self._build_graph(
                frame_filename=uploaded,
                ref_filename=ref_uploaded,
                save_prefix=f"{save_prefix}_{i:05d}",
                params=params,
            )
            ctx.client.submit_and_wait(
                graph,
                timeout=ctx.config.completion_timeout,
                poll_interval=ctx.config.poll_interval,
                download_dir=enhanced_dir,
            )
            if (i + 1) % 25 == 0 or i == len(frames) - 1:
                log.info("reactor: %d/%d frames processed", i + 1, len(frames))

        enhanced_frames = sorted(enhanced_dir.glob(f"{save_prefix}_*.png"))
        if not enhanced_frames:
            raise WorkflowError("ReActor produced no output frames")

        out_path = ctx.output_dir / params["output_filename"]
        media.encode_from_frame_list(
            enhanced_frames,
            out_path,
            fps=int(fps),
            encoder=params["encoder"],
            crf=params["crf"],
        )
        return RunResult(
            files=[out_path],
            metadata={"frames_processed": len(enhanced_frames)},
        )

    @staticmethod
    def _build_graph(
        frame_filename: str,
        ref_filename: str,
        save_prefix: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        graph: dict[str, Any] = {
            "ref": {"class_type": "LoadImage", "inputs": {"image": ref_filename}},
            "frame": {"class_type": "LoadImage", "inputs": {"image": frame_filename}},
            "swap": {
                "class_type": "ReActorFaceSwap",
                "inputs": {
                    "enabled": True,
                    "input_image": ["frame", 0],
                    "source_image": ["ref", 0],
                    "swap_model": params["swap_model"],
                    "facedetection": "retinaface_resnet50",
                    "face_restore_model": params["face_restore_model"]
                    if params["restore_face"]
                    else "none",
                    "face_restore_visibility": params["face_restore_visibility"],
                    "codeformer_weight": params["codeformer_weight"],
                    "detect_gender_input": "no",
                    "detect_gender_source": "no",
                    "input_faces_index": params["face_index_input"],
                    "source_faces_index": params["face_index_source"],
                    "console_log_level": 1,
                },
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["swap", 0],
                    "filename_prefix": save_prefix,
                },
            },
        }
        return graph
