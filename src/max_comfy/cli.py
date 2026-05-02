"""max-comfy command-line interface."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from . import __version__, media
from .client import Client
from .config import Config
from .exceptions import MaxComfyError
from .runner import Batch, JobRunner
from .server import Server
from .workflows import TemplateWorkflow, discover, registry

log = logging.getLogger("max_comfy")


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_param(text: str) -> tuple[str, Any]:
    """Parse a ``key=value`` CLI param, with type inference + JSON fallback."""
    if "=" not in text:
        raise click.BadParameter(f"--param must be key=value (got {text!r})")
    key, raw = text.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    # Try JSON first so users can pass numbers, bools, lists, dicts.
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def _params_from_options(values: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v in values:
        k, val = _parse_param(v)
        out[k] = val
    return out


def _load_config_with_overrides(
    config_path: Path | None,
    overrides: dict[str, Any],
) -> Config:
    cfg = Config.load(explicit=config_path)
    if overrides:
        cfg = cfg.merged_with(overrides)
    return cfg


def _populate_registry(cfg: Config) -> None:
    discover(dirs=cfg.resolved_workflow_dirs(), modules=cfg.workflow_modules)


def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


# ---------------------------------------------------------------------- #
# Top-level group
# ---------------------------------------------------------------------- #
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version")
@click.option("-v", "--verbose", count=True, help="Increase log verbosity (-v, -vv).")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a TOML config file.",
)
@click.option("--host", help="ComfyUI host (default 127.0.0.1).")
@click.option("--port", type=int, help="ComfyUI port (default 8188).")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Override output directory.",
)
@click.option(
    "--workflow-dir",
    "workflow_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    help="Additional directory to scan for workflows. May be repeated.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: int,
    config_path: Path | None,
    host: str | None,
    port: int | None,
    output_dir: Path | None,
    workflow_dirs: tuple[Path, ...],
) -> None:
    """Generic CLI for selecting, parameterising and batching ComfyUI workflows."""
    _setup_logging(verbose)
    overrides: dict[str, Any] = {}
    if host:
        overrides["comfyui_host"] = host
    if port:
        overrides["comfyui_port"] = port
    if output_dir:
        overrides["output_dir"] = str(output_dir)
    if workflow_dirs:
        overrides["workflow_dirs"] = [str(p) for p in workflow_dirs]
    cfg = _load_config_with_overrides(config_path, overrides)
    _populate_registry(cfg)
    ctx.obj = cfg


# ---------------------------------------------------------------------- #
# `list`
# ---------------------------------------------------------------------- #
@cli.command("list")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
@click.pass_obj
def list_workflows(cfg: Config, as_json: bool) -> None:
    """List discovered workflows."""
    items = registry.items()
    if as_json:
        click.echo(json.dumps(
            [
                {"name": name, "description": wf.description, "type": type(wf).__name__}
                for name, wf in items
            ],
            indent=2,
        ))
        return
    if not items:
        click.echo("No workflows found.")
        click.echo()
        click.echo("Search paths:")
        for d in cfg.resolved_workflow_dirs():
            marker = " " if d.is_dir() else "*"
            click.echo(f"  {marker} {d}")
        click.echo()
        click.echo(
            "Add JSON templates or Python plugins to one of these directories,\n"
            "or set the MAX_COMFY_WORKFLOWS environment variable."
        )
        return
    name_w = max(len(n) for n, _ in items)
    type_w = max(len(type(wf).__name__) for _, wf in items)
    click.echo(f"{'NAME'.ljust(name_w)}  {'TYPE'.ljust(type_w)}  DESCRIPTION")
    click.echo(f"{'-' * name_w}  {'-' * type_w}  -----------")
    for name, wf in items:
        click.echo(
            f"{name.ljust(name_w)}  "
            f"{type(wf).__name__.ljust(type_w)}  "
            f"{wf.description or '-'}"
        )


# ---------------------------------------------------------------------- #
# `show`
# ---------------------------------------------------------------------- #
@cli.command("show")
@click.argument("workflow_name")
@click.pass_obj
def show_workflow(cfg: Config, workflow_name: str) -> None:
    """Show details for a workflow (placeholders, defaults, source)."""
    wf = registry.get(workflow_name)
    click.echo(f"Workflow: {wf.name}")
    click.echo(f"Type:     {type(wf).__name__}")
    click.echo(f"Source:   {getattr(wf, 'path', '(python plugin)')}")
    if wf.description:
        click.echo(f"Summary:  {wf.description}")
    if wf.required:
        click.echo(f"Required: {', '.join(wf.required)}")
    if wf.defaults:
        click.echo("Defaults:")
        for k, v in wf.defaults.items():
            click.echo(f"  {k} = {v!r}")
    if isinstance(wf, TemplateWorkflow):
        placeholders = sorted(wf.placeholders())
        if placeholders:
            click.echo(f"Placeholders ({len(placeholders)}): {', '.join(placeholders)}")


# ---------------------------------------------------------------------- #
# `render`
# ---------------------------------------------------------------------- #
@cli.command("render")
@click.argument("workflow_name")
@click.option("-p", "--param", "params", multiple=True, help="key=value (repeatable).")
@click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Load params from a JSON file.",
)
@click.pass_obj
def render_workflow(
    cfg: Config,
    workflow_name: str,
    params: tuple[str, ...],
    params_file: Path | None,
) -> None:
    """Render a workflow template to JSON without submitting (for debugging)."""
    wf = registry.get(workflow_name)
    p: dict[str, Any] = {}
    if params_file:
        p.update(json.loads(params_file.read_text()))
    p.update(_params_from_options(params))
    merged = {**wf.defaults, **cfg.defaults, **p}
    try:
        graph = wf.build(merged)
    except NotImplementedError:
        raise click.ClickException(
            f"Workflow {workflow_name!r} does not implement build() — it overrides run() instead. "
            "Render is only available for build-style workflows."
        ) from None
    click.echo(json.dumps(graph, indent=2))


# ---------------------------------------------------------------------- #
# `run`
# ---------------------------------------------------------------------- #
@cli.command("run")
@click.argument("workflow_name")
@click.option("-p", "--param", "params", multiple=True, help="key=value (repeatable).")
@click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Load params from a JSON file.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for this run.",
)
@click.option("--id", "job_id", default="single", help="Identifier for this run (default: 'single').")
@click.option("--no-server", is_flag=True, help="Skip auto-spawning a managed ComfyUI server.")
@click.pass_obj
def run_one(
    cfg: Config,
    workflow_name: str,
    params: tuple[str, ...],
    params_file: Path | None,
    output_dir: Path | None,
    job_id: str,
    no_server: bool,
) -> None:
    """Run a single workflow once."""
    p: dict[str, Any] = {}
    if params_file:
        p.update(json.loads(params_file.read_text()))
    p.update(_params_from_options(params))

    runner = JobRunner(cfg)
    if not no_server:
        try:
            runner.ensure_server()
        except MaxComfyError as exc:
            raise click.ClickException(str(exc)) from None
    try:
        result = runner.run_one(
            workflow=workflow_name,
            params=p,
            output_dir=output_dir,
            job_id=job_id,
        )
    finally:
        runner.shutdown()

    if result.success:
        click.echo(f"OK  {result.job_id}  ({result.duration_s:.1f}s)")
        for f in result.files:
            click.echo(f"    -> {f}")
    else:
        click.echo(f"FAIL {result.job_id}: {result.error}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------- #
# `batch`
# ---------------------------------------------------------------------- #
@cli.command("batch")
@click.argument("jobs_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--resume", is_flag=True, help="Skip jobs already completed (per .progress.json).")
@click.option("--no-server", is_flag=True, help="Skip auto-spawning a managed ComfyUI server.")
@click.pass_obj
def batch_run(cfg: Config, jobs_file: Path, resume: bool, no_server: bool) -> None:
    """Run a batch of jobs from a JSON file."""
    runner = JobRunner(cfg)
    if not no_server:
        try:
            runner.ensure_server()
        except MaxComfyError as exc:
            raise click.ClickException(str(exc)) from None

    def on_result(result: Any, idx: int, total: int) -> None:
        status = "OK  " if result.success else "FAIL"
        click.echo(f"[{idx + 1}/{total}] {status} {result.job_id}  ({result.duration_s:.1f}s)")
        if not result.success:
            click.echo(f"      {result.error}", err=True)
        for f in result.files:
            click.echo(f"      -> {f}")

    try:
        results = runner.run_batch(jobs_file, resume=resume, on_result=on_result)
    finally:
        runner.shutdown()

    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    click.echo()
    click.echo(f"Done: {succeeded} ok, {failed} failed")
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------- #
# `validate`
# ---------------------------------------------------------------------- #
@cli.command("validate")
@click.argument("jobs_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_obj
def validate_jobs(cfg: Config, jobs_file: Path) -> None:
    """Validate a jobs file without running it. Useful in CI."""
    try:
        batch = Batch.load(jobs_file)
    except MaxComfyError as exc:
        raise click.ClickException(str(exc)) from None

    issues: list[str] = []
    workflows_used: set[str] = set()
    if batch.workflow:
        workflows_used.add(batch.workflow)
    for job in batch.jobs:
        wf_name = job.workflow or batch.workflow
        if not wf_name:
            issues.append(f"Job {job.id}: no workflow specified")
            continue
        workflows_used.add(wf_name)
        if wf_name not in registry:
            issues.append(f"Job {job.id}: unknown workflow {wf_name!r}")
            continue
        wf = registry.get(wf_name)
        merged = {**cfg.defaults, **batch.defaults, **wf.defaults, **job.params}
        for req in wf.required:
            if req not in merged:
                issues.append(f"Job {job.id}: missing required param {req!r}")

    click.echo(f"Jobs:      {len(batch.jobs)}")
    click.echo(f"Workflows: {', '.join(sorted(workflows_used)) or '(none)'}")
    click.echo(f"Output:    {batch.output_dir}")
    if issues:
        click.echo()
        click.echo(f"{len(issues)} issue(s):", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)
    click.echo()
    click.echo("All jobs valid.")


# ---------------------------------------------------------------------- #
# `server` group
# ---------------------------------------------------------------------- #
@cli.group("server")
def server_group() -> None:
    """Manage a ComfyUI subprocess (optional convenience)."""


@server_group.command("status")
@click.pass_obj
def server_status(cfg: Config) -> None:
    """Check whether ComfyUI is reachable on the configured host:port."""
    client = Client(host=cfg.comfyui_host, port=cfg.comfyui_port)
    if client.health_check():
        click.echo(f"OK  {cfg.comfyui_host}:{cfg.comfyui_port} is reachable.")
        try:
            stats = client.system_stats()
            sysinfo = stats.get("system", {})
            click.echo(f"    OS:      {sysinfo.get('os', 'unknown')}")
            click.echo(f"    Python:  {sysinfo.get('python_version', 'unknown')}")
            for dev in stats.get("devices", []):
                name = dev.get("name", "?")
                vram_total = dev.get("vram_total")
                vram_free = dev.get("vram_free")
                if vram_total and vram_free:
                    click.echo(
                        f"    Device:  {name}  "
                        f"({_format_size(vram_free)} free / {_format_size(vram_total)})"
                    )
                else:
                    click.echo(f"    Device:  {name}")
        except Exception:  # noqa: BLE001
            pass
    else:
        click.echo(f"FAIL  {cfg.comfyui_host}:{cfg.comfyui_port} is not reachable.")
        sys.exit(1)


@server_group.command("start")
@click.option("--foreground/--background", default=True, help="Run in foreground (default).")
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write ComfyUI stdout/stderr to this file.",
)
@click.pass_obj
def server_start(cfg: Config, foreground: bool, log_file: Path | None) -> None:
    """Spawn a ComfyUI subprocess with the configured paths."""
    if cfg.comfyui_path is None:
        raise click.ClickException(
            "comfyui_path is not configured. Set it in your TOML config or use an existing "
            "ComfyUI server and skip `server start`."
        )
    server = Server(
        comfyui_path=cfg.comfyui_path,
        host=cfg.comfyui_host,
        port=cfg.comfyui_port,
        output_dir=cfg.output_dir,
        input_dir=cfg.input_dir,
        python_exe=cfg.python_exe,
        extra_args=cfg.server_extra_args,
        log_file=log_file,
        start_timeout=cfg.server_start_timeout,
    )
    try:
        server.start()
    except MaxComfyError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"ComfyUI started on {cfg.comfyui_host}:{cfg.comfyui_port} (PID {server._process.pid if server._process else '?'})")
    if not foreground:
        return
    click.echo("Running in foreground. Press Ctrl+C to stop.")
    try:
        if server._process:
            server._process.wait()
    except KeyboardInterrupt:
        click.echo()
        click.echo("Stopping ComfyUI...")
        server.stop()


@server_group.command("stop")
@click.pass_obj
def server_stop(cfg: Config) -> None:
    """Politely ask the running ComfyUI to free memory (no kill — start was foreground)."""
    client = Client(host=cfg.comfyui_host, port=cfg.comfyui_port)
    if not client.health_check():
        click.echo("Server is not reachable; nothing to stop.")
        return
    client.free_memory()
    click.echo("Sent free-memory request to ComfyUI.")


# ---------------------------------------------------------------------- #
# `media` group — pure-ffmpeg utilities (no ComfyUI required)
# ---------------------------------------------------------------------- #
@cli.group("media")
def media_group() -> None:
    """FFmpeg-backed media utilities (info/extract/stitch/encode/interpolate/upscale/sharpen).

    All commands shell out to ffmpeg + ffprobe — no ComfyUI server is touched.
    """


@media_group.command("info")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def media_info(video: Path, as_json: bool) -> None:
    """Probe a video for fps / dimensions / duration / codec."""
    info = media.get_video_info(video)
    if as_json:
        click.echo(json.dumps({
            "path": str(info.path),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "duration_s": info.duration_s,
            "frame_count": info.frame_count,
            "codec": info.codec,
            "pixel_format": info.pixel_format,
        }, indent=2))
        return
    click.echo(f"Path:       {info.path}")
    click.echo(f"Dimensions: {info.width}x{info.height}")
    click.echo(f"FPS:        {info.fps:.3f}")
    click.echo(f"Duration:   {info.duration_s:.3f}s")
    click.echo(f"Frames:     {info.frame_count}")
    click.echo(f"Codec:      {info.codec}")
    click.echo(f"Pixel fmt:  {info.pixel_format}")


@media_group.command("extract-frames")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write frames into.",
)
@click.option("--fps", type=float, help="Resample to this fps before extraction.")
@click.option("--pattern", default="frame_%05d.png", show_default=True)
def media_extract_frames(
    video: Path, output_dir: Path, fps: float | None, pattern: str
) -> None:
    """Extract every frame of a video as PNG."""
    frames = media.extract_frames(video, output_dir, pattern=pattern, fps=fps)
    click.echo(f"Wrote {len(frames)} frames to {output_dir}")


@media_group.command("extract-last-frame")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output PNG path.",
)
def media_extract_last(video: Path, output: Path) -> None:
    """Extract the last frame of a video as PNG."""
    media.extract_last_frame(video, output)
    click.echo(f"Wrote {output}")


@media_group.command("encode")
@click.argument(
    "frames_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--fps", type=int, default=24, show_default=True)
@click.option("--pattern", default="frame_%05d.png", show_default=True)
@click.option("--encoder", help="Override encoder (default: auto-detect).")
@click.option("--crf", type=int, default=18, show_default=True)
def media_encode(
    frames_dir: Path,
    output: Path,
    fps: int,
    pattern: str,
    encoder: str | None,
    crf: int,
) -> None:
    """Encode a directory of numbered frames into a video."""
    media.encode_from_frames(
        frames_dir, output, fps=fps, pattern=pattern, encoder=encoder, crf=crf
    )
    click.echo(f"Wrote {output}")


@media_group.command("stitch")
@click.argument(
    "inputs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--drop-first-frame",
    is_flag=True,
    help="Skip frame 0 of every segment after the first (deduplicates chained videos).",
)
@click.option("--crossfade-frames", type=int, default=0, show_default=True)
@click.option("--fps", type=int, help="Required when --crossfade-frames > 0.")
@click.option("--encoder", help="Override encoder (default: auto-detect).")
@click.option("--crf", type=int, default=18, show_default=True)
def media_stitch(
    inputs: tuple[Path, ...],
    output: Path,
    drop_first_frame: bool,
    crossfade_frames: int,
    fps: int | None,
    encoder: str | None,
    crf: int,
) -> None:
    """Concatenate two or more videos into one."""
    media.stitch_videos(
        list(inputs),
        output,
        drop_first_frame=drop_first_frame,
        crossfade_frames=crossfade_frames,
        fps=fps,
        encoder=encoder,
        crf=crf,
    )
    click.echo(f"Wrote {output}")


@media_group.command("interpolate")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--fps", type=int, required=True, help="Target framerate.")
@click.option(
    "--method",
    type=click.Choice(["minterpolate", "blend"]),
    default="minterpolate",
    show_default=True,
)
@click.option("--encoder", help="Override encoder (default: auto-detect).")
@click.option("--crf", type=int, default=18, show_default=True)
def media_interpolate(
    video: Path, output: Path, fps: int, method: str, encoder: str | None, crf: int
) -> None:
    """Frame-interpolate a video to a higher framerate."""
    media.interpolate(video, output, target_fps=fps, method=method, encoder=encoder, crf=crf)
    click.echo(f"Wrote {output}")


@media_group.command("upscale")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--scale", type=float, default=2.0, show_default=True)
@click.option(
    "--algorithm",
    default="lanczos",
    show_default=True,
    help="ffmpeg scale flags (lanczos/bicubic/spline/neighbor).",
)
@click.option("--encoder", help="Override encoder (default: auto-detect).")
@click.option("--crf", type=int, default=18, show_default=True)
def media_upscale(
    video: Path, output: Path, scale: float, algorithm: str, encoder: str | None, crf: int
) -> None:
    """Spatially upscale a video by a factor."""
    media.upscale(video, output, scale=scale, algorithm=algorithm, encoder=encoder, crf=crf)
    click.echo(f"Wrote {output}")


@media_group.command("sharpen")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--amount", type=float, default=1.0, show_default=True)
@click.option("--encoder", help="Override encoder (default: auto-detect).")
@click.option("--crf", type=int, default=18, show_default=True)
def media_sharpen(
    video: Path, output: Path, amount: float, encoder: str | None, crf: int
) -> None:
    """Apply unsharp-mask sharpening."""
    media.sharpen(video, output, amount=amount, encoder=encoder, crf=crf)
    click.echo(f"Wrote {output}")


@media_group.command("transcode")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--encoder", help="Override encoder (default: auto-detect).")
@click.option("--crf", type=int, default=18, show_default=True)
def media_transcode(video: Path, output: Path, encoder: str | None, crf: int) -> None:
    """Re-encode a video without applying any filters (e.g. webp -> mp4)."""
    media.transcode(video, output, encoder=encoder, crf=crf)
    click.echo(f"Wrote {output}")


@media_group.command("detect-encoder")
def media_detect_encoder() -> None:
    """Print the encoder that max-comfy will use by default."""
    click.echo(media.detect_encoder())


# ---------------------------------------------------------------------- #
# Onboarding: doctor / init / models / install-comfyui
# ---------------------------------------------------------------------- #
_OK = "[OK]  "
_WARN = "[WARN]"
_FAIL = "[FAIL]"
_INFO = "[INFO]"


@cli.command("doctor")
@click.pass_obj
def doctor(cfg: Config) -> None:
    """Diagnose the environment: ffmpeg, ComfyUI reachability, models, workflows."""
    issues = 0
    click.echo("Checking environment...\n")

    # --- ffmpeg / ffprobe ---------------------------------------------- #
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path:
        click.echo(f"  {_OK} ffmpeg  -> {ffmpeg_path}")
    else:
        click.echo(f"  {_FAIL} ffmpeg not on PATH")
        click.echo("         Install: https://ffmpeg.org/download.html")
        issues += 1
    if ffprobe_path:
        click.echo(f"  {_OK} ffprobe -> {ffprobe_path}")
    else:
        click.echo(f"  {_FAIL} ffprobe not on PATH (ships with ffmpeg)")
        issues += 1
    if ffmpeg_path and ffprobe_path:
        try:
            enc = media.detect_encoder()
            click.echo(f"  {_OK} ffmpeg encoder available: {enc}")
        except MaxComfyError as exc:
            click.echo(f"  {_WARN} encoder detection: {exc}")
            issues += 1

    # --- ComfyUI reachable --------------------------------------------- #
    click.echo()
    client = Client(host=cfg.comfyui_host, port=cfg.comfyui_port)
    if client.health_check():
        click.echo(f"  {_OK} ComfyUI reachable at {cfg.comfyui_host}:{cfg.comfyui_port}")
        try:
            stats = client.system_stats()
            for dev in stats.get("devices") or []:
                vram_total = dev.get("vram_total")
                vram_free = dev.get("vram_free")
                line = f"         GPU: {dev.get('name', '?')}"
                if vram_total and vram_free:
                    line += f"  ({_format_size(vram_free)} free / {_format_size(vram_total)})"
                click.echo(line)
        except Exception:  # noqa: BLE001
            pass
    else:
        click.echo(f"  {_WARN} ComfyUI not reachable at {cfg.comfyui_host}:{cfg.comfyui_port}")
        click.echo("         Start it: cd <ComfyUI dir> && python main.py")
        click.echo("         Or set comfyui_path and run `max-comfy server start`.")
        issues += 1

    # --- ComfyUI install path ------------------------------------------ #
    click.echo()
    if cfg.comfyui_path:
        path = Path(cfg.comfyui_path).expanduser()
        if path.is_dir() and (path / "main.py").is_file():
            click.echo(f"  {_OK} ComfyUI install at {path}")
            models_dir = path / "models"
            if models_dir.is_dir():
                cats = [d for d in models_dir.iterdir() if d.is_dir()]
                total_files = sum(
                    1 for c in cats for f in c.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                )
                click.echo(f"  {_OK} models/ has {len(cats)} categories, {total_files} files")
                click.echo("         Run `max-comfy models list` for the full inventory.")
            else:
                click.echo(f"  {_WARN} no models/ subdir at {path}")
                issues += 1
        else:
            click.echo(f"  {_FAIL} comfyui_path is set but invalid: {cfg.comfyui_path}")
            click.echo(f"         Expected directory containing main.py at {path}")
            issues += 1
    else:
        click.echo(f"  {_INFO} comfyui_path is not configured.")
        click.echo("         Optional — only needed for `server start` and `models list`.")
        click.echo("         Run `max-comfy init` to create a starter config.")

    # --- Workflow discovery -------------------------------------------- #
    click.echo()
    names = registry.names()
    click.echo(f"  Workflows discovered: {len(names)}")
    for n in names:
        click.echo(f"    - {n}")

    click.echo()
    if issues == 0:
        click.echo("All checks passed.")
    else:
        click.echo(f"{issues} issue(s) found. See messages above.")
        sys.exit(1)


@cli.command("init")
@click.option(
    "--path",
    "init_path",
    type=click.Path(path_type=Path),
    default=Path("./max-comfy.toml"),
    show_default=True,
    help="Where to write the config file.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def init_config(init_path: Path, force: bool) -> None:
    """Create a starter `max-comfy.toml` in the current directory.

    Auto-detects a ComfyUI install at common paths (`~/ComfyUI`, `./ComfyUI`)
    and pre-populates `comfyui_path` if it finds one.
    """
    if init_path.exists() and not force:
        raise click.ClickException(
            f"{init_path} already exists. Re-run with --force to overwrite."
        )

    candidates = [
        Path.home() / "ComfyUI",
        Path.home() / "comfyui",
        Path.cwd() / "ComfyUI",
        Path.cwd().parent / "ComfyUI",
    ]
    detected: Path | None = None
    for c in candidates:
        if (c / "main.py").is_file():
            detected = c
            break

    lines = [
        "# max-comfy-cli — starter config (created by `max-comfy init`).",
        "# See examples/configs/example.toml in the repo for the full annotated reference.",
        "",
        '# ComfyUI server connection',
        'comfyui_host = "127.0.0.1"',
        "comfyui_port = 8188",
        "",
        "# ComfyUI install path (only required for `server start`, `models list`,",
        "# and `doctor` model inventory).",
    ]
    if detected:
        lines.append(f'comfyui_path = "{detected}"')
    else:
        lines.append('# comfyui_path = "~/ComfyUI"')
    lines += [
        "",
        "# I/O",
        'output_dir = "./output"',
        "",
        "# Where to look for workflow plugins (in addition to ~/.config/max-comfy/workflows/",
        "# and ./workflows/, both of which are always scanned).",
        'workflow_dirs = ["./workflows"]',
        "",
        "# Polling",
        "poll_interval = 1.0",
        "completion_timeout = 7200.0  # 2h — raise for very long video jobs",
        "",
        "# Defaults applied to every job's params (job values override).",
        "[defaults]",
        '# checkpoint = "sd_xl_base_1.0.safetensors"',
        '# negative_prompt = "blurry, low quality, distorted"',
        "",
    ]
    init_path.write_text("\n".join(lines))
    click.echo(f"Wrote {init_path}")
    if detected:
        click.echo(f"Detected ComfyUI at {detected}")
    else:
        click.echo(
            "No ComfyUI install detected. Edit the file and uncomment "
            "`comfyui_path` once you have one (or run `max-comfy install-comfyui`)."
        )

    click.echo()
    click.echo("Next steps:")
    click.echo("  1. Place models under <ComfyUI>/models/checkpoints, /loras, /vae, etc.")
    click.echo("  2. Run `max-comfy doctor` to verify the environment.")
    click.echo("  3. Run `max-comfy list` to see available workflows.")


@cli.group("models")
def models_group() -> None:
    """Inspect the ComfyUI models on disk."""


_MODEL_CATEGORY_HINTS = {
    "checkpoints": "Full models — SDXL, SD1.5, Flux, SVD, etc.",
    "loras": "LoRA adapters",
    "vae": "Standalone VAEs",
    "clip": "Text encoders (CLIP, T5, Gemma)",
    "clip_vision": "CLIP vision encoders (SVD, IPAdapter)",
    "controlnet": "ControlNet models",
    "upscale_models": "RealESRGAN, etc.",
    "embeddings": "Textual inversion embeddings",
    "hypernetworks": "Hypernetworks",
    "ipadapter": "IPAdapter models",
    "unet": "Standalone UNETs (some video models)",
    "diffusion_models": "Diffusion models (Flux, etc.)",
    "vae_approx": "Latent previewers",
}


@models_group.command("list")
@click.option(
    "--category",
    help="Show only one category (e.g. checkpoints, loras, vae). Default: all.",
)
@click.option("--limit", type=int, default=50, show_default=True, help="Max files per category.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.pass_obj
def models_list(
    cfg: Config, category: str | None, limit: int, as_json: bool
) -> None:
    """List models found in your ComfyUI install."""
    if not cfg.comfyui_path:
        raise click.ClickException(
            "comfyui_path is not configured. Run `max-comfy init`, "
            "or set it manually in your TOML config."
        )
    base = Path(cfg.comfyui_path).expanduser() / "models"
    if not base.is_dir():
        raise click.ClickException(f"No models/ directory at {base}")

    categories: list[Path]
    if category:
        target = base / category
        if not target.is_dir():
            raise click.ClickException(f"Category not found: {target}")
        categories = [target]
    else:
        categories = sorted([d for d in base.iterdir() if d.is_dir()])

    inventory: dict[str, list[dict[str, Any]]] = {}
    for cat in categories:
        files = sorted(
            (f for f in cat.iterdir() if f.is_file() and not f.name.startswith(".")),
            key=lambda p: p.name,
        )
        inventory[cat.name] = [
            {"name": f.name, "size": f.stat().st_size, "path": str(f)}
            for f in files
        ]

    if as_json:
        click.echo(json.dumps(inventory, indent=2))
        return

    click.echo(f"ComfyUI: {base.parent}\n")
    if not inventory:
        click.echo("(no model categories found)")
        return
    for name, files in inventory.items():
        total_bytes = sum(f["size"] for f in files)
        hint = _MODEL_CATEGORY_HINTS.get(name)
        header = f"{name}/  ({len(files)} files, {_format_size(total_bytes)})"
        if hint:
            header += f"   — {hint}"
        click.echo(header)
        if not files:
            click.echo("  (empty)")
        else:
            for f in files[:limit]:
                click.echo(f"  {f['name'].ljust(60)} {_format_size(f['size']):>10}")
            if len(files) > limit:
                click.echo(f"  ... and {len(files) - limit} more (use --limit to show)")
        click.echo()


@cli.command("selftest")
@click.option(
    "--checkpoint",
    help="Checkpoint name to use. Defaults to the first one found in <ComfyUI>/models/checkpoints/.",
)
@click.option(
    "--steps",
    type=int,
    default=1,
    show_default=True,
    help="KSampler steps. Keep low — this is a smoke test.",
)
@click.option(
    "--size",
    type=int,
    default=64,
    show_default=True,
    help="Square image size in pixels. Tiny by default for speed.",
)
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    show_default=True,
    help="Max seconds to wait for completion.",
)
@click.pass_obj
def selftest(
    cfg: Config,
    checkpoint: str | None,
    steps: int,
    size: int,
    timeout: float,
) -> None:
    """End-to-end smoke test against your real ComfyUI server.

    Submits a tiny known-good SDXL/SD1.5-compatible txt2img graph (1 step,
    64x64 by default) and verifies the round-trip works. If this passes,
    your max-comfy install can talk to your ComfyUI install — the rest is
    workflow-specific configuration.

    Run it once after `max-comfy init` + `max-comfy doctor` to confirm
    everything's wired up.
    """
    client = Client(host=cfg.comfyui_host, port=cfg.comfyui_port)
    if not client.health_check():
        raise click.ClickException(
            f"ComfyUI not reachable at {cfg.comfyui_host}:{cfg.comfyui_port}. "
            "Start it with `max-comfy server start` or run `cd <ComfyUI> && python main.py`."
        )

    if not checkpoint:
        if cfg.comfyui_path:
            ckpt_dir = Path(cfg.comfyui_path).expanduser() / "models" / "checkpoints"
            if ckpt_dir.is_dir():
                candidates = sorted(
                    list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.ckpt"))
                )
                if candidates:
                    checkpoint = candidates[0].name
                    click.echo(f"Auto-selected checkpoint: {checkpoint}")
        if not checkpoint:
            raise click.ClickException(
                "Pass --checkpoint <name>, or set comfyui_path in your config so I can pick one."
            )

    graph: dict[str, Any] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "max-comfy selftest", "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": size, "height": size, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "max-comfy-selftest"},
        },
    }

    # Static check first — if this fails, our own builder is broken, no point asking ComfyUI.
    from .workflows import validate_graph as _validate_graph

    issues = _validate_graph(graph)
    if issues:
        raise click.ClickException(
            "Selftest graph failed static validation (this is a max-comfy bug):\n  - "
            + "\n  - ".join(issues)
        )

    click.echo(f"Submitting smoke-test graph ({size}x{size}, {steps} step, ckpt={checkpoint})...")
    try:
        result = client.submit_and_wait(graph, timeout=timeout, poll_interval=cfg.poll_interval)
    except MaxComfyError as exc:
        click.echo(f"FAIL  {exc}", err=True)
        sys.exit(1)

    click.echo(f"OK    prompt {result.prompt_id} completed.")
    click.echo(f"      {len(result.outputs)} output(s):")
    for out in result.outputs:
        click.echo(f"        {out.filename}  (node {out.node_id}, type={out.type})")
    click.echo()
    click.echo("ComfyUI integration verified. You're good to go.")


@cli.command("install-comfyui")
@click.option(
    "--path",
    "install_path",
    type=click.Path(path_type=Path),
    default=Path.home() / "ComfyUI",
    show_default=True,
    help="Directory to clone ComfyUI into.",
)
@click.option(
    "--branch",
    default="master",
    show_default=True,
    help="Git branch/tag to check out.",
)
@click.option(
    "--update-config",
    is_flag=True,
    help="Update ./max-comfy.toml with comfyui_path = <install_path>.",
)
def install_comfyui(install_path: Path, branch: str, update_config: bool) -> None:
    """Clone ComfyUI to disk. (Does NOT install Python deps — see notes below.)

    This is opt-in and reversible: it just runs `git clone`. ComfyUI's actual
    Python dependencies (especially PyTorch) need to be installed by you so
    you pick the right build for your GPU (CUDA / ROCm / MPS / CPU).
    """
    if shutil.which("git") is None:
        raise click.ClickException("git not on PATH; install git first.")

    install_path = install_path.expanduser().resolve()
    if install_path.exists():
        if (install_path / "main.py").is_file():
            click.echo(f"ComfyUI already present at {install_path}")
        else:
            raise click.ClickException(
                f"{install_path} exists but doesn't look like a ComfyUI install. "
                "Remove it or pick a different --path."
            )
    else:
        click.echo(f"Cloning ComfyUI into {install_path}...")
        try:
            subprocess.run(
                [
                    "git", "clone",
                    "--depth", "1",
                    "--branch", branch,
                    "https://github.com/comfyanonymous/ComfyUI",
                    str(install_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise click.ClickException(f"git clone failed: {exc}") from None

    if update_config:
        cfg_path = Path("./max-comfy.toml")
        if cfg_path.is_file():
            text = cfg_path.read_text()
            new_line = f'comfyui_path = "{install_path}"'
            if "comfyui_path" in text:
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    stripped = line.strip().lstrip("#").strip()
                    if stripped.startswith("comfyui_path"):
                        lines[i] = new_line
                        break
                cfg_path.write_text("\n".join(lines) + "\n")
            else:
                cfg_path.write_text(text.rstrip() + "\n" + new_line + "\n")
            click.echo(f"Updated {cfg_path}")
        else:
            click.echo("No ./max-comfy.toml found — run `max-comfy init` first.")

    click.echo()
    click.echo("Next steps — install ComfyUI's Python deps for YOUR GPU:")
    click.echo()
    click.echo(f"  cd {install_path}")
    click.echo("  python -m venv venv && source venv/bin/activate")
    click.echo()
    click.echo("  # NVIDIA (CUDA 12.1):")
    click.echo(
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    click.echo()
    click.echo("  # AMD (ROCm 6.0):")
    click.echo(
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0"
    )
    click.echo()
    click.echo("  # CPU-only (slow):")
    click.echo("  pip install torch torchvision torchaudio")
    click.echo()
    click.echo("  pip install -r requirements.txt")
    click.echo()
    click.echo(f"Then run ComfyUI:  cd {install_path} && python main.py")
    click.echo("And verify with:    max-comfy doctor")


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main() -> None:
    try:
        cli(standalone_mode=True)
    except MaxComfyError as exc:
        # MediaError, WorkflowError, ClientError, etc. all inherit from MaxComfyError.
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
