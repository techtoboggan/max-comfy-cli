# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-05-02

First public release.

### Added

- **Workflow plugin system.** JSON templates with `{{var}}` placeholders (auto-typed
  for whole-string placeholders, with optional defaults like `{{seed:0}}`) plus
  Python `Workflow` subclasses registered via `@register`. Auto-discovery from
  `$MAX_COMFY_WORKFLOWS`, config-declared dirs, `--workflow-dir` flags,
  `~/.config/max-comfy/workflows/`, and `./workflows/`.
- **Built-in `video-chain` workflow.** Runs a per-segment workflow N times,
  threading each segment's last frame forward as the next segment's input
  image, then stitches with optional first-frame deduplication and crossfade.
  Optional post-processing: motion-compensated interpolation, Lanczos upscale,
  unsharp-mask sharpen.
- **Built-in `video-postprocess` workflow.** Pure-ffmpeg pipeline (no ComfyUI
  required) for stitch / interpolate / upscale / sharpen / transcode.
- **Built-in `reactor-face-swap` workflow.** Per-frame ReActor + CodeFormer face
  swap on a video, with resume support for already-processed frames.
- **`max_comfy.media` module.** First-class ffmpeg/ffprobe primitives: frame
  extraction, encoding from frames, video stitching with frame-drop dedup and
  xfade crossfade, motion-compensated interpolation, Lanczos upscaling,
  unsharp-mask sharpening, generic filter chains, encoder auto-detection,
  video probing.
- **`max-comfy media` subcommand group.** CLI surface for every media primitive:
  `info`, `extract-frames`, `extract-last-frame`, `encode`, `stitch`,
  `interpolate`, `upscale`, `sharpen`, `transcode`, `detect-encoder`.
- **Onboarding commands.**
  - `max-comfy init` — write a starter `max-comfy.toml`, auto-detect ComfyUI.
  - `max-comfy doctor` — diagnose the environment (ffmpeg, ComfyUI reachability,
    GPU info, model dirs, discovered workflows).
  - `max-comfy models list [--category X] [--json]` — inventory installed models.
  - `max-comfy install-comfyui [--path ...] [--update-config]` — opt-in
    `git clone` of ComfyUI plus printed `pip install` instructions for
    CUDA / ROCm / CPU.
- **Batch runner.** `max-comfy batch <jobs.json>` with atomic resume support
  (`.progress.json`), per-job workflow override, and merged params.
- **Workflow composition.** `RunContext.registry` lets any `run()` look up and
  invoke another workflow.
- **`Client.upload_input`.** Upload local files to ComfyUI's input directory
  via `/upload/image`, decoupling chain workflows from where ComfyUI's input
  dir lives.
- **`Server`.** Optional managed ComfyUI subprocess with start/stop/restart/
  context-manager API.
- **CLI commands.** `list`, `show`, `render`, `run`, `batch`, `validate`,
  `server status|start|stop`, `media`, `init`, `doctor`, `models`,
  `install-comfyui`.
- **Examples.** SDXL txt2img and img2img templates, programmatic SDXL+LoRA
  workflow, SVD segment workflow paired with `video-chain`, and jobs files
  demonstrating single-image, batch, chained-video, ReActor postprocess, and
  ffmpeg-only postprocess pipelines.
- **GitHub Actions.** CI matrix on Python 3.10/3.11/3.12 with ffmpeg installed
  for media integration tests; tag-driven PyPI release using trusted publishing.
- **Tests.** 62 passing tests covering templates, config, workflow registry,
  batch loader, runner progress, media argument construction, encoder
  detection, and built-in workflow orchestration with mocked clients.

[Unreleased]: https://github.com/techtoboggan/max-comfy-cli/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/techtoboggan/max-comfy-cli/releases/tag/v0.2.0
