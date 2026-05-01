# max-comfy-cli

A clean, generic CLI for selecting, parameterising, and batch-running ComfyUI
workflows — without ever opening the ComfyUI web UI. With first-class support
for **multi-segment video chaining**, **ffmpeg post-processing**, and
**ReActor face-swap**.

If you've built a workflow in ComfyUI and want to:

* run it 50 times with different prompts,
* chain it into a longer video by feeding each segment's last frame into the
  next one's input,
* upscale, sharpen, frame-interpolate, or stitch the results,
* face-swap a finished video with ReActor + CodeFormer,
* or hand the workflow to a teammate who doesn't want to deal with the node graph,

this is for you.

## What it does

* **Drives ComfyUI over HTTP.** Submits prompt graphs, polls for completion,
  downloads outputs. No ComfyUI Python imports at runtime — just a server you
  point at.
* **Workflows are pluggable.** Drop a JSON template (with `{{var}}`
  placeholders) or a Python class into a workflow directory and it shows up in
  the CLI.
* **Built-in `video-chain` workflow** runs a per-segment workflow N times,
  threading each segment's last frame forward as the next segment's input
  image, then stitches everything with optional first-frame deduplication and
  crossfade.
* **First-class ffmpeg.** Frame extraction, motion-compensated interpolation,
  Lanczos upscaling, unsharp-mask sharpening, encoder auto-detection. All
  exposed as `max-comfy media` subcommands and as a Python API in
  `max_comfy.media`.
* **Built-in `reactor-face-swap` workflow** for per-frame face replacement on
  a finished video using the ReActor and (optional) CodeFormer custom nodes.
* **Batch runner with resume.** Run hundreds of jobs from a single JSON file;
  resume picks up where it left off if interrupted.
* **No model assumptions.** SDXL, Flux, Wan, LTX, SVD, Hunyuan, AnimateDiff —
  anything you can build in ComfyUI works. Examples ship for SDXL and SVD; the
  core is model-agnostic.

## Installation

Once published to PyPI:

```bash
pip install max-comfy-cli
```

Until then, install directly from GitHub:

```bash
pip install git+https://github.com/techtoboggan/max-comfy-cli
```

Or clone the repo (recommended if you want the example workflows / jobs
shipped under `examples/`):

```bash
git clone https://github.com/techtoboggan/max-comfy-cli
cd max-comfy-cli
pip install -e ".[dev]"
```

You also need:

* **`ffmpeg`** and **`ffprobe`** on your `$PATH`. Any standard ffmpeg install
  works (`apt install ffmpeg` / `brew install ffmpeg` / WinGet / the official
  static builds). Required for any `media` operation and for the
  chain/postprocess/ReActor workflows.
* A working **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** install
  with the models / custom nodes your workflows need. `max-comfy` talks to it
  over HTTP. If you don't have one yet:

  ```bash
  max-comfy install-comfyui --update-config
  ```

  This clones ComfyUI to `~/ComfyUI` (override with `--path`) and prints the
  exact `pip install` for your GPU (CUDA / ROCm / CPU). It does **not** install
  Python deps automatically — you pick the right PyTorch build for your
  hardware. Reversible at any time with `rm -rf ~/ComfyUI`.

## Quickstart (5 minutes)

```bash
# 1. Install the tool (one of the three options above).
pip install git+https://github.com/techtoboggan/max-comfy-cli

# 2. Set up a starter config in your working directory (auto-detects
#    ComfyUI at ~/ComfyUI / ./ComfyUI).
max-comfy init

# 3. (Optional) If you don't have ComfyUI yet:
max-comfy install-comfyui --update-config
# ... follow the printed instructions to install PyTorch for your GPU.

# 4. Verify everything is wired up.
max-comfy doctor

# 5. Start ComfyUI (in another shell):
cd ~/ComfyUI && python main.py
# (Or, if comfyui_path is in your config: `max-comfy server start`.)

# 6. List workflows.
max-comfy list

# 7. Run something — a single image:
max-comfy run sdxl_txt2img \
    -p prompt="a serene sunset over mountains" \
    -p checkpoint="sd_xl_base_1.0.safetensors" \
    -p seed=42

# 8. Or a chained video with full post-processing:
max-comfy batch examples/jobs/chained_video.json

# 9. Or pure-ffmpeg post-processing (no ComfyUI needed):
max-comfy media interpolate raw.webp -o smooth.mp4 --fps 60
max-comfy media upscale  smooth.mp4 -o big.mp4    --scale 2
max-comfy media stitch a.mp4 b.mp4 c.mp4 -o joined.mp4 --drop-first-frame
```

## Where to put your models

ComfyUI looks for models in subdirectories of `<ComfyUI>/models/`. Drop
files into the matching subdirectory and ComfyUI will pick them up on its
next startup (or click *Refresh* in the web UI).

| Subdirectory       | What goes here                                                          |
| ------------------ | ----------------------------------------------------------------------- |
| `checkpoints/`     | Full models — SDXL, SD 1.5, Flux, Hunyuan, SVD, Wan, LTX checkpoints    |
| `loras/`           | LoRA adapters (`.safetensors`)                                          |
| `vae/`             | Standalone VAEs                                                         |
| `clip/`            | Text encoders (CLIP, T5, Gemma) — Flux / Wan / LTX need these           |
| `clip_vision/`     | CLIP vision encoders (SVD, IPAdapter)                                   |
| `controlnet/`      | ControlNet models                                                       |
| `upscale_models/`  | Upscalers (RealESRGAN, etc.)                                            |
| `embeddings/`      | Textual inversion embeddings                                            |
| `ipadapter/`       | IPAdapter weights                                                       |
| `unet/`            | Standalone UNETs (some video models)                                    |
| `diffusion_models/`| Diffusion models (Flux full)                                            |

To see what you currently have installed:

```bash
max-comfy models list                   # everything
max-comfy models list --category loras  # one category
max-comfy models list --json            # machine-readable
```

Common sources:

* **Civitai** ([civitai.com](https://civitai.com)) — community SDXL/SD1.5
  checkpoints, LoRAs, embeddings.
* **Hugging Face** ([huggingface.co](https://huggingface.co)) — official
  model releases (Stability AI, Black Forest Labs, etc.). Look for repos with
  a `model_index.json` or `*.safetensors` in their root.
* **ComfyUI Manager** custom node — installs models + custom nodes from inside
  ComfyUI's UI.

## Diagnosing problems

`max-comfy doctor` is the first thing to try when something isn't working. It
checks:

* `ffmpeg` / `ffprobe` are on PATH and have a working video encoder.
* The configured ComfyUI server is reachable on `host:port`.
* If `comfyui_path` is set, the install looks valid and `models/` exists.
* What workflows are discoverable, including the three built-ins.

Example:

```
$ max-comfy doctor
Checking environment...

  [OK]   ffmpeg  -> /usr/bin/ffmpeg
  [OK]   ffprobe -> /usr/bin/ffprobe
  [OK]   ffmpeg encoder available: libsvtav1

  [OK]   ComfyUI reachable at 127.0.0.1:8188
         GPU: NVIDIA GeForce RTX 4090  (22.4 GB free / 24.0 GB)

  [OK]   ComfyUI install at /home/tristan/ComfyUI
  [OK]   models/ has 8 categories, 47 files
         Run `max-comfy models list` for the full inventory.

  Workflows discovered: 6
    - reactor-face-swap
    - sdxl_txt2img
    - svd_segment
    - video-chain
    - video-postprocess
    - ...

All checks passed.
```

## Built-in workflows

Three workflows ship with the package and are always available — no workflow
directory required:

| Name                  | Type        | Purpose                                                    |
| --------------------- | ----------- | ---------------------------------------------------------- |
| `video-chain`         | composition | Multi-segment video, last-frame chaining, stitch + post.   |
| `video-postprocess`   | ffmpeg-only | Stitch / interpolate / upscale / sharpen / transcode.      |
| `reactor-face-swap`   | per-frame   | ReActor + CodeFormer face swap on a video.                 |

Run `max-comfy show <name>` for the full parameter list.

### `video-chain` — multi-segment video with chaining

This is the workflow most people will use. It composes any per-segment
"image-to-video" workflow into a longer chained video:

```jsonc
{
  "workflow": "video-chain",
  "output_dir": "./output/chained",
  "defaults": {
    "segment_workflow": "svd_segment",     // the per-segment workflow's name
    "checkpoint": "svd_xt.safetensors",    // forwarded to each segment
    "fps": 24,
    "frames": 25,                          // per segment
    "stitch": true,
    "drop_first_frame": true,              // dedupe segment-N's frame 0
    "crossfade_frames": 0,                 // optional, requires fps
    "interpolate_to_fps": 60,              // post: smooth motion via minterpolate
    "upscale": 2.0,                        // post: 2x Lanczos upscale
    "sharpen": 0.6,                        // post: unsharp mask
    "output_filename": "chained_final.mp4"
  },
  "jobs": [
    {
      "id": "demo_chain",
      "initial_image": "/path/to/start_frame.png",
      "segments": [
        { "seed": 42 },
        { "seed": 100 },
        { "seed": 256 }
      ]
    }
  ]
}
```

How it works:

1. `initial_image` (if provided) is uploaded to ComfyUI as the first segment's
   `input_image`. Otherwise, the first segment runs without one
   (text-to-video, if your segment workflow supports it).
2. For each segment, the per-segment workflow runs and produces a video file.
3. After each segment, ffmpeg extracts the last frame and uploads it as the
   next segment's `input_image`. Per-segment overrides (e.g. `seed`) are
   merged on top of the common params.
4. Once all segments are done, they're stitched (with optional first-frame
   drop and crossfade), then optionally interpolated, upscaled, and sharpened.

The contract a segment workflow needs to satisfy:

* Accept `input_image` (a string filename of an uploaded ComfyUI input) as a
  parameter when run as part of a chain.
* Produce a video file (`.webp`, `.mp4`, or `.gif`) as one of its outputs.
  `video-chain` picks the first match by extension.

`examples/workflows/svd_segment.json` is a working example using vanilla SVD
nodes.

### `video-postprocess` — pure ffmpeg pipeline

Use when you've already got a video and want to stitch / interpolate / upscale
/ sharpen / transcode it. No ComfyUI server is touched:

```bash
max-comfy run video-postprocess \
    -p input=raw.webp \
    -p transcode=true \
    -p interpolate_to_fps=60 \
    -p upscale=2 \
    -p sharpen=1.0 \
    -p output_filename=final.mp4
```

### `reactor-face-swap` — per-frame face swap on a video

Requires the **ReActor** custom node in your ComfyUI install
(`ComfyUI/custom_nodes/comfyui-reactor-node`). With `restore_face=true`
(default) it also runs CodeFormer for face restoration:

```bash
max-comfy run reactor-face-swap \
    -p input_video=source.mp4 \
    -p reference_face=face.png \
    -p codeformer_weight=0.5 \
    -p output_filename=swapped.mp4
```

Frames are extracted, each one submitted to ComfyUI in turn (the loop has
resume support — already-processed frames are skipped on rerun), then re-encoded.

## Defining a workflow

### Path 1 — JSON template (no Python)

In ComfyUI, build a workflow and click **Save (API Format)**. You'll get a JSON
file like:

```json
{
  "5": {
    "class_type": "KSampler",
    "inputs": { "seed": 1234, "steps": 30, "cfg": 7.0, "model": ["1", 0], ... }
  }
}
```

Replace any value you want to parameterise with `{{var}}`. Defaults inline as
`{{var:default}}`:

```json
"seed": "{{seed:0}}",
"steps": "{{steps:30}}"
```

When the entire string is just a placeholder (`"{{seed}}"`), the parameter's
original Python type is preserved — `42` stays an int, `6.5` stays a float,
`[1, 2]` stays a list. Strings with embedded placeholders interpolate as text.

Drop the file into one of the search directories (see below) and run
`max-comfy list` to confirm.

Optional: place a sidecar `*.toml` next to it for metadata:

```toml
description = "SDXL text-to-image"
required = ["prompt", "checkpoint"]

[defaults]
steps = 30
cfg = 6.5
negative_prompt = "blurry, low quality"
```

### Path 2 — Python plugin

Subclass `Workflow` for cases the JSON template can't express — conditional
logic, loops, multi-step orchestration, post-processing hooks.

```python
from max_comfy import Workflow, register

@register
class MyWorkflow(Workflow):
    name = "my-flow"
    description = "Does the thing"
    required = ("prompt",)
    defaults = {"steps": 30, "cfg": 7.0}

    def build(self, params):
        # Return a single ComfyUI graph dict.
        return {...}
```

For multi-step flows (e.g. generate, then enhance, then decode) override
`run()`. The context gives you a Client, the merged params, an output
directory, and the workflow registry (so you can compose other workflows):

```python
from max_comfy import RunResult, Workflow, register, media

@register
class MultiStep(Workflow):
    name = "multi-step"

    def run(self, ctx):
        # ctx.client    -> max_comfy.Client (HTTP)
        # ctx.config    -> Config
        # ctx.params    -> merged params (defaults + job)
        # ctx.output_dir-> Path
        # ctx.registry  -> WorkflowRegistry, for sub-workflow lookup
        graph_a = build_first_graph(ctx.params)
        result_a = ctx.client.submit_and_wait(graph_a, download_dir=ctx.output_dir)

        # Compose a built-in workflow:
        post = ctx.registry.get("video-postprocess")
        post_ctx = RunContext(
            client=ctx.client, config=ctx.config, registry=ctx.registry,
            params={"input": str(result_a.downloaded[0]), "interpolate_to_fps": 60},
            output_dir=ctx.output_dir,
        )
        post.run(post_ctx)

        # Use ffmpeg directly:
        media.upscale(result_a.downloaded[0], ctx.output_dir / "big.mp4", scale=2.0)
        return RunResult(files=[ctx.output_dir / "big.mp4"])
```

Drop the file into a workflow directory — auto-discovery imports it and the
`@register` decorator wires it into the global registry.

### Workflow search paths

Workflows are loaded from (in order, all combined):

1. `$MAX_COMFY_WORKFLOWS` (env var)
2. Any `workflow_dirs` in your config TOML
3. `--workflow-dir` flags on the command line (repeatable)
4. `~/.config/max-comfy/workflows/`
5. `./workflows/` in the working directory

`*.json` files become `TemplateWorkflow`s. `*.py` files are imported and any
`@register`-decorated subclasses are added.

## Batch jobs

A jobs file describes one or more workflow runs:

```json
{
  "workflow": "sdxl_txt2img",
  "output_dir": "./output/run-1",
  "defaults": {
    "checkpoint": "sd_xl_base_1.0.safetensors",
    "steps": 30
  },
  "jobs": [
    {"id": "sunset",  "prompt": "a sunset over mountains", "seed": 42},
    {"id": "forest",  "prompt": "a misty forest",          "seed": 100, "steps": 50},
    {"id": "swap-wf", "workflow": "sdxl-with-loras",       "prompt": "..."}
  ]
}
```

* `defaults` are merged into every job's params (job values win on conflict).
* Per-job `workflow` overrides the top-level `workflow`.
* `id` is optional; auto-numbered if omitted.

Run it:

```bash
max-comfy batch jobs.json
max-comfy batch jobs.json --resume   # skip jobs already in .progress.json
```

`max-comfy validate jobs.json` checks the file (referenced workflows exist,
required params are satisfied) without running anything.

## `media` — ffmpeg utilities

Available without any ComfyUI server:

```bash
max-comfy media info              video.mp4
max-comfy media extract-frames    video.mp4 -o frames/
max-comfy media extract-last-frame video.mp4 -o last.png
max-comfy media encode            frames/  -o video.mp4 --fps 24
max-comfy media stitch            a.mp4 b.mp4 c.mp4 -o joined.mp4 --drop-first-frame --crossfade-frames 8 --fps 24
max-comfy media interpolate       in.mp4 -o out.mp4 --fps 60 --method minterpolate
max-comfy media upscale           in.mp4 -o big.mp4 --scale 2 --algorithm lanczos
max-comfy media sharpen           in.mp4 -o sharp.mp4 --amount 1.0
max-comfy media transcode         in.webp -o in.mp4
max-comfy media detect-encoder    # prints "libx264" (or whatever's available)
```

Programmatically:

```python
from max_comfy import media

info = media.get_video_info("clip.mp4")  # VideoInfo(width, height, fps, ...)
media.extract_last_frame("clip.mp4", "last.png")
media.stitch_videos(["a.mp4", "b.mp4"], "joined.mp4", drop_first_frame=True)
media.interpolate("in.mp4", "smooth.mp4", target_fps=60)
media.upscale("in.mp4", "big.mp4", scale=2.0)
media.sharpen("in.mp4", "sharp.mp4", amount=1.0)
media.apply_filter_chain("in.mp4", "out.mp4", ["unsharp=5:5:1.0", "scale=iw*2:ih*2"])
```

## Configuration

`max-comfy` reads TOML config from, in order:

1. `~/.config/max-comfy/config.toml`
2. `./max-comfy.toml`
3. `--config /path/to/file.toml`

See [`examples/configs/example.toml`](examples/configs/example.toml) for every
option, all annotated.

CLI flags (`--port`, `--host`, `--output-dir`, `--workflow-dir`) override
config-file values.

## Programmatic use

The same primitives are exported from the package root:

```python
from max_comfy import Client, Config, JobRunner, TemplateWorkflow, Workflow, register, media

cfg = Config(comfyui_port=8188, output_dir=Path("./out"))
runner = JobRunner(cfg)

# Run a one-off
result = runner.run_one(
    workflow="video-chain",
    params={
        "segment_workflow": "svd_segment",
        "checkpoint": "svd_xt.safetensors",
        "initial_image": "/path/to/start.png",
        "segments": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
        "interpolate_to_fps": 60,
        "upscale": 2.0,
    },
)

# Or use the lower-level Client directly
with Client(port=8188) as client:
    client.wait_ready()
    uploaded = client.upload_input("/path/to/face.png")
    result = client.submit_and_wait(graph, download_dir=Path("./out"))
```

## CLI reference

```
# Onboarding
max-comfy init                          # write a starter max-comfy.toml
max-comfy install-comfyui [--path ...]  # opt-in clone of ComfyUI to disk
max-comfy doctor                        # check ffmpeg, ComfyUI, workflows
max-comfy models list [--category X]    # list installed checkpoints/loras/etc.

# Workflow discovery + execution
max-comfy list                          # discovered workflows
max-comfy show <workflow>               # placeholders, defaults, source
max-comfy render <workflow> -p k=v ...  # render template to stdout (debug)
max-comfy run <workflow> -p k=v ...     # run once
max-comfy batch <jobs.json> [--resume]  # batch run
max-comfy validate <jobs.json>          # static-check a jobs file

# ComfyUI server lifecycle
max-comfy server status                 # ping the configured ComfyUI
max-comfy server start                  # spawn ComfyUI (needs comfyui_path)
max-comfy server stop                   # ask ComfyUI to free memory

# Media utilities (no ComfyUI needed)
max-comfy media <subcommand> ...        # info/extract/encode/stitch/etc.
```

Global options apply to every subcommand:

```
-c, --config <path.toml>
    --host <host>
    --port <int>
-o, --output-dir <path>
    --workflow-dir <path>     (repeatable)
-v, --verbose                 (repeat for debug)
-V, --version
```

## How does this differ from the ComfyUI web UI?

| Need                                           | Web UI         | max-comfy        |
| ---------------------------------------------- | -------------- | ---------------- |
| Build a graph visually                         | yes            | use ComfyUI then export API JSON |
| Run a saved workflow                           | per-click      | scriptable, batch-able |
| Run a workflow 100× with different prompts     | tedious        | one jobs file    |
| Multi-segment video with last-frame chaining   | manual rebuild | `video-chain`    |
| Stitch + interpolate + upscale + sharpen       | external tools | `media` / `video-postprocess` |
| ReActor face-swap on a finished video          | per-frame UI   | `reactor-face-swap` |
| Resume after a crash mid-batch                 | start over     | `--resume`       |
| Schedule via cron / CI                         | no             | yes              |
| Validate a config without running it           | no             | `validate`       |
| Hand off to a teammate who doesn't know ComfyUI| UI required    | CLI only         |

ComfyUI builds the graph; `max-comfy` runs it.

## License

MIT.
