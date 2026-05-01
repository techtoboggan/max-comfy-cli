# Examples

This directory ships:

* [`workflows/`](workflows) — sample workflow definitions you can copy/adapt.
  * `sdxl_txt2img.json` — basic text-to-image.
  * `sdxl_img2img.json` — image-to-image.
  * `sdxl_loras.py` — programmatic SDXL with a configurable LoRA stack.
  * `svd_segment.json` — Stable Video Diffusion image-to-video; pair with
    the built-in `video-chain` workflow.
* [`jobs/`](jobs) — sample batch files showing the JSON job format.
  * `single_image.json` / `batch_images.json` — basic SDXL batches.
  * `chained_video.json` — multi-segment video with chaining + interpolation +
    upscaling + sharpening.
  * `reactor_postprocess.json` — face-swap a finished video.
  * `postprocess_only.json` — pure ffmpeg post-processing (no ComfyUI).
* [`configs/`](configs) — annotated TOML config showing every option.

## Built-in workflows

Three workflows ship with the package and are always available — you don't
need to put them in a workflow directory:

| Name                  | Type        | Purpose                                                    |
| --------------------- | ----------- | ---------------------------------------------------------- |
| `video-chain`         | composition | Multi-segment video, last-frame chaining, stitch + post.   |
| `video-postprocess`   | ffmpeg-only | Stitch / interpolate / upscale / sharpen / transcode.      |
| `reactor-face-swap`   | per-frame   | ReActor + CodeFormer face swap on a video. Needs ComfyUI.  |

Run `max-comfy list` to confirm.

## Trying the examples

```bash
# 1. Start ComfyUI on port 8188 (in another shell).

# 2. List workflows (built-ins + the examples directory).
max-comfy --workflow-dir examples/workflows list

# 3. Inspect one.
max-comfy --workflow-dir examples/workflows show video-chain

# 4. Run a chained video.
max-comfy --workflow-dir examples/workflows batch examples/jobs/chained_video.json

# 5. Pure-ffmpeg post-processing (no ComfyUI server needed).
max-comfy run video-postprocess \
    -p input=/path/to/raw.webp \
    -p transcode=true \
    -p interpolate_to_fps=60 \
    -p upscale=2 \
    -p output_filename=smooth.mp4

# 6. Or use the media subcommands directly.
max-comfy media info /path/to/video.mp4
max-comfy media stitch a.mp4 b.mp4 c.mp4 -o joined.mp4 --drop-first-frame
max-comfy media interpolate raw.webp -o smooth.mp4 --fps 60
max-comfy media upscale raw.webp -o big.mp4 --scale 2
```

## Workflow file types

* **`*.json`** — a ComfyUI API-format workflow with `{{var}}` placeholders.
  Optional sidecar `*.toml` declares `description`, `required`, `[defaults]`.
* **`*.py`** — a Python module that registers `Workflow` subclasses via
  `@max_comfy.register`.

## Adapting a workflow from ComfyUI

1. Build the workflow in ComfyUI's UI.
2. Click **Save (API Format)** to download the JSON.
3. Open the JSON; replace any value you want to parameterise with a placeholder
   (`"text": "{{prompt}}"`, `"seed": "{{seed:0}}"`).
4. (Optional) Add a sidecar TOML with metadata.
5. Drop both into a workflow directory; run `max-comfy list`.
