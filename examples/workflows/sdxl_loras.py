"""Example: Python-based workflow plugin.

Demonstrates how to subclass :class:`max_comfy.Workflow` to construct a graph
programmatically. This one chains an arbitrary number of LoRA loaders onto an
SDXL checkpoint — something awkward to express in a static JSON template.

Drop this file into a workflow directory (e.g. ``./workflows/`` or
``~/.config/max-comfy/workflows/``) and it will be picked up automatically.

Usage:
    max-comfy run sdxl-with-loras \\
        -p prompt="cyberpunk samurai" \\
        -p checkpoint=sd_xl_base_1.0.safetensors \\
        -p 'loras=[{"name": "style.safetensors", "weight": 0.8}]'
"""

from __future__ import annotations

from typing import Any

from max_comfy import Workflow, register


@register
class SDXLWithLoras(Workflow):
    name = "sdxl-with-loras"
    description = "SDXL txt2img with a configurable LoRA stack"
    required = ("prompt", "checkpoint")
    defaults = {
        "negative_prompt": "blurry, low quality, distorted",
        "loras": [],  # list of {name: str, weight: float}
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg": 6.5,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "seed": 0,
        "output_prefix": "sdxl_loras",
    }

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        graph: dict[str, Any] = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": params["checkpoint"]},
            },
        }

        # Chain LoRAs: each consumes the previous model+clip slot.
        last_model: tuple[str, int] = ("1", 0)
        last_clip: tuple[str, int] = ("1", 1)
        for i, lora in enumerate(params.get("loras") or []):
            node_id = f"lora_{i}"
            graph[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora["name"],
                    "strength_model": float(lora.get("weight", 1.0)),
                    "strength_clip": float(lora.get("weight", 1.0)),
                    "model": list(last_model),
                    "clip": list(last_clip),
                },
            }
            last_model = (node_id, 0)
            last_clip = (node_id, 1)

        graph.update({
            "pos": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": params["prompt"], "clip": list(last_clip)},
            },
            "neg": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": params["negative_prompt"], "clip": list(last_clip)},
            },
            "latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": params["width"],
                    "height": params["height"],
                    "batch_size": 1,
                },
            },
            "sample": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": params["seed"],
                    "steps": params["steps"],
                    "cfg": params["cfg"],
                    "sampler_name": params["sampler"],
                    "scheduler": params["scheduler"],
                    "denoise": 1.0,
                    "model": list(last_model),
                    "positive": ["pos", 0],
                    "negative": ["neg", 0],
                    "latent_image": ["latent", 0],
                },
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["sample", 0], "vae": ["1", 2]},
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["decode", 0],
                    "filename_prefix": params["output_prefix"],
                },
            },
        })
        return graph
