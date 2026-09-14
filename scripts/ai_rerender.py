"""
NavBench3D - Step 3: AI Re-rendering Pipeline.

Converts Blender-rendered synthetic images into photorealistic images.
Supports multiple backends ranked by recommendation:

  1. flux2       — FLUX.2-dev: 32B, Mistral-3 VLM + Rectified Flow, native editing, up to 4MP
  2. flux2-klein — FLUX.2-klein: 4B/9B distilled, 4-step inference, sub-second
  3. flux-kontext— FLUX.1-Kontext-dev: 12B instruction editing, good structure preservation
  4. qwen       — Qwen-Image: dual VL+VAE encoding, Apache 2.0
  5. flux-cn    — FLUX.1-dev + ControlNet Union Pro: depth+canny dual control
  6. sd35       — Stable Diffusion 3.5 Large: solid img2img
  7. sdxl       — SDXL + ControlNet: legacy fallback

Usage:
    python scripts/ai_rerender.py --renders-dir data/renders --backend flux2
    python scripts/ai_rerender.py --renders-dir data/renders --backend flux2-klein
    python scripts/ai_rerender.py --renders-dir data/renders --backend flux-kontext
    python scripts/ai_rerender.py --renders-dir data/renders --backend all  # run all backends for comparison
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
from PIL import Image


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_device_and_dtype():
    """Detect GPU and choose optimal dtype. A100 80GB → bf16 on GPU, no offload."""
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
        device = "cuda"
        dtype = torch.bfloat16  # A100 natively supports bf16
        need_offload = vram_gb < 30  # Only offload if VRAM < 30GB
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.0f}GB) — dtype=bf16, offload={'yes' if need_offload else 'no'}")
    else:
        device = "cpu"
        dtype = torch.float32
        need_offload = False
    return device, dtype, need_offload


# ============================================================
# Backend 1: FLUX.2-dev (32B, Native Editing + Generation)
# ============================================================

class Flux2Backend:
    """
    FLUX.2-dev: 32B parameter model combining Mistral-3 24B VLM encoder
    + Rectified Flow Transformer. Supports native generation AND editing
    in one unified model — no ControlNet or separate edit model needed.

    Key advantages over FLUX.1:
      - 32B params (vs 12B) → significantly better quality
      - Native image editing built-in (no Kontext adapter needed)
      - Up to 4MP output (2048×2048) for high-resolution re-rendering
      - Mistral-3 VLM provides deep semantic understanding of input image

    Requires ~40GB VRAM in bf16. A100 80GB handles this comfortably.
    """

    def __init__(self, config: dict):
        import torch
        from diffusers import Flux2Pipeline

        ai_cfg = config["ai_render"]
        model_id = ai_cfg.get("flux2_model", "black-forest-labs/FLUX.2-dev")

        device, dtype, need_offload = _get_device_and_dtype()

        print(f"  Loading FLUX.2-dev: {model_id}")
        self.pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        # FLUX.2-dev is ~56B params (~112GB bf16), always use CPU offload on 80GB GPU
        self.pipe.enable_model_cpu_offload()
        print("  Using model CPU offload (model too large for 80GB VRAM)")

        self.config = config
        self._torch = torch

    def rerender(self, rgb_path: str, depth_path: str,
                 normal_path: Optional[str] = None) -> Image.Image:
        ai_cfg = self.config["ai_render"]

        rgb_image = Image.open(rgb_path).convert("RGB")
        width, height = rgb_image.size

        # FLUX.2 supports up to 4MP — use native resolution aligned to 16px
        target_w = min((width // 16) * 16 or 1024, 2048)
        target_h = min((height // 16) * 16 or 1024, 2048)
        rgb_resized = rgb_image.resize((target_w, target_h), Image.LANCZOS)

        prompt = ai_cfg.get(
            "flux2_prompt",
            "Transform this synthetic 3D render into a photorealistic photograph. "
            "Keep the exact same room layout, furniture positions, and camera angle. "
            "Add realistic lighting, material textures, shadows, and natural imperfections. "
            "Make it look like a real interior photograph taken with a DSLR camera."
        )

        result = self.pipe(
            image=rgb_resized,
            prompt=prompt,
            guidance_scale=ai_cfg.get("flux2_guidance_scale", 3.5),
            num_inference_steps=ai_cfg.get("flux2_num_steps", 28),
            generator=self._torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        if result.size != (width, height):
            result = result.resize((width, height), Image.LANCZOS)
        return result

    @property
    def name(self):
        return "FLUX.2-dev (32B Native Editing)"


# ============================================================
# Backend 2: FLUX.2-klein (Distilled, Fast Inference)
# ============================================================

class Flux2KleinBackend:
    """
    FLUX.2-klein: Distilled variants of FLUX.2 for fast batch processing.
      - 9B version: ~29GB VRAM, 4-step inference
      - 4B version: Apache 2.0, sub-second generation, ~16GB VRAM

    Use this for rapid batch re-rendering when FLUX.2-dev is too slow.
    Quality is lower than FLUX.2-dev but still superior to FLUX.1/SDXL.
    """

    def __init__(self, config: dict):
        import torch
        from diffusers import FluxImg2ImgPipeline

        ai_cfg = config["ai_render"]
        model_id = ai_cfg.get("flux2_klein_model", "black-forest-labs/FLUX.2-klein-8B")

        device, dtype, need_offload = _get_device_and_dtype()

        print(f"  Loading FLUX.2-klein: {model_id}")
        self.pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if need_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        self.config = config
        self._torch = torch

    def rerender(self, rgb_path: str, depth_path: str,
                 normal_path: Optional[str] = None) -> Image.Image:
        ai_cfg = self.config["ai_render"]

        rgb_image = Image.open(rgb_path).convert("RGB")
        width, height = rgb_image.size

        target_w = min((width // 16) * 16 or 1024, 2048)
        target_h = min((height // 16) * 16 or 1024, 2048)
        rgb_resized = rgb_image.resize((target_w, target_h), Image.LANCZOS)

        prompt = ai_cfg.get(
            "flux2_klein_prompt",
            "Transform this synthetic 3D render into a photorealistic photograph. "
            "Keep the exact same room layout, furniture positions, and camera angle. "
            "Add realistic lighting, material textures, shadows, and natural imperfections."
        )

        # Klein uses fewer steps (4-step distilled)
        result = self.pipe(
            image=rgb_resized,
            prompt=prompt,
            strength=ai_cfg.get("flux2_klein_strength", 0.55),
            guidance_scale=ai_cfg.get("flux2_klein_guidance_scale", 3.5),
            num_inference_steps=ai_cfg.get("flux2_klein_num_steps", 4),
            generator=self._torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        if result.size != (width, height):
            result = result.resize((width, height), Image.LANCZOS)
        return result

    @property
    def name(self):
        return "FLUX.2-klein (Distilled Fast)"


# ============================================================
# Backend 3: FLUX.1-Kontext-dev (Instruction-based editing)
# ============================================================

class FluxKontextBackend:
    """
    FLUX.1-Kontext-dev: 12B instruction-based image editing model.
    The key insight: this is an EDITING model, not a generation model.
    It takes the synthetic render as input and edits it to be photorealistic
    while inherently preserving the original structure — no ControlNet needed.

    Best for: structure preservation + photorealism. The edit instruction
    tells it to "make photorealistic" rather than regenerating from scratch.
    """

    def __init__(self, config: dict):
        import torch
        from diffusers import FluxKontextPipeline

        ai_cfg = config["ai_render"]
        model_id = ai_cfg.get("kontext_model", "black-forest-labs/FLUX.1-Kontext-dev")

        device, dtype, need_offload = _get_device_and_dtype()

        print(f"  Loading FLUX Kontext: {model_id}")
        self.pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if need_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        self.config = config
        self._torch = torch

    def rerender(self, rgb_path: str, depth_path: str,
                 normal_path: Optional[str] = None) -> Image.Image:
        ai_cfg = self.config["ai_render"]

        rgb_image = Image.open(rgb_path).convert("RGB")
        width, height = rgb_image.size

        # Kontext works best at 1024x1024 or multiples of 16
        target_w = min((width // 16) * 16 or 1024, 1536)
        target_h = min((height // 16) * 16 or 1024, 1536)
        rgb_resized = rgb_image.resize((target_w, target_h), Image.LANCZOS)

        # Instruction-based editing: tell it to make photorealistic
        edit_prompt = ai_cfg.get(
            "kontext_prompt",
            "Transform this synthetic 3D render into a photorealistic photograph. "
            "Keep the exact same room layout, furniture positions, and camera angle. "
            "Add realistic lighting, material textures, shadows, and natural imperfections. "
            "Make it look like a real interior photograph taken with a DSLR camera."
        )

        result = self.pipe(
            image=rgb_resized,
            prompt=edit_prompt,
            guidance_scale=ai_cfg.get("kontext_guidance_scale", 2.5),
            num_inference_steps=ai_cfg.get("kontext_num_steps", 28),
            generator=self._torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        if result.size != (width, height):
            result = result.resize((width, height), Image.LANCZOS)
        return result

    @property
    def name(self):
        return "FLUX.1-Kontext-dev (Instruction Edit)"


# ============================================================
# Backend 4: Qwen-Image (Dual VL+VAE encoding)
# ============================================================

class QwenImageBackend:
    """
    Qwen-Image: Alibaba's latest (2025.08) image generation model.
    Architecture: Diffusion Transformer with dual encoding:
      - Qwen2.5-VL encoder for semantic understanding of input image
      - VAE encoder for pixel-level reconstruction fidelity
    This dual encoding theoretically gives both high-level structure
    understanding AND pixel-level faithfulness.

    Apache 2.0 license — fully open for commercial use.
    """

    def __init__(self, config: dict):
        import torch
        from diffusers import DiffusionPipeline

        ai_cfg = config["ai_render"]
        model_id = ai_cfg.get("qwen_image_model", "Qwen/Qwen-Image")

        device, dtype, need_offload = _get_device_and_dtype()

        print(f"  Loading Qwen-Image: {model_id}")
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if need_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        self.config = config
        self._torch = torch

    def rerender(self, rgb_path: str, depth_path: str,
                 normal_path: Optional[str] = None) -> Image.Image:
        ai_cfg = self.config["ai_render"]

        rgb_image = Image.open(rgb_path).convert("RGB")
        width, height = rgb_image.size

        # Qwen-Image supports diverse aspect ratios natively
        target_w = min((width // 16) * 16 or 1024, 1664)
        target_h = min((height // 16) * 16 or 1024, 1664)
        rgb_resized = rgb_image.resize((target_w, target_h), Image.LANCZOS)

        edit_prompt = ai_cfg.get(
            "qwen_prompt",
            "Convert this synthetic interior render to a photorealistic photograph. "
            "Maintain the exact same room layout and furniture placement. "
            "Apply realistic lighting, material textures, and natural shadows."
        )

        # Qwen-Image supports img2img via input image
        result = self.pipe(
            prompt=edit_prompt,
            image=rgb_resized,
            num_inference_steps=ai_cfg.get("qwen_num_steps", 50),
            true_cfg_scale=ai_cfg.get("qwen_cfg_scale", 4.0),
            generator=self._torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        if result.size != (width, height):
            result = result.resize((width, height), Image.LANCZOS)
        return result

    @property
    def name(self):
        return "Qwen-Image (Dual VL+VAE Encoding)"


# ============================================================
# Backend 5: FLUX.1-dev + ControlNet Union Pro
# ============================================================

class FluxControlNetBackend:
    """
    FLUX.1-dev with ControlNet Union Pro: depth+canny dual conditioning.
    Most proven approach for geometric fidelity via explicit spatial control.
    """

    def __init__(self, config: dict):
        import torch
        from diffusers import FluxControlNetPipeline, FluxControlNetModel
        from diffusers.models import FluxMultiControlNetModel

        ai_cfg = config["ai_render"]
        base_model = ai_cfg.get("flux_model", "black-forest-labs/FLUX.1-dev")
        cn_model = ai_cfg.get("flux_controlnet", "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro")

        device, dtype, need_offload = _get_device_and_dtype()

        print(f"  Loading FLUX ControlNet: {cn_model}")
        controlnet = FluxControlNetModel.from_pretrained(cn_model, torch_dtype=dtype)
        controlnet = FluxMultiControlNetModel([controlnet])

        print(f"  Loading FLUX base: {base_model}")
        self.pipe = FluxControlNetPipeline.from_pretrained(
            base_model, controlnet=controlnet, torch_dtype=dtype,
        )
        if need_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        self.config = config
        self._torch = torch

    def _extract_canny(self, image: Image.Image) -> Image.Image:
        import cv2
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr
        edges = cv2.Canny(gray, 50, 150)
        return Image.fromarray(edges).convert("RGB")

    def rerender(self, rgb_path: str, depth_path: str,
                 normal_path: Optional[str] = None) -> Image.Image:
        ai_cfg = self.config["ai_render"]

        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path).convert("RGB")
        width, height = rgb_image.size

        target_w = min((width // 16) * 16 or 1024, 1536)
        target_h = min((height // 16) * 16 or 1024, 1536)
        target_size = (target_w, target_h)

        depth_resized = depth_image.resize(target_size, Image.LANCZOS)
        canny_image = self._extract_canny(rgb_image).resize(target_size, Image.LANCZOS)

        scales = ai_cfg.get("flux_conditioning_scales", [0.5, 0.4])

        result = self.pipe(
            prompt=ai_cfg["prompt"],
            control_image=[depth_resized, canny_image],
            control_mode=[2, 0],
            controlnet_conditioning_scale=scales,
            num_inference_steps=ai_cfg.get("flux_num_steps", 24),
            guidance_scale=ai_cfg.get("flux_guidance_scale", 3.5),
            height=target_h, width=target_w,
            generator=self._torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        if result.size != (width, height):
            result = result.resize((width, height), Image.LANCZOS)
        return result

    @property
    def name(self):
        return "FLUX.1-dev + ControlNet Union Pro"


# ============================================================
# Backend 5b: FLUX.1-dev + ControlNet + Img2Img (Triple Constraint)
# ============================================================

class FluxCNImg2ImgBackend:
    """
    FLUX.1-dev + ControlNet Union Pro + Img2Img: TRIPLE constraint approach.

    Three layers of structural preservation:
      1. Img2Img: original image as pixel-level starting point (strength controls change %)
      2. ControlNet depth: 3D geometry hard constraint from depth map
      3. ControlNet canny: edge/contour hard constraint from original render

    This is the optimal backend for photorealistic re-rendering with maximum
    structural fidelity. FLUX.1-dev (12B) fits entirely in A100 80GB VRAM
    without CPU offload, giving ~5x speedup over FLUX.2-dev.
    """

    def __init__(self, config: dict):
        import torch
        from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel
        from diffusers.models import FluxMultiControlNetModel

        ai_cfg = config["ai_render"]
        base_model = ai_cfg.get("flux_cn_i2i_model", "sayakpaul/FLUX.1-merged")
        cn_model = ai_cfg.get("flux_controlnet", "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro")

        device, dtype, need_offload = _get_device_and_dtype()

        print(f"  Loading FLUX ControlNet: {cn_model}")
        controlnet = FluxControlNetModel.from_pretrained(cn_model, torch_dtype=dtype)
        controlnet = FluxMultiControlNetModel([controlnet])

        print(f"  Loading FLUX base (ungated): {base_model}")
        self.pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
            base_model, controlnet=controlnet, torch_dtype=dtype,
        )
        if need_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        self.config = config
        self._torch = torch

    def _extract_canny(self, image: Image.Image) -> Image.Image:
        import cv2
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr
        edges = cv2.Canny(gray, 50, 150)
        return Image.fromarray(edges).convert("RGB")

    def rerender(self, rgb_path: str, depth_path: str,
                 normal_path: Optional[str] = None) -> Image.Image:
        ai_cfg = self.config["ai_render"]

        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path).convert("RGB")
        width, height = rgb_image.size

        target_w = min((width // 16) * 16 or 1024, 1536)
        target_h = min((height // 16) * 16 or 1024, 1536)
        target_size = (target_w, target_h)

        rgb_resized = rgb_image.resize(target_size, Image.LANCZOS)
        depth_resized = depth_image.resize(target_size, Image.LANCZOS)
        canny_image = self._extract_canny(rgb_image).resize(target_size, Image.LANCZOS)

        # Triple constraint parameters
        strength = ai_cfg.get("flux_cn_i2i_strength", 0.20)  # Only modify 20% of pixels
        cn_scales = ai_cfg.get("flux_cn_i2i_conditioning_scales", [0.6, 0.5])  # [depth, canny]
        num_steps = ai_cfg.get("flux_cn_i2i_num_steps", 28)

        result = self.pipe(
            prompt=ai_cfg["prompt"],
            image=rgb_resized,                          # Img2Img: pixel starting point
            control_image=[depth_resized, canny_image],  # ControlNet: structural constraints
            control_mode=[2, 0],                         # 2=depth, 0=canny
            strength=strength,                           # Low strength = high structure preservation
            controlnet_conditioning_scale=cn_scales,
            num_inference_steps=num_steps,
            guidance_scale=ai_cfg.get("flux_guidance_scale", 3.5),
            height=target_h, width=target_w,
            generator=self._torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        if result.size != (width, height):
            result = result.resize((width, height), Image.LANCZOS)
        return result

    @property
    def name(self):
        return "FLUX.1-dev + ControlNet + Img2Img (Triple Constraint)"


# ============================================================
# Backend 6: SD 3.5 Large
# ============================================================

class SD35Backend:
    """SD3.5 Large img2img. Good quality, lighter weight."""

    def __init__(self, config: dict):
        import torch
        from diffusers import StableDiffusion3Img2ImgPipeline

        ai_cfg = config["ai_render"]
        model_id = ai_cfg.get("sd35_model", "stabilityai/stable-diffusion-3.5-large")
        device, dtype, need_offload = _get_device_and_dtype()

        print(f"  Loading SD3.5 Large: {model_id}")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if need_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        self.config = config
        self._torch = torch

    def rerender(self, rgb_path: str, depth_path: str,
                 normal_path: Optional[str] = None) -> Image.Image:
        ai_cfg = self.config["ai_render"]
        rgb_image = Image.open(rgb_path).convert("RGB")
        width, height = rgb_image.size
        rgb_resized = rgb_image.resize((1024, 1024), Image.LANCZOS)

        result = self.pipe(
            prompt=ai_cfg["prompt"],
            negative_prompt=ai_cfg.get("negative_prompt", ""),
            image=rgb_resized,
            strength=ai_cfg.get("sd35_strength", 0.55),
            guidance_scale=ai_cfg.get("sd35_guidance_scale", 5.0),
            num_inference_steps=ai_cfg.get("sd35_num_steps", 28),
            generator=self._torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        if result.size != (width, height):
            result = result.resize((width, height), Image.LANCZOS)
        return result

    @property
    def name(self):
        return "Stable Diffusion 3.5 Large"


# ============================================================
# Backend 7: SDXL + ControlNet (Legacy)
# ============================================================

class SDXLBackend:
    """Legacy SDXL + ControlNet. Use only if other backends unavailable."""

    def __init__(self, config: dict):
        import torch
        from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL

        ai_cfg = config["ai_render"]
        device, dtype_unused, need_offload = _get_device_and_dtype()
        dtype = torch.float16  # SDXL works better with fp16

        controlnet = ControlNetModel.from_pretrained(
            ai_cfg.get("controlnet_depth", "diffusers/controlnet-depth-sdxl-1.0"), torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            ai_cfg.get("model", "stabilityai/stable-diffusion-xl-base-1.0"),
            controlnet=controlnet, vae=vae, torch_dtype=dtype)
        if need_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

        self.config = config
        self._torch = torch

    def rerender(self, rgb_path: str, depth_path: str,
                 normal_path: Optional[str] = None) -> Image.Image:
        ai_cfg = self.config["ai_render"]
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path).convert("L")
        width, height = rgb_image.size

        rgb_resized = rgb_image.resize((1024, 1024), Image.LANCZOS)
        depth_resized = depth_image.resize((1024, 1024), Image.LANCZOS)

        result = self.pipe(
            prompt=ai_cfg["prompt"], negative_prompt=ai_cfg.get("negative_prompt", ""),
            image=rgb_resized, control_image=depth_resized,
            strength=ai_cfg.get("strength", 0.65),
            guidance_scale=ai_cfg.get("guidance_scale", 7.5),
            num_inference_steps=ai_cfg.get("num_inference_steps", 30),
            controlnet_conditioning_scale=ai_cfg.get("controlnet_conditioning_scale", 0.8),
            generator=self._torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        if result.size != (width, height):
            result = result.resize((width, height), Image.LANCZOS)
        return result

    @property
    def name(self):
        return "SDXL + ControlNet (Legacy)"


# ============================================================
# Geometric Consistency Verification
# ============================================================

def verify_geometric_consistency(
    original_path: str, rerendered_path: str, depth_path: str,
    threshold: float = 0.15,
) -> dict:
    """Verify geometric consistency via edge SSIM and depth-weighted comparison."""
    from skimage.metrics import structural_similarity as ssim
    from scipy import ndimage

    original = np.array(Image.open(original_path).convert("L"))
    rerendered = np.array(Image.open(rerendered_path).convert("L"))
    depth = np.array(Image.open(depth_path).convert("L"))

    if original.shape != rerendered.shape:
        rerendered = np.array(
            Image.open(rerendered_path).convert("L").resize(
                (original.shape[1], original.shape[0]), Image.LANCZOS))

    edges_orig = ndimage.sobel(original.astype(float))
    edges_reren = ndimage.sobel(rerendered.astype(float))

    data_range = max(edges_orig.max() - edges_orig.min(), 1e-8)
    edge_ssim = ssim(edges_orig, edges_reren, data_range=data_range)

    depth_weight = 1.0 - depth.astype(float) / 255.0
    depth_weight = depth_weight / (depth_weight.sum() + 1e-8)
    diff = np.abs(original.astype(float) - rerendered.astype(float)) / 255.0
    weighted_diff = float((diff * depth_weight).sum())

    edge_corr = float(np.corrcoef(edges_orig.ravel(), edges_reren.ravel())[0, 1])

    return {
        "edge_ssim": float(edge_ssim),
        "edge_correlation": edge_corr,
        "weighted_pixel_diff": weighted_diff,
        "passes": bool(edge_ssim > (1.0 - threshold) and edge_corr > 0.5),
    }


# ============================================================
# Scene Processing
# ============================================================

def process_scene(backend, scene_renders_dir: str, output_dir: str, config: dict) -> dict:
    scene_path = Path(scene_renders_dir)
    out_path = Path(output_dir) / scene_path.name
    out_path.mkdir(parents=True, exist_ok=True)

    results = {"scene_id": scene_path.name, "backend": backend.name, "views": []}

    view_dirs = sorted([
        d for d in scene_path.iterdir()
        if d.is_dir() and d.name.startswith("view_")
    ])

    for view_dir in view_dirs:
        rgb_path = view_dir / "rgb.png"
        depth_path = view_dir / "depth.png"
        normal_path = view_dir / "normal.png"

        if not rgb_path.exists() or not depth_path.exists():
            print(f"  Skipping {view_dir.name}: missing rgb or depth")
            continue

        view_output = out_path / view_dir.name
        view_output.mkdir(exist_ok=True)

        rerendered = backend.rerender(
            str(rgb_path), str(depth_path),
            str(normal_path) if normal_path.exists() else None,
        )
        output_path = view_output / "rgb_photorealistic.png"
        rerendered.save(str(output_path), quality=95)

        consistency = verify_geometric_consistency(
            str(rgb_path), str(output_path), str(depth_path))

        results["views"].append({
            "view_id": view_dir.name,
            "output_path": str(output_path),
            "consistency": consistency,
        })

        status = "PASS" if consistency["passes"] else "FAIL"
        print(f"  {view_dir.name}: edge_ssim={consistency['edge_ssim']:.3f} "
              f"edge_corr={consistency['edge_correlation']:.3f} [{status}]")

    topdown_src = scene_path / "topdown.png"
    if topdown_src.exists():
        import shutil
        shutil.copy2(str(topdown_src), str(out_path / "topdown.png"))

    with open(out_path / "rerender_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# Main
# ============================================================

BACKENDS = {
    "flux-cn-i2i":  ("FLUX.1-dev + ControlNet + Img2Img — triple constraint, highest fidelity (~24GB)", FluxCNImg2ImgBackend),
    "flux2":        ("FLUX.2-dev — 32B native editing, best quality, up to 4MP (~40GB)", Flux2Backend),
    "flux2-klein":  ("FLUX.2-klein — distilled 8B, 4-step fast inference (~20GB)", Flux2KleinBackend),
    "flux-kontext": ("FLUX.1-Kontext-dev — 12B instruction editing (~24GB)", FluxKontextBackend),
    "qwen":         ("Qwen-Image — dual VL+VAE encoding, Apache 2.0 (~30GB)", QwenImageBackend),
    "flux-cn":      ("FLUX.1-dev + ControlNet Union Pro — depth+canny dual control (~24GB)", FluxControlNetBackend),
    "sd35":         ("Stable Diffusion 3.5 Large — solid img2img (~16GB)", SD35Backend),
    "sdxl":         ("SDXL + ControlNet — legacy fallback (~10GB)", SDXLBackend),
}


def main():
    parser = argparse.ArgumentParser(
        description="AI Re-rendering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends (recommended order for A100 80GB):
  flux2          FLUX.2-dev (32B) — native editing, best quality, up to 4MP
  flux2-klein    FLUX.2-klein (8B) — distilled, 4-step fast inference
  flux-kontext   FLUX.1-Kontext-dev (12B) — instruction-based editing
  qwen           Qwen-Image — dual VL+VAE encoding, Apache 2.0
  flux-cn        FLUX.1-dev + ControlNet Union Pro — depth+canny dual control
  sd35           Stable Diffusion 3.5 Large — solid img2img
  sdxl           SDXL + ControlNet — legacy fallback
  all            Run ALL backends for comparison (outputs to separate dirs)
        """,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--backend", default="flux2",
                        choices=list(BACKENDS.keys()) + ["all"])
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    base_output_dir = args.output_dir or config["data"]["ai_renders_dir"]
    renders_dir = Path(args.renders_dir)

    if args.scene_id:
        scene_dirs = [renders_dir / args.scene_id]
    else:
        scene_dirs = sorted([d for d in renders_dir.iterdir() if d.is_dir()])

    # Determine which backends to run
    if args.backend == "all":
        backends_to_run = list(BACKENDS.keys())
    else:
        backends_to_run = [args.backend]

    for backend_name in backends_to_run:
        backend_desc, backend_cls = BACKENDS[backend_name]

        # Each backend gets its own output directory when running "all"
        if args.backend == "all":
            output_dir = str(Path(base_output_dir) / backend_name)
        else:
            output_dir = base_output_dir

        print(f"\n{'='*60}")
        print(f"Backend: {backend_desc}")
        print(f"Output:  {output_dir}")
        print(f"Scenes:  {len(scene_dirs)}")
        print(f"{'='*60}")

        if not args.verify_only:
            print("Loading model pipeline...")
            try:
                backend = backend_cls(config)
            except Exception as e:
                print(f"  FAILED to load backend '{backend_name}': {e}")
                print(f"  Skipping this backend.")
                continue
            print("Pipeline loaded.")
        else:
            backend = None

        pass_count = 0
        fail_count = 0

        for i, scene_dir in enumerate(scene_dirs):
            print(f"[{i+1}/{len(scene_dirs)}] {scene_dir.name}")

            if args.skip_existing and (Path(output_dir) / scene_dir.name / "rerender_results.json").exists():
                print("  Skipping (already processed)")
                continue

            try:
                if args.verify_only:
                    result_file = Path(output_dir) / scene_dir.name / "rerender_results.json"
                    if result_file.exists():
                        with open(result_file) as f:
                            result = json.load(f)
                    else:
                        continue
                else:
                    result = process_scene(backend, str(scene_dir), output_dir, config)

                for v in result.get("views", []):
                    if v.get("consistency", {}).get("passes", False):
                        pass_count += 1
                    else:
                        fail_count += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        total = pass_count + fail_count
        print(f"\n--- {backend_name} Summary ---")
        print(f"Total views: {total}")
        print(f"Passed: {pass_count} | Failed: {fail_count}")
        if total > 0:
            print(f"Pass rate: {pass_count / total * 100:.1f}%")

        # Free GPU memory before loading next backend
        if not args.verify_only and backend is not None:
            del backend
            if 'torch' in dir():
                import torch
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
