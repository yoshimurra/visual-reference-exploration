#!/usr/bin/env python3
"""SADis backend for color-reference mixture x texture-strength grids.

The model implementation stays in ``vendor/SADis``. This script adds a stable
CLI, two-reference color interpolation, deterministic output paths, and a
low-memory initialization path suitable for consumer GPUs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def existing_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def first_image(directory: Path) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    candidates = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No image found in: {directory}")
    return candidates[0]


def resolve_image(explicit: Path | None, default_directory: Path) -> Path:
    return explicit if explicit is not None else first_image(default_directory)


def interpolate_reference_images(
    image_a: Image.Image,
    image_b: Image.Image,
    mix: float,
    color_space: str,
) -> Image.Image:
    image_b = image_b.resize(image_a.size, Image.Resampling.LANCZOS)
    rgb_a = np.asarray(image_a, dtype=np.float32) / 255.0
    rgb_b = np.asarray(image_b, dtype=np.float32) / 255.0

    if color_space == "rgb":
        mixed_rgb = (1.0 - mix) * rgb_a + mix * rgb_b
    elif color_space == "lab":
        lab_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2LAB)
        lab_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2LAB)
        mixed_lab = (1.0 - mix) * lab_a + mix * lab_b
        mixed_rgb = cv2.cvtColor(mixed_lab, cv2.COLOR_LAB2RGB)
    else:
        raise ValueError(f"Unsupported image interpolation: {color_space}")

    return Image.fromarray(np.uint8(np.clip(mixed_rgb, 0.0, 1.0) * 255.0))


def as_list(image: Image.Image | Iterable[Image.Image]) -> list[Image.Image]:
    return [image] if isinstance(image, Image.Image) else list(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sadis-root", type=Path, default=Path("vendor/SADis"))
    parser.add_argument("--color-a", type=existing_file)
    parser.add_argument("--color-b", type=existing_file)
    parser.add_argument("--texture", type=existing_file)
    parser.add_argument("--color_mix", type=float, required=True)
    parser.add_argument("--texture_scale", type=float, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--interpolation", choices=("embedding", "rgb", "lab"), default="embedding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt", default="a girl")
    parser.add_argument(
        "--negative-prompt",
        default=(
            "text, watermark, lowres, low quality, worst quality, deformed, "
            "glitch, low contrast, noisy, saturation, blurry, gray color"
        ),
    )
    parser.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--image-encoder", type=Path, default=Path("models/image_encoder"))
    parser.add_argument(
        "--ip-checkpoint",
        type=Path,
        default=Path("sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--color-scale", type=float, default=1.1)
    parser.add_argument("--gray-scale", type=float, default=1.1)
    parser.add_argument("--wct-guidance", type=float, default=0.5)
    parser.add_argument("--wct-start", type=float, default=0.2)
    parser.add_argument("--wct-end", type=float, default=0.3)
    parser.add_argument("--wct-noise", type=float, default=0.01)
    parser.add_argument("--punish-weight", type=float, default=0.003)
    parser.add_argument("--punish-type", default="soft-weight")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.color_mix <= 1.0:
        raise ValueError("--color_mix must be in [0, 1]")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.texture_scale < 0.0:
        raise ValueError("--texture_scale must be non-negative")

    repo_root = Path(__file__).resolve().parents[1]
    sadis_root = args.sadis_root.expanduser().resolve()
    if not (sadis_root / "pipeline_stable_diffusion_xl.py").is_file():
        raise FileNotFoundError(
            f"SADis was not found at {sadis_root}. Run scripts/bootstrap_sadis.sh first."
        )
    sys.path.insert(0, str(sadis_root))

    from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline  # type: ignore
    from ip_adapter import IPAdapterPlusXL  # type: ignore
    from ip_adapter.ip_adapter import punish_weight_module  # type: ignore
    from ip_adapter.utils import get_generator  # type: ignore

    color_a_path = resolve_image(args.color_a, repo_root / "assets/color_a")
    color_b_path = resolve_image(args.color_b, repo_root / "assets/color_b")
    texture_path = resolve_image(args.texture, repo_root / "assets/texture")

    color_a = Image.open(color_a_path).convert("RGB")
    color_b = Image.open(color_b_path).convert("RGB")
    texture = Image.open(texture_path).convert("RGB").convert("L")

    if args.interpolation in {"rgb", "lab"}:
        color = interpolate_reference_images(
            color_a, color_b, args.color_mix, args.interpolation
        )
        color_b_for_embedding: Image.Image | None = None
    else:
        color = color_a
        color_b_for_embedding = color_b

    model_device = "cpu" if args.low_memory else args.device
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    ip_model = IPAdapterPlusXL(
        pipe,
        str(args.image_encoder),
        str(args.ip_checkpoint),
        model_device,
        num_tokens=16,
        target_blocks=["up_blocks.0.attentions.1"],
        ca_mask=False,
    )
    if args.low_memory:
        pipe.enable_model_cpu_offload(gpu_id=int(args.device.split(":")[-1]))
    runtime_device = torch.device(args.device)

    @torch.inference_mode()
    def encode_image_tokens() -> tuple[torch.Tensor, torch.Tensor]:
        image_encoder = ip_model.image_encoder
        image_proj_model = ip_model.image_proj_model
        if args.low_memory:
            image_encoder.to(runtime_device, dtype=torch.float16)
            image_proj_model.to(runtime_device, dtype=torch.float16)

        def hidden(images: Image.Image | Iterable[Image.Image]) -> torch.Tensor:
            pixels = ip_model.clip_image_processor(
                images=as_list(images), return_tensors="pt"
            ).pixel_values
            pixels = pixels.to(runtime_device, dtype=torch.float16)
            return image_encoder(pixels, output_hidden_states=True).hidden_states[-2]

        color_embed = hidden(color)
        gray_embed = hidden(color.convert("L"))
        if color_b_for_embedding is not None:
            color_b_embed = hidden(color_b_for_embedding)
            gray_b_embed = hidden(color_b_for_embedding.convert("L"))
            mix = float(args.color_mix)
            color_embed = (1.0 - mix) * color_embed + mix * color_b_embed
            gray_embed = (1.0 - mix) * gray_embed + mix * gray_b_embed

        texture_embed = hidden(texture)
        texture_np = np.asarray(texture)
        mean_value = np.asarray(texture_np.mean(), dtype=np.uint8)
        pure_gray = Image.fromarray(np.full_like(texture_np, mean_value, dtype=np.uint8))
        pure_gray_embed = hidden(pure_gray)

        if args.punish_weight != 0 and args.punish_type.lower() != "none":
            dtype = texture_embed.dtype
            _, token_count, _ = texture_embed.shape
            matrix = torch.cat([texture_embed, pure_gray_embed], dim=1).squeeze(0)
            latent_size = matrix.shape[0]
            matrix = punish_weight_module(
                matrix.permute(1, 0).float(),
                latent_size,
                alpha=args.punish_weight,
                method=args.punish_type,
            )
            texture_embed = matrix.permute(1, 0)[:token_count].unsqueeze(0).to(dtype)

        combined = (
            args.color_scale * color_embed
            - args.gray_scale * gray_embed
            + args.texture_scale * texture_embed
        )
        image_prompt_embeds = image_proj_model(combined)
        zero_pixels = ip_model.clip_image_processor(
            images=[color], return_tensors="pt"
        ).pixel_values.to(runtime_device, dtype=torch.float16)
        zero_embed = image_encoder(
            torch.zeros_like(zero_pixels), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = image_proj_model(zero_embed)

        if args.low_memory:
            image_encoder.to("cpu")
            image_proj_model.to("cpu")
            torch.cuda.empty_cache()
        return image_prompt_embeds, uncond_image_prompt_embeds

    ip_model.set_scale(1.0)
    image_prompt_embeds, uncond_image_prompt_embeds = encode_image_tokens()

    with torch.inference_mode():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            [args.prompt],
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[args.negative_prompt],
        )
        embed_device = prompt_embeds.device
        image_prompt_embeds = image_prompt_embeds.to(embed_device)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(embed_device)
        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
        )

    generator = get_generator(args.seed, args.device)
    images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
        wct_starts_step=args.wct_start * args.steps,
        wct_ends_step=args.wct_end * args.steps,
        wct_guidance=args.wct_guidance,
        csd_iter_num=1,
        clr_ref_img_dir=str(color_a_path),
        sty_ref_img_dir=str(texture_path),
        wctnoise_add_scale=args.wct_noise,
    ).images

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"mix{args.color_mix:.3f}_texture{args.texture_scale:.3f}_"
        f"{args.interpolation}_seed{args.seed}.png"
    )
    images[0].save(output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
