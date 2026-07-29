#!/usr/bin/env python3
"""Create a labeled contact sheet from run_grid.py outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def find_image(directory: Path) -> Path:
    images = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image found in {directory}")
    return images[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument("--label-height", type=int, default=42)
    args = parser.parse_args()

    manifest_path = args.input / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = max(item["row"] for item in manifest) + 1
    columns = max(item["column"] for item in manifest) + 1

    sheet = Image.new(
        "RGB",
        (columns * args.cell_size, rows * (args.cell_size + args.label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for item in manifest:
        image_path = find_image(Path(item["output_dir"]))
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((args.cell_size, args.cell_size), Image.Resampling.LANCZOS)
            x = item["column"] * args.cell_size + (args.cell_size - image.width) // 2
            y = item["row"] * (args.cell_size + args.label_height)
            sheet.paste(image, (x, y))
            label = f"color={item['color_mix']:.2f}  texture={item['texture_scale']:.2f}"
            draw.text((item["column"] * args.cell_size + 8, y + args.cell_size + 10), label, fill="black")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
