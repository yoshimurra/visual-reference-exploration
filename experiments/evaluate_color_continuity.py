#!/usr/bin/env python3
"""Measure adjacent-image color changes in Lab space.

This is a lightweight diagnostic, not a perceptual-quality metric. It reports
mean CIE76 distance between neighboring cells along each grid axis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def find_image(directory: Path) -> Path:
    images = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image found in {directory}")
    return images[0]


def lab_image(path: Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return cv2.cvtColor(array, cv2.COLOR_RGB2LAB)


def mean_delta_e(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second, axis=2).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    cells = {(item["row"], item["column"]): item for item in manifest}
    lab_cache = {
        key: lab_image(find_image(Path(item["output_dir"])))
        for key, item in cells.items()
    }

    records: list[dict[str, object]] = []
    for (row, column), item in sorted(cells.items()):
        for axis, neighbor_key in (
            ("color", (row, column + 1)),
            ("texture", (row + 1, column)),
        ):
            if neighbor_key not in cells:
                continue
            neighbor = cells[neighbor_key]
            records.append({
                "axis": axis,
                "row": row,
                "column": column,
                "from_color_mix": item["color_mix"],
                "to_color_mix": neighbor["color_mix"],
                "from_texture_scale": item["texture_scale"],
                "to_texture_scale": neighbor["texture_scale"],
                "mean_delta_e76": mean_delta_e(lab_cache[(row, column)], lab_cache[neighbor_key]),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    for axis in ("color", "texture"):
        values = [float(record["mean_delta_e76"]) for record in records if record["axis"] == axis]
        if values:
            print(f"{axis}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
