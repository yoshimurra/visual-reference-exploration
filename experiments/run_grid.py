#!/usr/bin/env python3
"""Run a coarse or local color/texture exploration grid.

The backend script must accept:
  --color_mix FLOAT
  --texture_scale FLOAT
  --steps INT
  --interpolation NAME
  --seed INT
  --output_dir PATH

This repository intentionally keeps the experiment coordinator separate from
model-specific SADis code.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def linspace(center: float, radius: float, size: int, lower: float | None = None) -> list[float]:
    if size < 2:
        values = [center]
    else:
        start = center - radius
        step = 2.0 * radius / (size - 1)
        values = [start + step * index for index in range(size)]
    if lower is not None:
        values = [max(lower, value) for value in values]
    return values


def parameter_grid(config: dict[str, Any]) -> tuple[list[float], list[float]]:
    if "color_mixes" in config:
        return list(config["color_mixes"]), list(config["texture_scales"])

    size = int(config["grid_size"])
    colors = linspace(
        float(config["center_color_mix"]),
        float(config["color_radius"]),
        size,
        lower=0.0,
    )
    colors = [min(1.0, value) for value in colors]
    textures = linspace(
        float(config["center_texture_scale"]),
        float(config["texture_radius"]),
        size,
        lower=0.0,
    )
    return colors, textures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    color_mixes, texture_scales = parameter_grid(config)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for row, texture_scale in enumerate(texture_scales):
        for column, color_mix in enumerate(color_mixes):
            cell_dir = args.output / f"row{row:02d}_col{column:02d}"
            command = [
                args.python,
                str(args.backend),
                "--color_mix",
                f"{color_mix:.6f}",
                "--texture_scale",
                f"{texture_scale:.6f}",
                "--steps",
                str(int(config.get("steps", 30))),
                "--interpolation",
                str(config.get("interpolation", "embedding")),
                "--seed",
                str(int(config.get("seed", 42))),
                "--output_dir",
                str(cell_dir),
            ]
            if args.low_memory:
                command.append("--low-memory")

            record = {
                "row": row,
                "column": column,
                "color_mix": color_mix,
                "texture_scale": texture_scale,
                "output_dir": str(cell_dir),
                "command": command,
            }
            manifest.append(record)
            print(" ".join(command), flush=True)
            if not args.dry_run:
                cell_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True)

    with (args.output / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
