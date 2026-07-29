#!/usr/bin/env python3
"""Create a local-grid config from one cell in a coarse-grid manifest."""

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--col", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/coarse_grid.json"))
    parser.add_argument("--color-radius", type=float, default=0.10)
    parser.add_argument("--texture-radius", type=float, default=0.30)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}. Use --force to overwrite it.")
    if args.row < 0 or args.col < 0 or args.grid_size < 2:
        raise SystemExit("row/col must be non-negative and grid-size must be at least 2.")
    if args.color_radius <= 0 or args.texture_radius <= 0:
        raise SystemExit("Search radii must be positive.")

    manifest = read_json(args.manifest)
    selected = next(
        (
            item
            for item in manifest
            if item.get("row") == args.row and item.get("column") == args.col
        ),
        None,
    )
    if selected is None:
        raise SystemExit(f"Cell row={args.row}, col={args.col} was not found.")

    base = read_json(args.base_config)
    config = {
        "grid_size": args.grid_size,
        "center_color_mix": float(selected["color_mix"]),
        "color_radius": args.color_radius,
        "center_texture_scale": float(selected["texture_scale"]),
        "texture_radius": args.texture_radius,
        "steps": int(base.get("steps", 30)),
        "seed": int(base.get("seed", 42)),
        "interpolation": base.get("interpolation", "embedding"),
        "source_selection": {
            "row": args.row,
            "column": args.col,
            "manifest": str(args.manifest),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")

    print(
        f"Created {args.output}: center_color_mix={config['center_color_mix']}, "
        f"center_texture_scale={config['center_texture_scale']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
