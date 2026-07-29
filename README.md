# Visual Reference Exploration

A separate experiment repository for reference-image-based SDXL/SADis exploration. The prompt is treated as a fixed content condition; color and texture are explored through direct parameter manipulation.

## Experiments

- Coarse grid: color-reference mixture x texture strength
- Local refinement around a selected grid cell
- Contact-sheet generation
- Adjacent-cell color-continuity diagnostics in Lab space

## Backend requirement

The experiment runner calls an existing SADis-compatible inference script. The backend must accept these arguments:

```text
--color_mix FLOAT
--texture_scale FLOAT
--steps INT
--interpolation embedding|rgb|lab
--seed INT
--output_dir PATH
--low-memory
```

The current `research2026` script already supports color mixture, steps, interpolation, and low-memory mode. Add CLI arguments for `texture_scale`, `seed`, and `output_dir`, then pass them to the existing variables and save path.

## Coarse exploration

```bash
python experiments/run_grid.py \
  --config configs/coarse_grid.json \
  --backend ../research2026/infer_style_plus_color_texture.py \
  --output experiment_results/coarse_grid \
  --low-memory
```

Use `--dry-run` to inspect commands without running SDXL.

## Local refinement

Edit the center and radius values in `configs/local_grid.json`, then run:

```bash
python experiments/run_grid.py \
  --config configs/local_grid.json \
  --backend ../research2026/infer_style_plus_color_texture.py \
  --output experiment_results/local_grid \
  --low-memory
```

## Contact sheet

```bash
python experiments/make_contact_sheet.py \
  --input experiment_results/coarse_grid \
  --output experiment_results/coarse_grid.jpg
```

## Color-continuity diagnostic

Requires Pillow, NumPy, and OpenCV.

```bash
python experiments/evaluate_color_continuity.py \
  --input experiment_results/coarse_grid \
  --output experiment_results/color_continuity.csv
```

The metric is mean pixelwise CIE76 distance between adjacent generated images. It is intended as a simple diagnostic rather than a complete perceptual evaluation.

## Repository policy

Model weights, generated images, caches, and local paths are excluded from Git. Keep the existing research repository unchanged and use it only as the generation backend.
