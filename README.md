# Visual Reference Exploration

A standalone experiment repository for reference-image-based SDXL/SADis exploration. The prompt is treated as a fixed content condition; color and texture are explored through direct parameter manipulation.

## Included experiments

- Coarse grid: color-reference mixture x texture strength
- Local refinement around a selected grid cell
- Embedding, RGB, and Lab color interpolation
- Contact-sheet generation
- Adjacent-cell color-continuity diagnostics in Lab space

## Setup

Clone the repository, then fetch the upstream SADis implementation:

```bash
git clone https://github.com/yoshimurra/visual-reference-exploration.git
cd visual-reference-exploration
bash scripts/bootstrap_sadis.sh
```

Create the SADis environment using the upstream environment file or your existing `color_texture` environment.

Place the IP-Adapter model files locally. They are intentionally excluded from Git:

```text
models/image_encoder/
sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
```

Create these reference-image directories and place at least one image in each:

```text
assets/color_a/
assets/color_b/
assets/texture/
```

The first filename in each directory is used by default. Specific files can also be supplied to the backend with `--color-a`, `--color-b`, and `--texture`.

## Test one generation

```bash
python backend/infer_sadis_grid.py \
  --color_mix 0.5 \
  --texture_scale 1.0 \
  --steps 5 \
  --interpolation embedding \
  --seed 42 \
  --output_dir experiment_results/smoke_test \
  --low-memory
```

The backend supports three interpolation modes:

- `embedding`: interpolate CLIP hidden-state features from color references A and B
- `rgb`: interpolate the reference images in RGB space before feature extraction
- `lab`: interpolate the reference images in Lab space before feature extraction

## Coarse exploration

```bash
python experiments/run_grid.py \
  --config configs/coarse_grid.json \
  --backend backend/infer_sadis_grid.py \
  --output experiment_results/coarse_grid \
  --low-memory
```

Use `--dry-run` to inspect all 25 commands without loading SDXL.

## Local refinement

Edit the center and radius values in `configs/local_grid.json`, then run:

```bash
python experiments/run_grid.py \
  --config configs/local_grid.json \
  --backend backend/infer_sadis_grid.py \
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

```bash
python experiments/evaluate_color_continuity.py \
  --input experiment_results/coarse_grid \
  --output experiment_results/color_continuity.csv
```

The metric is mean pixelwise CIE76 distance between adjacent generated images. It is a simple diagnostic rather than a complete perceptual evaluation.

## Reproducibility notes

For comparisons, keep the prompt, seed, inference steps, color references, texture reference, and model settings fixed. Change only the grid parameters. Generated images, downloaded model weights, and the upstream SADis checkout are excluded from Git.
