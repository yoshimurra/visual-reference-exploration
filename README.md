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

## Local refinement from a selected cell

After the coarse grid finishes, select a cell from the contact sheet. Rows and columns are zero-based. For example, to refine around `row=2`, `col=3`:

```bash
python experiments/create_local_config.py \
  --manifest experiment_results/coarse_grid/manifest.json \
  --row 2 \
  --col 3 \
  --color-radius 0.10 \
  --texture-radius 0.30 \
  --output configs/local_grid_selected.json
```

This creates a 5 x 5 local search centered at the selected cell while preserving the coarse experiment's seed, steps, and interpolation mode. Then run:

```bash
python experiments/run_grid.py \
  --config configs/local_grid_selected.json \
  --backend backend/infer_sadis_grid.py \
  --output experiment_results/local_grid_row02_col03 \
  --low-memory
```

Use `--grid-size`, `--color-radius`, and `--texture-radius` to change the local search resolution and range. Add `--force` when intentionally replacing an existing local config.

## Contact sheet

```bash
python experiments/make_contact_sheet.py \
  --input experiment_results/coarse_grid \
  --output experiment_results/coarse_grid.jpg
```

For the local grid:

```bash
python experiments/make_contact_sheet.py \
  --input experiment_results/local_grid_row02_col03 \
  --output experiment_results/local_grid_row02_col03.jpg
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
