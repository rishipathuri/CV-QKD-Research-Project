# CV-QKD Research Project (Simulation)

This repository contains a simulation-only pipeline for **Gaussian-modulated coherent-state CV‑QKD**, including:

- physics models (`cv_qkd_project/physics/`)
- detector-efficiency mismatch side-channel modeling (`cv_qkd_project/side_channel/`)
- brute-force modulation optimization (`cv_qkd_project/optimization/`)
- dataset generation + preprocessing (`cv_qkd_project/dataset/`)
- neural modulation predictor training/evaluation (`cv_qkd_project/model/`)
- experiments + robustness studies (`cv_qkd_project/experiments/`)

## Setup

Create a virtual environment (recommended), then install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run the full pipeline

From the repo root:

```bash
python3 -m cv_qkd_project.main
```

### What this does (high level)

On each run, `cv_qkd_project.main`:

1. Generates raw labeled data **only if** `data/raw/` contains no CSVs.
2. Runs preprocessing **only if** `data/processed/` is missing required `.npy` files.
3. Trains the neural predictor **only if** `checkpoints/best_model.pt` does not exist.
4. Runs experiments + robustness studies and saves CSV/plots under `outputs/`.
5. Writes a reproducibility snapshot to `outputs/results/run_config.json`.

## Outputs

Generated artifacts are written under:

- **Raw/processed datasets**: `data/raw/`, `data/processed/` (ignored by git)
- **Model checkpoints**: `checkpoints/` (ignored by git)
- **Results/plots**: `outputs/results/`, `outputs/figures/` (ignored by git)

### Paper-ready figures (tracked)

Copy final publication figures into:

- `figures/` (tracked)

This keeps curated figures separate from auto-generated `outputs/figures/`.

To snapshot the *current* run’s figures into `figures/` with stable filenames:

```bash
python3 -m cv_qkd_project.publish_figures
```

If you re-run training from scratch (e.g. delete `checkpoints/`), results may shift slightly.
The training loop is configured to be reproducible by default (fixed seed + deterministic mode),
but GPU stacks can still introduce tiny differences on some systems.

### Reproducibility snapshot

Each full pipeline run writes:

- `outputs/results/run_config.json`

By default `outputs/` is ignored, but **this file is explicitly allowed** to be tracked for reproducibility notes (see `.gitignore`).
