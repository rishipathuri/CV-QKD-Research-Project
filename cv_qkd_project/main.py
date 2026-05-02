from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cv_qkd_project import config
from cv_qkd_project.dataset.generate import generate_dataset
from cv_qkd_project.dataset.preprocess import preprocess_dataset
from cv_qkd_project.experiments.experiment1_ideal import run_experiment1_ideal
from cv_qkd_project.experiments.experiment1b_mismatch_optimal import run_experiment1b_mismatch_optimal
from cv_qkd_project.experiments.experiment2_naive import run_experiment2_naive
from cv_qkd_project.experiments.experiment3_adaptive import run_experiment3_adaptive
from cv_qkd_project.experiments.robustness import run_all_robustness
from cv_qkd_project.figures.plot_utils import plot_three_experiments
from cv_qkd_project.model.train import train


def _np_to_jsonable(obj):
    """Best-effort conversion of numpy objects into JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_run_config_snapshot(out_path: str | os.PathLike = "outputs/results/run_config.json") -> Path:
    """
    Persist a JSON snapshot of important simulation/training configuration values.

    This lives under `outputs/results/` (normally ignored) but can be explicitly
    tracked via `.gitignore` exceptions if desired for reproducibility notes.
    """
    cfg_dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cv_qkd_project.config": {
            "T_MIN": float(config.T_MIN),
            "T_MAX": float(config.T_MAX),
            "XI_MIN": float(config.XI_MIN),
            "XI_MAX": float(config.XI_MAX),
            "ETA_MIN": float(config.ETA_MIN),
            "ETA_MAX": float(config.ETA_MAX),
            "V_EL": float(config.V_EL),
            "BETA": float(config.BETA),
            "MISMATCH_NOISE_SCALE": float(config.MISMATCH_NOISE_SCALE),
            "V_A_GRID_SIZE": int(config.V_A_GRID_SIZE),
            "V_A_MIN": float(config.V_A_MIN),
            "V_A_MAX": float(config.V_A_MAX),
            "DATASET_SIZE_N": int(config.DATASET_SIZE_N),
            # Large but bounded (200 points): keep as array for exact reproducibility.
            "V_A_GRID": _np_to_jsonable(config.V_A_GRID),
        },
        "paths": {
            "raw_data_dir": "data/raw",
            "processed_dir": "data/processed",
            "checkpoint": "checkpoints/best_model.pt",
            "results_dir": "outputs/results",
            "figures_dir": "outputs/figures",
            "paper_figures_dir": "figures",
        },
    }

    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    out_path.write_text(json.dumps(cfg_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[main] wrote config snapshot: {out_path}")
    return out_path


def _raw_data_empty(raw_dir: str | os.PathLike = "data/raw") -> bool:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return True
    # treat as empty if no CSVs are present
    return len(list(raw_dir.glob("*.csv"))) == 0


def _processed_ready(processed_dir: str | os.PathLike = "data/processed") -> bool:
    processed_dir = Path(processed_dir)
    needed = [
        "X_train.npy",
        "X_val.npy",
        "X_test.npy",
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
        "scaler.npy",
    ]
    return processed_dir.exists() and all((processed_dir / f).exists() for f in needed)


def _checkpoint_exists(path: str | os.PathLike = "checkpoints/best_model.pt") -> bool:
    return Path(path).exists()


def _summary_at_T(df: pd.DataFrame, T0: float, y_col: str) -> float:
    """
    Return y at the closest T value (by absolute difference).
    """
    idx = int(np.argmin(np.abs(df["T"].to_numpy(dtype=float) - float(T0))))
    return float(df.iloc[idx][y_col])


def run_pipeline() -> None:
    # 1) Dataset generation (if needed)
    if _raw_data_empty():
        print("[main] data/raw is empty -> generating dataset")
        generate_dataset(N=config.DATASET_SIZE_N)
    else:
        print("[main] data/raw present -> skipping dataset generation")

    # 2) Preprocess (if needed)
    if not _processed_ready():
        print("[main] data/processed missing/incomplete -> preprocessing")
        preprocess_dataset()
    else:
        print("[main] data/processed ready -> skipping preprocessing")

    # 3) Train (if needed)
    if not _checkpoint_exists():
        print("[main] checkpoints/best_model.pt missing -> training")
        train()
    else:
        print("[main] checkpoint exists -> skipping training")

    # 4) Run experiments (include mismatch-aware upper bound)
    run_experiment1_ideal()
    run_experiment1b_mismatch_optimal()
    run_experiment2_naive()
    run_experiment3_adaptive()

    # 5) Robustness studies
    run_all_robustness()

    # 6) Combined figure
    plot_three_experiments()

    # 7) Persist reproducibility snapshot
    write_run_config_snapshot()

    # 8) Summary at T=0.5
    exp1 = pd.read_csv("outputs/results/experiment1.csv")
    exp1b = pd.read_csv("outputs/results/experiment1b_mismatch_opt.csv")
    exp2 = pd.read_csv("outputs/results/experiment2.csv")
    exp3 = pd.read_csv("outputs/results/experiment3.csv")
    T0 = 0.5
    K_ideal = _summary_at_T(exp1, T0, "K_max")
    K_mismatch_opt = _summary_at_T(exp1b, T0, "K_max")
    K_naive = _summary_at_T(exp2, T0, "K_actual")
    K_adapt = _summary_at_T(exp3, T0, "K_actual")
    recovery_vs_ideal = (K_adapt / K_ideal * 100.0) if K_ideal > 0 else float("nan")
    recovery_vs_mismatch_opt = (K_adapt / K_mismatch_opt * 100.0) if K_mismatch_opt > 0 else float("nan")

    print("\n=== Summary at T=0.5 (closest grid point) ===")
    print(f"Ideal key rate   : {K_ideal:.6g} bits/use")
    print(f"Mismatch-opt rate: {K_mismatch_opt:.6g} bits/use")
    print(f"Naive key rate   : {K_naive:.6g} bits/use")
    print(f"Adaptive key rate: {K_adapt:.6g} bits/use")
    print(f"Recovery vs ideal: {recovery_vs_ideal:.2f}%")
    print(f"Recovery vs mismatch-opt: {recovery_vs_mismatch_opt:.2f}%")


if __name__ == "__main__":
    run_pipeline()

