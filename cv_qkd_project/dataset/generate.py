from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cv_qkd_project import config
from cv_qkd_project.optimization.brute_force import optimal_VA
from cv_qkd_project.side_channel.key_rate_mismatch import key_rate_mismatch


def generate_dataset(
    N: int = config.DATASET_SIZE_N,
    seed: int = 123,
    out_inputs_csv: str | os.PathLike = "data/raw/dataset_inputs.csv",
    out_labels_csv: str | os.PathLike = "data/raw/dataset_labels.csv",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simulation dataset for NN-based modulation optimization under mismatch.

    Samples tuples (T, xi, eta1, eta2) uniformly within the bounds in `config.py`
    and labels each tuple with the brute-force optimal modulation variance V_A*
    computed using the mismatch-aware key rate.

    Outputs
    -------
    - Inputs: shape (N, 4) with columns [T, xi, eta1, eta2]
    - Labels: shape (N,) with V_A* (SNU)
    """
    rng = np.random.default_rng(seed)

    T = rng.uniform(config.T_MIN, config.T_MAX, size=N)
    xi = rng.uniform(config.XI_MIN, config.XI_MAX, size=N)
    eta1 = rng.uniform(config.ETA_MIN, config.ETA_MAX, size=N)
    eta2 = rng.uniform(config.ETA_MIN, config.ETA_MAX, size=N)

    X = np.stack([T, xi, eta1, eta2], axis=1)
    y = np.empty((N,), dtype=float)

    for i in tqdm(range(N), desc="Generating dataset (brute-force V_A*)"):
        t, x, e1, e2 = X[i]
        v_star, _ = optimal_VA(
            T=float(t),
            xi=float(x),
            eta1=float(e1),
            eta2=float(e2),
            V_el=config.V_EL,
            beta=config.BETA,
            key_rate_fn=key_rate_mismatch,
        )
        y[i] = v_star

    out_inputs_csv = Path(out_inputs_csv)
    out_labels_csv = Path(out_labels_csv)
    os.makedirs(out_inputs_csv.parent, exist_ok=True)
    os.makedirs(out_labels_csv.parent, exist_ok=True)

    pd.DataFrame(X, columns=["T", "xi", "eta1", "eta2"]).to_csv(out_inputs_csv, index=False)
    pd.DataFrame({"V_A_star": y}).to_csv(out_labels_csv, index=False)

    return X, y


def print_dataset_stats(X: np.ndarray, y: np.ndarray) -> None:
    cols = ["T", "xi", "eta1", "eta2"]
    print("\nDataset statistics")
    for j, name in enumerate(cols):
        c = X[:, j]
        print(f"- {name}: mean={c.mean():.6g}, std={c.std():.6g}, min={c.min():.6g}, max={c.max():.6g}")
    print(f"- V_A_star: mean={y.mean():.6g}, std={y.std():.6g}, min={y.min():.6g}, max={y.max():.6g}")


if __name__ == "__main__":
    X, y = generate_dataset()
    print_dataset_stats(X, y)
    print("\nSaved:")
    print("- data/raw/dataset_inputs.csv")
    print("- data/raw/dataset_labels.csv")

