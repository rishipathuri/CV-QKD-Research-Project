from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_dataset(
    inputs_csv: str | os.PathLike = "data/raw/dataset_inputs.csv",
    labels_csv: str | os.PathLike = "data/raw/dataset_labels.csv",
    out_dir: str | os.PathLike = "data/processed",
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """
    Load raw CSVs, split (80/10/10), standardize features, and save processed arrays.

    - Train/val/test split uses a fixed RNG seed for reproducibility.
    - Standardization uses ONLY training mean/std, then applies to all splits.
    - Scaler parameters are saved to `data/processed/scaler.npy` as a (2,4) array:
        scaler[0] = means, scaler[1] = stds
    - Split arrays saved as:
        X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy
    """
    inputs_csv = Path(inputs_csv)
    labels_csv = Path(labels_csv)
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    X = pd.read_csv(inputs_csv).to_numpy(dtype=float)
    y = pd.read_csv(labels_csv)["V_A_star"].to_numpy(dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched rows: X has {X.shape[0]}, y has {y.shape[0]}")
    if X.shape[1] != 4:
        raise ValueError(f"Expected 4 input features (T, xi, eta1, eta2), got {X.shape[1]}")

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds = np.where(stds > 0, stds, 1.0)

    X_train_s = (X_train - means) / stds
    X_val_s = (X_val - means) / stds
    X_test_s = (X_test - means) / stds

    scaler = np.stack([means, stds], axis=0)
    np.save(out_dir / "scaler.npy", scaler)

    np.save(out_dir / "X_train.npy", X_train_s)
    np.save(out_dir / "X_val.npy", X_val_s)
    np.save(out_dir / "X_test.npy", X_test_s)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "y_test.npy", y_test)

    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "means": means,
        "stds": stds,
    }


if __name__ == "__main__":
    out = preprocess_dataset()
    print("Saved processed arrays to `data/processed/`")
    print("- shapes:",
          "X_train", out["X_train"].shape,
          "X_val", out["X_val"].shape,
          "X_test", out["X_test"].shape,
          "y_train", out["y_train"].shape,
          "y_val", out["y_val"].shape,
          "y_test", out["y_test"].shape)
    print("- scaler means:", out["means"])
    print("- scaler stds: ", out["stds"])

