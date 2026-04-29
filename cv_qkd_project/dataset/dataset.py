from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class QKDDataset(Dataset):
    """
    PyTorch dataset for CV-QKD modulation optimization.

    This dataset reads the processed arrays produced by `dataset/preprocess.py`
    and returns:

        (x, log_y)

    where:
    - x is a standardized 4-feature input tensor (float32)
    - log_y is log(V_A*) (float32), so the network can learn in log-space

    Expected processed files in `processed_dir`:
    - X_{split}.npy and y_{split}.npy, where split in {train,val,test}
    """

    def __init__(self, split: str = "train", processed_dir: str | Path = "data/processed"):
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        processed_dir = Path(processed_dir)
        X_path = processed_dir / f"X_{split}.npy"
        y_path = processed_dir / f"y_{split}.npy"

        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)

        if self.X.ndim != 2 or self.X.shape[1] != 4:
            raise ValueError(f"Expected X shape (N,4), got {self.X.shape}")
        if self.y.ndim != 1 or self.y.shape[0] != self.X.shape[0]:
            raise ValueError(f"Expected y shape (N,), got {self.y.shape} for X {self.X.shape}")

        # Safe log: V_A* should be >= 1.01 by construction, but guard anyway.
        self.log_y = np.log(np.maximum(self.y, 1e-12)).astype(np.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.log_y[idx], dtype=torch.float32)
        return x, y


if __name__ == "__main__":
    # Smoke test (requires processed files to exist)
    ds = QKDDataset(split="train")
    x, y = ds[0]
    print("x shape:", x.shape, "dtype:", x.dtype)
    print("y (log V_A*):", y.item())

