from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cv_qkd_project.dataset.dataset import QKDDataset
from cv_qkd_project.model.network import VAPredictor


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 512
    lr: float = 1e-3
    max_epochs: int = 500
    early_stop_patience: int = 30
    lr_patience: int = 20
    lr_factor: float = 0.5
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123
    deterministic: bool = True


def _set_reproducible(seed: int, deterministic: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Try to make CUDA/cuDNN behavior deterministic. This can slow training a bit.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _epoch_mse(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            bs = int(x.shape[0])
            mse_sum += float(loss.item()) * bs
            n += bs
    return mse_sum / max(1, n)


def train(
    processed_dir: str | os.PathLike = "data/processed",
    out_checkpoint: str | os.PathLike = "checkpoints/best_model.pt",
    out_loss_plot: str | os.PathLike = "outputs/figures/training_loss.png",
    cfg: TrainConfig = TrainConfig(),
) -> dict[str, list[float] | float]:
    """
    Train VAPredictor on QKDDataset in log-space.

    Saves:
    - best model weights to `checkpoints/best_model.pt` (best val MSE)
    - train/val loss curve plot to `outputs/figures/training_loss.png`
    """
    _set_reproducible(seed=cfg.seed, deterministic=cfg.deterministic)
    device = cfg.device
    model = VAPredictor(input_dim=4).to(device)

    train_ds = QKDDataset(split="train", processed_dir=processed_dir)
    val_ds = QKDDataset(split="val", processed_dir=processed_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
    )

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    out_checkpoint = Path(out_checkpoint)
    out_loss_plot = Path(out_loss_plot)
    os.makedirs(out_checkpoint.parent, exist_ok=True)
    os.makedirs(out_loss_plot.parent, exist_ok=True)

    for epoch in range(cfg.max_epochs):
        model.train()
        mse_sum = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            optimizer.step()

            bs = int(x.shape[0])
            mse_sum += float(loss.item()) * bs
            n += bs

        train_mse = mse_sum / max(1, n)
        val_mse = _epoch_mse(model, val_loader, device)

        train_losses.append(train_mse)
        val_losses.append(val_mse)

        scheduler.step(val_mse)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch:03d} train_mse={train_mse:.6g} val_mse={val_mse:.6g} lr={lr_now:.3g}")

        if val_mse < best_val - 1e-12:
            best_val = val_mse
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_mse": best_val,
                    "best_epoch": best_epoch,
                    "train_cfg": cfg.__dict__,
                },
                out_checkpoint,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"Early stopping at epoch={epoch} (best_epoch={best_epoch}, best_val={best_val:.6g})")
                break

    # Plot training curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train MSE")
    plt.plot(val_losses, label="val MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log-space)")
    plt.title("Training/validation loss")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_loss_plot, dpi=200)
    print(f"Saved: {out_loss_plot}")
    print(f"Saved best checkpoint: {out_checkpoint}")

    return {
        "best_val_mse": best_val,
        "best_epoch": float(best_epoch),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


if __name__ == "__main__":
    train()

