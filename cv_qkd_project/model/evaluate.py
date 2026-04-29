from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cv_qkd_project import config
from cv_qkd_project.model.network import VAPredictor
from cv_qkd_project.optimization.brute_force import optimal_VA
from cv_qkd_project.side_channel.key_rate_mismatch import key_rate_mismatch


def _load_processed(split: str, processed_dir: str | os.PathLike = "data/processed") -> tuple[np.ndarray, np.ndarray]:
    processed_dir = Path(processed_dir)
    X = np.load(processed_dir / f"X_{split}.npy").astype(np.float32)
    y = np.load(processed_dir / f"y_{split}.npy").astype(np.float32)
    return X, y


def _invert_standardize(X_std: np.ndarray, processed_dir: str | os.PathLike = "data/processed") -> np.ndarray:
    scaler = np.load(Path(processed_dir) / "scaler.npy").astype(float)  # (2,4): means, stds
    means = scaler[0]
    stds = scaler[1]
    return X_std * stds + means


def evaluate(
    checkpoint_path: str | os.PathLike = "checkpoints/best_model.pt",
    processed_dir: str | os.PathLike = "data/processed",
    out_parity_plot: str | os.PathLike = "outputs/figures/parity_plot.png",
    batch_size: int = 2048,
) -> dict[str, float]:
    """
    Evaluate best checkpoint on the test set.

    Reports:
    - MAE on V_A* (in linear space)
    - Mean key-rate recovery ratio: mean( K(VA_pred) / K(VA_opt) ) as %
    - Inference speed vs brute-force optimizer over 1000 samples
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = VAPredictor(input_dim=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    X_test_std, y_test = _load_processed("test", processed_dir=processed_dir)
    X_test = _invert_standardize(X_test_std, processed_dir=processed_dir)  # columns: T, xi, eta1, eta2

    # Predict in batches
    loader = DataLoader(torch.from_numpy(X_test_std), batch_size=batch_size, shuffle=False)
    preds_log = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds_log.append(model(xb).detach().cpu().numpy())
    preds_log = np.concatenate(preds_log, axis=0).astype(np.float64)

    VA_pred = np.exp(preds_log)
    VA_true = y_test.astype(np.float64)

    # Clamp to grid bounds to avoid ridiculous extrapolation
    VA_pred = np.clip(VA_pred, config.V_A_MIN, config.V_A_MAX)

    mae = float(np.mean(np.abs(VA_pred - VA_true)))

    # Key rate recovery: compute K at predicted VA and at (brute-force) optimal VA label.
    T = X_test[:, 0]
    xi = X_test[:, 1]
    eta1 = X_test[:, 2]
    eta2 = X_test[:, 3]

    K_pred = np.zeros_like(VA_pred, dtype=float)
    K_opt = np.zeros_like(VA_true, dtype=float)

    for i in range(VA_pred.shape[0]):
        K_pred[i] = float(
            key_rate_mismatch(
                V_A=float(VA_pred[i]),
                T=float(T[i]),
                xi=float(xi[i]),
                eta1=float(eta1[i]),
                eta2=float(eta2[i]),
                V_el=config.V_EL,
                beta=config.BETA,
            )
        )
        K_opt[i] = float(
            key_rate_mismatch(
                V_A=float(VA_true[i]),
                T=float(T[i]),
                xi=float(xi[i]),
                eta1=float(eta1[i]),
                eta2=float(eta2[i]),
                V_el=config.V_EL,
                beta=config.BETA,
            )
        )

    eps = 1e-12
    mask = K_opt > eps
    recovery_ratio = float(np.mean((K_pred[mask] / K_opt[mask])) * 100.0) if np.any(mask) else float("nan")

    # Parity plot
    out_parity_plot = Path(out_parity_plot)
    os.makedirs(out_parity_plot.parent, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(VA_true, VA_pred, s=6, alpha=0.4)
    mn = float(min(VA_true.min(), VA_pred.min()))
    mx = float(max(VA_true.max(), VA_pred.max()))
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"True $V_A^*$ (SNU)")
    plt.ylabel(r"Predicted $V_A^*$ (SNU)")
    plt.title("Parity plot: $V_A^*$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_parity_plot, dpi=200)
    print(f"Saved: {out_parity_plot}")

    # Speed benchmark: NN inference vs brute force over 1000 samples
    n_bench = min(1000, X_test_std.shape[0])
    X_bench_std = torch.from_numpy(X_test_std[:n_bench]).to(device)
    X_bench = X_test[:n_bench]

    # NN inference timing
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.predict_VA(X_bench_std).detach().cpu().numpy()
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()
    nn_s = t1 - t0

    # Brute-force timing (uses the full V_A grid each sample)
    t2 = time.perf_counter()
    for i in range(n_bench):
        t, x, e1, e2 = X_bench[i]
        optimal_VA(
            T=float(t),
            xi=float(x),
            eta1=float(e1),
            eta2=float(e2),
            V_el=config.V_EL,
            beta=config.BETA,
            key_rate_fn=key_rate_mismatch,
        )
    t3 = time.perf_counter()
    brute_s = t3 - t2

    speedup = float(brute_s / max(nn_s, 1e-12))

    print(f"Test MAE(V_A*) = {mae:.6g} SNU")
    print(f"Mean key-rate recovery ratio = {recovery_ratio:.3f}% (over {int(mask.sum())} nonzero-opt samples)")
    print(f"Inference benchmark over {n_bench} samples:")
    print(f"- NN time: {nn_s:.4f} s ({n_bench/nn_s:.1f} samples/s)")
    print(f"- Brute-force time: {brute_s:.4f} s ({n_bench/brute_s:.2f} samples/s)")
    print(f"- Speedup: {speedup:.1f}x")

    return {
        "mae_VA": mae,
        "key_rate_recovery_pct": recovery_ratio,
        "nn_time_s_1000": float(nn_s),
        "bruteforce_time_s_1000": float(brute_s),
        "speedup_factor": speedup,
    }


if __name__ == "__main__":
    evaluate()

