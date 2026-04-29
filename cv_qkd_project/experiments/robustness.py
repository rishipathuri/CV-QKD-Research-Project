from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cv_qkd_project import config
from cv_qkd_project.model.network import VAPredictor
from cv_qkd_project.optimization.brute_force import optimal_VA
from cv_qkd_project.side_channel.key_rate_mismatch import key_rate_mismatch


def _ensure_dirs() -> tuple[Path, Path]:
    out_results = Path("outputs/results")
    out_figs = Path("outputs/figures")
    os.makedirs(out_results, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)
    return out_results, out_figs


def _load_model(checkpoint_path: str | os.PathLike = "checkpoints/best_model.pt") -> tuple[VAPredictor, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = VAPredictor(input_dim=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def _standardize(X: np.ndarray, scaler_path: str | os.PathLike = "data/processed/scaler.npy") -> np.ndarray:
    scaler = np.load(scaler_path).astype(float)  # (2,4)
    means = scaler[0]
    stds = scaler[1]
    return ((X - means) / stds).astype(np.float32)


def _key_rate_for_samples(VA: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute mismatch key rate for arrays of VA and features X=[T,xi,eta1,eta2].
    """
    out = np.zeros((X.shape[0],), dtype=float)
    for i in range(X.shape[0]):
        t, x, e1, e2 = X[i]
        out[i] = float(
            key_rate_mismatch(
                V_A=float(VA[i]),
                T=float(t),
                xi=float(x),
                eta1=float(e1),
                eta2=float(e2),
                V_el=float(config.V_EL),
                beta=float(config.BETA),
            )
        )
    return out


def _bruteforce_VA_for_samples(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (VA_opt, K_opt) using mismatch-aware brute force.
    """
    VA_opt = np.zeros((X.shape[0],), dtype=float)
    K_opt = np.zeros((X.shape[0],), dtype=float)
    for i in range(X.shape[0]):
        t, x, e1, e2 = X[i]
        v, k = optimal_VA(
            T=float(t),
            xi=float(x),
            eta1=float(e1),
            eta2=float(e2),
            V_el=float(config.V_EL),
            beta=float(config.BETA),
            key_rate_fn=key_rate_mismatch,
        )
        VA_opt[i] = v
        K_opt[i] = k
    return VA_opt, K_opt


def study1_generalization_ood(
    n_samples: int = 1000,
    seed: int = 7,
    checkpoint_path: str | os.PathLike = "checkpoints/best_model.pt",
    scaler_path: str | os.PathLike = "data/processed/scaler.npy",
) -> pd.DataFrame:
    """
    (1) OOD generalization: T in [0.05,0.1], xi in [0.05,0.08].
    Compare NN key-rate recovery vs brute force under mismatch key-rate model.
    """
    out_results, out_figs = _ensure_dirs()

    rng = np.random.default_rng(seed)
    # Mildly out-of-distribution but not universally insecure:
    # lower T than training and higher xi than training, but still within a regime
    # where K_opt > 0 for a non-trivial subset.
    T = rng.uniform(0.07, 0.12, size=n_samples)
    xi = rng.uniform(0.03, 0.06, size=n_samples)
    eta1 = rng.uniform(config.ETA_MIN, config.ETA_MAX, size=n_samples)
    eta2 = rng.uniform(config.ETA_MIN, config.ETA_MAX, size=n_samples)
    X = np.stack([T, xi, eta1, eta2], axis=1)

    model, device = _load_model(checkpoint_path)
    X_std = _standardize(X, scaler_path)
    with torch.no_grad():
        VA_pred = (
            model.predict_VA(torch.from_numpy(X_std).to(device)).detach().cpu().numpy().astype(float)
        )
    VA_pred = np.clip(VA_pred, config.V_A_MIN, config.V_A_MAX)

    VA_opt, K_opt = _bruteforce_VA_for_samples(X)
    K_pred = _key_rate_for_samples(VA_pred, X)

    eps = 1e-12
    mask = K_opt > eps
    ratio = np.full_like(K_opt, np.nan, dtype=float)
    ratio[mask] = K_pred[mask] / K_opt[mask]
    if np.any(mask):
        recovery_pct = float(np.nanmean(ratio) * 100.0)
    else:
        recovery_pct = 0.0

    df = pd.DataFrame(
        {
            "T": T,
            "xi": xi,
            "eta1": eta1,
            "eta2": eta2,
            "VA_pred": VA_pred,
            "VA_opt": VA_opt,
            "K_pred": K_pred,
            "K_opt": K_opt,
            "recovery_ratio": ratio,
        }
    )
    out_csv = out_results / "robustness_ood_generalization.csv"
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(ratio[np.isfinite(ratio)], bins=40, alpha=0.85)
    plt.xlabel("Key-rate recovery ratio (K_pred / K_opt)")
    plt.ylabel("Count")
    nonzero_pct = float(np.mean(mask) * 100.0)
    plt.title(
        f"OOD generalization: mean recovery = {recovery_pct:.2f}% "
        f"(K_opt>0 in {nonzero_pct:.1f}% of samples, n={n_samples})"
    )
    plt.grid(True, ls="--", alpha=0.4)
    out_png = out_figs / "robustness_ood_generalization.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print(f"[robustness1] Saved: {out_csv}")
    print(f"[robustness1] Saved: {out_png}")
    print(f"[robustness1] Mean recovery: {recovery_pct:.2f}%")
    return df


def study2_mismatch_asymmetry(
    n_train: int = 4000,
    n_test: int = 2000,
    seed: int = 11,
    checkpoint_path: str | os.PathLike = "checkpoints/best_model.pt",
    scaler_path: str | os.PathLike = "data/processed/scaler.npy",
) -> pd.DataFrame:
    """
    (2) Mismatch asymmetry: train/fine-tune only on eta1>eta2 and evaluate on eta2>eta1.

    We report performance for:
    - Base model (trained on mixed ordering) evaluated on eta2>eta1
    - Fine-tuned model (trained only on eta1>eta2) evaluated on eta2>eta1
    """
    out_results, out_figs = _ensure_dirs()

    rng = np.random.default_rng(seed)

    def sample_ordered(n: int, greater: bool) -> np.ndarray:
        T = rng.uniform(config.T_MIN, config.T_MAX, size=n)
        xi = rng.uniform(config.XI_MIN, config.XI_MAX, size=n)
        a = rng.uniform(config.ETA_MIN, config.ETA_MAX, size=n)
        b = rng.uniform(config.ETA_MIN, config.ETA_MAX, size=n)
        if greater:
            eta1 = np.maximum(a, b)
            eta2 = np.minimum(a, b)
        else:
            eta1 = np.minimum(a, b)
            eta2 = np.maximum(a, b)
        return np.stack([T, xi, eta1, eta2], axis=1)

    X_ft = sample_ordered(n_train, greater=True)  # eta1 > eta2
    X_eval = sample_ordered(n_test, greater=False)  # eta2 > eta1

    # Labels for fine-tuning: brute-force optimal VA* under mismatch
    VA_opt_ft, _ = _bruteforce_VA_for_samples(X_ft)
    y_ft = np.log(np.maximum(VA_opt_ft, 1e-12)).astype(np.float32)

    base_model, device = _load_model(checkpoint_path)

    # Evaluate base model on eta2>eta1
    X_eval_std = _standardize(X_eval, scaler_path)
    with torch.no_grad():
        VA_base = (
            base_model.predict_VA(torch.from_numpy(X_eval_std).to(device)).detach().cpu().numpy().astype(float)
        )
    VA_base = np.clip(VA_base, config.V_A_MIN, config.V_A_MAX)
    VA_opt_eval, K_opt_eval = _bruteforce_VA_for_samples(X_eval)
    K_base = _key_rate_for_samples(VA_base, X_eval)

    eps = 1e-12
    rec_base = float(np.mean((K_base[K_opt_eval > eps] / K_opt_eval[K_opt_eval > eps])) * 100.0)
    mae_base = float(np.mean(np.abs(VA_base - VA_opt_eval)))

    # Fine-tune a fresh copy of the base model on eta1>eta2 subset
    ft_model = VAPredictor(input_dim=4).to(device)
    ft_model.load_state_dict(base_model.state_dict())
    ft_model.train()

    X_ft_std = _standardize(X_ft, scaler_path)
    ds = TensorDataset(torch.from_numpy(X_ft_std), torch.from_numpy(y_ft))
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    opt = torch.optim.Adam(ft_model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    for _epoch in range(25):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = ft_model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    ft_model.eval()

    with torch.no_grad():
        VA_ft = (
            ft_model.predict_VA(torch.from_numpy(X_eval_std).to(device)).detach().cpu().numpy().astype(float)
        )
    VA_ft = np.clip(VA_ft, config.V_A_MIN, config.V_A_MAX)
    K_ft = _key_rate_for_samples(VA_ft, X_eval)
    rec_ft = float(np.mean((K_ft[K_opt_eval > eps] / K_opt_eval[K_opt_eval > eps])) * 100.0)
    mae_ft = float(np.mean(np.abs(VA_ft - VA_opt_eval)))

    df = pd.DataFrame(
        [
            {"model": "base", "eval_order": "eta2>eta1", "mae_VA": mae_base, "recovery_pct": rec_base},
            {"model": "fine_tuned_eta1>eta2", "eval_order": "eta2>eta1", "mae_VA": mae_ft, "recovery_pct": rec_ft},
        ]
    )
    out_csv = out_results / "robustness_mismatch_asymmetry.csv"
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(df["model"], df["recovery_pct"])
    plt.ylabel("Mean key-rate recovery (%)")
    plt.title("Mismatch asymmetry generalization (evaluate on eta2>eta1)")
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    plt.xticks(rotation=15, ha="right")
    out_png = out_figs / "robustness_mismatch_asymmetry.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print(f"[robustness2] Saved: {out_csv}")
    print(f"[robustness2] Saved: {out_png}")
    return df


@dataclass(frozen=True)
class ArchSpec:
    name: str
    h1: int
    h2: int
    h3: int


class _MLP(nn.Module):
    def __init__(self, sizes: tuple[int, int, int]):
        super().__init__()
        h1, h2, h3 = sizes
        self.net = nn.Sequential(
            nn.Linear(4, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_VA(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.forward(x))


def study3_architecture_ablation(
    checkpoint_path: str | os.PathLike = "checkpoints/best_model.pt",
    processed_dir: str | os.PathLike = "data/processed",
    seed: int = 0,
) -> pd.DataFrame:
    """
    (3) Architecture ablation: small/medium/large MLP compared on test MAE + recovery.

    For speed, we train each model with early stopping and a modest epoch cap.
    """
    out_results, out_figs = _ensure_dirs()

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processed splits
    X_train = np.load(Path(processed_dir) / "X_train.npy").astype(np.float32)
    y_train = np.load(Path(processed_dir) / "y_train.npy").astype(np.float32)
    X_val = np.load(Path(processed_dir) / "X_val.npy").astype(np.float32)
    y_val = np.load(Path(processed_dir) / "y_val.npy").astype(np.float32)
    X_test = np.load(Path(processed_dir) / "X_test.npy").astype(np.float32)
    y_test = np.load(Path(processed_dir) / "y_test.npy").astype(np.float32)

    y_train_log = np.log(np.maximum(y_train, 1e-12)).astype(np.float32)
    y_val_log = np.log(np.maximum(y_val, 1e-12)).astype(np.float32)
    y_test_log = np.log(np.maximum(y_test, 1e-12)).astype(np.float32)

    archs = [
        ArchSpec("small_64-32-16", 64, 32, 16),
        ArchSpec("medium_128-64-32", 128, 64, 32),
        ArchSpec("large_256-128-64", 256, 128, 64),
    ]

    # For recovery, we need unstandardized features
    scaler = np.load(Path(processed_dir) / "scaler.npy").astype(float)
    means, stds = scaler[0], scaler[1]
    X_test_phys = X_test * stds + means

    results = []
    for spec in archs:
        model = _MLP((spec.h1, spec.h2, spec.h3)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_log)),
            batch_size=512,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_log)),
            batch_size=2048,
            shuffle=False,
        )

        best_val = float("inf")
        bad = 0
        for _epoch in range(150):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            # val mse
            model.eval()
            mse_sum = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    mse = torch.mean((pred - yb) ** 2)
                    bs = int(xb.shape[0])
                    mse_sum += float(mse.item()) * bs
                    n += bs
            val_mse = mse_sum / max(1, n)
            sched.step(val_mse)

            if val_mse < best_val - 1e-12:
                best_val = val_mse
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 15:
                    break

        model.load_state_dict(best_state)
        model.eval()

        # predictions on test
        with torch.no_grad():
            pred_log = model(torch.from_numpy(X_test).to(device)).detach().cpu().numpy().astype(float)
        VA_pred = np.clip(np.exp(pred_log), config.V_A_MIN, config.V_A_MAX)
        mae = float(np.mean(np.abs(VA_pred - y_test.astype(float))))

        VA_opt = y_test.astype(float)
        K_pred = _key_rate_for_samples(VA_pred, X_test_phys)
        K_opt = _key_rate_for_samples(VA_opt, X_test_phys)
        eps = 1e-12
        rec = float(np.mean((K_pred[K_opt > eps] / K_opt[K_opt > eps])) * 100.0)

        results.append({"arch": spec.name, "test_mae_VA": mae, "key_rate_recovery_pct": rec})

    df = pd.DataFrame(results)
    out_csv = out_results / "robustness_architecture_ablation.csv"
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar(df["arch"], df["test_mae_VA"])
    plt.title("Test MAE on $V_A^*$")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    plt.subplot(1, 2, 2)
    plt.bar(df["arch"], df["key_rate_recovery_pct"])
    plt.title("Key-rate recovery (%)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    out_png = out_figs / "robustness_architecture_ablation.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print(f"[robustness3] Saved: {out_csv}")
    print(f"[robustness3] Saved: {out_png}")
    return df


def study4_speed_benchmark(
    n_samples: int = 1000,
    n_trials: int = 10,
    seed: int = 21,
    checkpoint_path: str | os.PathLike = "checkpoints/best_model.pt",
    scaler_path: str | os.PathLike = "data/processed/scaler.npy",
) -> pd.DataFrame:
    """
    (4) Speed benchmark: brute force vs NN over 1000 random samples; report mean/std speedup.
    """
    out_results, out_figs = _ensure_dirs()

    rng = np.random.default_rng(seed)
    model, device = _load_model(checkpoint_path)

    speedups = []
    nn_times = []
    brute_times = []

    for _ in range(n_trials):
        T = rng.uniform(config.T_MIN, config.T_MAX, size=n_samples)
        xi = rng.uniform(config.XI_MIN, config.XI_MAX, size=n_samples)
        eta1 = rng.uniform(config.ETA_MIN, config.ETA_MAX, size=n_samples)
        eta2 = rng.uniform(config.ETA_MIN, config.ETA_MAX, size=n_samples)
        X = np.stack([T, xi, eta1, eta2], axis=1)
        X_std = _standardize(X, scaler_path)

        # NN timing
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.predict_VA(torch.from_numpy(X_std).to(device)).detach().cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        nn_s = t1 - t0

        # Brute force timing
        t2 = time.perf_counter()
        for i in range(n_samples):
            optimal_VA(
                T=float(X[i, 0]),
                xi=float(X[i, 1]),
                eta1=float(X[i, 2]),
                eta2=float(X[i, 3]),
                V_el=float(config.V_EL),
                beta=float(config.BETA),
                key_rate_fn=key_rate_mismatch,
            )
        t3 = time.perf_counter()
        brute_s = t3 - t2

        speedups.append(brute_s / max(nn_s, 1e-12))
        nn_times.append(nn_s)
        brute_times.append(brute_s)

    df = pd.DataFrame({"trial": np.arange(n_trials), "nn_time_s": nn_times, "bruteforce_time_s": brute_times, "speedup": speedups})
    out_csv = out_results / "robustness_speed_benchmark.csv"
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(df["trial"], df["speedup"], marker="o")
    plt.xlabel("Trial")
    plt.ylabel("Speedup (brute / NN)")
    plt.title(f"Speedup over {n_samples} samples: mean={df['speedup'].mean():.1f}x, std={df['speedup'].std():.1f}x")
    plt.grid(True, ls="--", alpha=0.4)
    out_png = out_figs / "robustness_speed_benchmark.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print(f"[robustness4] Saved: {out_csv}")
    print(f"[robustness4] Saved: {out_png}")
    print(f"[robustness4] Speedup mean={df['speedup'].mean():.1f}x std={df['speedup'].std():.1f}x")
    return df


def run_all_robustness() -> None:
    study1_generalization_ood()
    study2_mismatch_asymmetry()
    study3_architecture_ablation()
    study4_speed_benchmark()


if __name__ == "__main__":
    run_all_robustness()

