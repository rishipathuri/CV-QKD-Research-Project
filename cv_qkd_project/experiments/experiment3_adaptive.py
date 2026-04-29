from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cv_qkd_project import config
from cv_qkd_project.model.network import VAPredictor
from cv_qkd_project.side_channel.key_rate_mismatch import key_rate_mismatch


def _standardize_features(X: np.ndarray, scaler_path: str | os.PathLike = "data/processed/scaler.npy") -> np.ndarray:
    scaler = np.load(scaler_path).astype(float)  # (2,4): means, stds
    means = scaler[0]
    stds = scaler[1]
    return (X - means) / stds


def run_experiment3_adaptive(
    out_csv: str | os.PathLike = "outputs/results/experiment3.csv",
    xi: float = 0.01,
    eta1: float = 0.8,
    eta2: float = 0.6,
    n_steps: int = 50,
    checkpoint_path: str | os.PathLike = "checkpoints/best_model.pt",
    scaler_path: str | os.PathLike = "data/processed/scaler.npy",
) -> pd.DataFrame:
    """
    Adaptive system: uses trained NN to predict V_A from (T, xi, eta1, eta2),
    then evaluates actual key rate using key_rate_mismatch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = VAPredictor(input_dim=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    Ts = np.linspace(config.T_MIN, config.T_MAX, n_steps)
    rows = []

    for T in Ts:
        feats = np.array([[float(T), float(xi), float(eta1), float(eta2)]], dtype=float)
        feats_std = _standardize_features(feats, scaler_path=scaler_path).astype(np.float32)
        x = torch.from_numpy(feats_std).to(device)
        with torch.no_grad():
            V_A_pred = float(model.predict_VA(x).detach().cpu().numpy().reshape(-1)[0])
        V_A_pred = float(np.clip(V_A_pred, config.V_A_MIN, config.V_A_MAX))

        K_actual = float(
            key_rate_mismatch(
                V_A=V_A_pred,
                T=float(T),
                xi=float(xi),
                eta1=float(eta1),
                eta2=float(eta2),
                V_el=float(config.V_EL),
                beta=float(config.BETA),
            )
        )

        rows.append(
            {
                "T": float(T),
                "xi": float(xi),
                "eta1": float(eta1),
                "eta2": float(eta2),
                "V_A_pred": float(V_A_pred),
                "K_actual": float(K_actual),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    os.makedirs(out_csv.parent, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df


if __name__ == "__main__":
    run_experiment3_adaptive()

