from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from cv_qkd_project import config
from cv_qkd_project.optimization.brute_force import optimal_VA
from cv_qkd_project.physics.key_rate import key_rate
from cv_qkd_project.side_channel.key_rate_mismatch import key_rate_mismatch
from cv_qkd_project.side_channel.mismatch import effective_eta


def run_experiment2_naive(
    out_csv: str | os.PathLike = "outputs/results/experiment2.csv",
    xi: float = 0.01,
    eta1: float = 0.8,
    eta2: float = 0.6,
    n_steps: int = 50,
) -> pd.DataFrame:
    """
    Naive system: chooses V_A as if detection were ideal (standard key_rate),
    but the actual key rate is evaluated with the true mismatch (eta1, eta2).
    """
    Ts = np.linspace(config.T_MIN, config.T_MAX, n_steps)
    rows = []

    # Assumption for "unaware of mismatch": the system uses an average efficiency
    # but does NOT account for the additional mismatch-induced noise.
    eta_naive = float(effective_eta(eta1, eta2))

    for T in Ts:
        V_A_naive, _ = optimal_VA(
            T=float(T),
            xi=float(xi),
            eta=eta_naive,
            V_el=float(config.V_EL),
            beta=float(config.BETA),
            key_rate_fn=key_rate,
        )
        K_actual = float(
            key_rate_mismatch(
                V_A=V_A_naive,
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
                "V_A_naive": float(V_A_naive),
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
    run_experiment2_naive()

