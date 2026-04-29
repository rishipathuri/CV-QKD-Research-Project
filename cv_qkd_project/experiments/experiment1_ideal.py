from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from cv_qkd_project import config
from cv_qkd_project.optimization.brute_force import optimal_VA
from cv_qkd_project.physics.key_rate import key_rate


def run_experiment1_ideal(
    out_csv: str | os.PathLike = "outputs/results/experiment1.csv",
    xi: float = 0.01,
    eta: float = 0.8,
    n_steps: int = 50,
) -> pd.DataFrame:
    """
    Ideal upper bound: eta1=eta2=eta, optimize using standard key_rate.
    """
    Ts = np.linspace(config.T_MIN, config.T_MAX, n_steps)
    rows = []
    for T in Ts:
        v_star, k_star = optimal_VA(
            T=float(T),
            xi=float(xi),
            eta=float(eta),
            V_el=float(config.V_EL),
            beta=float(config.BETA),
            key_rate_fn=key_rate,
        )
        rows.append({"T": float(T), "xi": float(xi), "eta": float(eta), "V_A_star": v_star, "K_max": k_star})

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    os.makedirs(out_csv.parent, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df


if __name__ == "__main__":
    run_experiment1_ideal()

