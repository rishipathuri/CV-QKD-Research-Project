from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cv_qkd_project import config
from cv_qkd_project.physics.key_rate import key_rate
from cv_qkd_project.side_channel.mismatch import effective_eta, mismatch_noise


def key_rate_mismatch(V_A, T, xi, eta1, eta2, V_el, beta):
    """
    Key rate under detector efficiency mismatch via an effective-noise mapping.

    This wraps the standard key-rate pipeline by converting a two-arm homodyne
    mismatch (eta1 != eta2) into:
    - an effective efficiency eta_eff = (eta1 + eta2)/2
    - an additive mismatch-induced noise term xi_mismatch (SNU)

    To reuse the existing `physics.key_rate.key_rate` implementation (which
    expects (eta, V_el) as detector parameters), we fold the mismatch noise into
    an *effective* electronic noise:

        V_el_eff = V_el + eta_eff * xi_mismatch

    so that the term V_el_eff/eta_eff in the mutual-information noise budget
    becomes V_el/eta_eff + xi_mismatch.

    Parameters
    ----------
    V_A : float or np.ndarray
        Alice modulation variance (SNU). Vectorized over NumPy arrays.
    T : float
        Channel transmittance (dimensionless).
    xi : float
        Channel excess noise (SNU, input-referred).
    eta1, eta2 : float
        Efficiencies of Bob's two homodyne detector arms (dimensionless).
    V_el : float
        Baseline electronic noise (SNU).
    beta : float
        Reconciliation efficiency (dimensionless).

    Returns
    -------
    np.ndarray
        Key rate under mismatch (bits/use), clipped at zero (same as standard).
    """
    eta_eff = float(effective_eta(eta1, eta2))
    xi_m = float(mismatch_noise(eta1, eta2))
    V_el_eff = V_el + eta_eff * xi_m
    return key_rate(V_A=V_A, T=T, xi=xi, eta=eta_eff, V_el=V_el_eff, beta=beta)


if __name__ == "__main__":
    outdir = Path("outputs/figures")
    os.makedirs(outdir, exist_ok=True)

    # Fixed parameters for validation plots
    T = 0.5
    xi = 0.01
    V_el = 0.01
    beta = 0.95
    eta_mean = 0.6

    # (1) Key rate vs delta_eta at fixed V_A
    V_A_fixed = 10.0
    deltas = np.linspace(0.0, 0.3, 61)
    Ks = np.zeros_like(deltas)
    for i, d in enumerate(deltas):
        eta1 = eta_mean + d / 2.0
        eta2 = eta_mean - d / 2.0
        Ks[i] = float(key_rate_mismatch(V_A=V_A_fixed, T=T, xi=xi, eta1=eta1, eta2=eta2, V_el=V_el, beta=beta))

    plt.figure(figsize=(8, 5))
    plt.plot(deltas, Ks, marker="o", ms=3)
    plt.xlabel(r"Mismatch $\Delta\eta = |\eta_1-\eta_2|$")
    plt.ylabel(r"Key rate $K$ (bits/use)")
    plt.title(rf"Key rate degradation vs mismatch (T={T}, xi={xi}, V_A={V_A_fixed})")
    plt.grid(True, ls="--", alpha=0.4)
    path1 = outdir / "key_rate_vs_delta_eta.png"
    plt.tight_layout()
    plt.savefig(path1, dpi=200)
    print(f"Saved: {path1}")

    # (2) Key rate vs V_A under several mismatch levels
    V_A_grid = config.V_A_GRID
    mismatch_levels = [0.0, 0.1, 0.2]

    plt.figure(figsize=(8, 5))
    for d in mismatch_levels:
        eta1 = eta_mean + d / 2.0
        eta2 = eta_mean - d / 2.0
        K = key_rate_mismatch(V_A=V_A_grid, T=T, xi=xi, eta1=eta1, eta2=eta2, V_el=V_el, beta=beta)
        plt.plot(V_A_grid, K, label=rf"$\Delta\eta={d:.1f}$")

        # Print where the optimum shifts for each mismatch
        idx = int(np.argmax(K))
        print(f"delta_eta={d:.1f}: V_A*={V_A_grid[idx]:.4g}, K_max={float(K[idx]):.4g}")

    plt.xscale("log")
    plt.xlabel(r"Modulation variance $V_A$ (SNU)")
    plt.ylabel(r"Key rate $K$ (bits/use)")
    plt.title(rf"Key rate vs $V_A$ under mismatch (T={T}, xi={xi}, eta_mean={eta_mean})")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    path2 = outdir / "key_rate_vs_VA_mismatch_levels.png"
    plt.tight_layout()
    plt.savefig(path2, dpi=200)
    print(f"Saved: {path2}")

