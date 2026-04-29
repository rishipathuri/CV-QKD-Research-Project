from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cv_qkd_project import config
from cv_qkd_project.optimization.brute_force import optimal_VA
from cv_qkd_project.physics.key_rate import key_rate


def _ensure_outdir() -> Path:
    outdir = Path("outputs/figures")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman rank correlation computed via ranks (no SciPy dependency here).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    return float(np.sum(rx * ry) / denom) if denom > 0 else float("nan")


def plot_key_rate_unimodal_examples(eta: float, V_el: float, beta: float) -> None:
    """
    (1) Key rate vs V_A for several channel conditions.
    """
    outdir = _ensure_outdir()
    V_A = config.V_A_GRID

    # Five conditions spanning "good" to "worse" channels.
    cases = [
        (0.9, 0.005),
        (0.8, 0.010),
        (0.6, 0.010),
        (0.4, 0.015),
        (0.2, 0.020),
    ]

    plt.figure(figsize=(8, 5))
    print("\n(1) Unimodal peak examples (key rate vs V_A)")
    for T, xi in cases:
        K = key_rate(V_A=V_A, T=T, xi=xi, eta=eta, V_el=V_el, beta=beta)
        v_star, k_star = optimal_VA(T=T, xi=xi, eta=eta, V_el=V_el, beta=beta, key_rate_fn=key_rate)

        # "Peak" sanity check (unimodality-ish): peak is not on extreme endpoints.
        idx_star = int(np.argmax(K))
        peak_not_endpoint = 0 < idx_star < (len(V_A) - 1)

        print(f"- T={T:.2f}, xi={xi:.3f}: V_A*={v_star:.4g}, K_max={k_star:.4g}, peak_not_endpoint={peak_not_endpoint}")
        plt.plot(V_A, K, label=f"T={T:.2f}, xi={xi:.3f}")
        plt.scatter([v_star], [k_star], s=25)

    plt.xscale("log")
    plt.xlabel(r"$V_A$ (SNU)")
    plt.ylabel(r"$K$ (bits/use)")
    plt.title("Key rate vs modulation variance (examples)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    path = outdir / "optimizer_validation_unimodal.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print(f"Saved: {path}")


def plot_optimal_VA_vs_T(xi: float, eta: float, V_el: float, beta: float) -> None:
    """
    (2) Optimal V_A as a function of T for fixed xi.
    """
    outdir = _ensure_outdir()

    Ts = np.linspace(config.T_MIN, config.T_MAX, 41)
    V_stars = np.zeros_like(Ts)
    K_stars = np.zeros_like(Ts)

    print("\n(2) Optimal V_A vs T (xi fixed)")
    for i, T in enumerate(Ts):
        v_star, k_star = optimal_VA(T=float(T), xi=xi, eta=eta, V_el=V_el, beta=beta, key_rate_fn=key_rate)
        V_stars[i] = v_star
        K_stars[i] = k_star

    rho = _spearman_r(Ts, V_stars)
    monotone_decreasing_fraction = float(np.mean(np.diff(V_stars) <= 0))

    trend = "decreasing" if rho < 0 else "increasing"
    print(f"- Spearman corr(T, V_A*) = {rho:.3f} (trend: {trend})")
    print(f"- Fraction of steps with V_A* decreasing as T increases: {monotone_decreasing_fraction:.2%}")
    print(f"- Sample endpoints: T={Ts[0]:.2f} -> V_A*={V_stars[0]:.4g}, T={Ts[-1]:.2f} -> V_A*={V_stars[-1]:.4g}")

    plt.figure(figsize=(8, 5))
    plt.plot(Ts, V_stars, marker="o", ms=3)
    plt.xlabel("Transmittance T")
    plt.ylabel(r"Optimal $V_A^*$ (SNU)")
    plt.title(rf"Optimal $V_A^*$ vs T (xi={xi:.3f})")
    plt.grid(True, ls="--", alpha=0.4)
    path = outdir / "optimizer_validation_optimal_VA_vs_T.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print(f"Saved: {path}")


def plot_max_key_rate_vs_distance(
    xi: float,
    eta: float,
    V_el: float,
    beta: float,
    alpha_db_per_km: float = 0.2,
    L_max_km: float = 80.0,
) -> None:
    """
    (3) Maximum key rate vs fiber distance using T = 10^(-alpha*L/10).
    """
    outdir = _ensure_outdir()

    Ls = np.linspace(0.0, L_max_km, 81)
    Ts = 10 ** (-(alpha_db_per_km * Ls) / 10.0)

    K_max = np.zeros_like(Ls)
    V_star = np.zeros_like(Ls)

    print("\n(3) Max key rate vs distance")
    for i, (L, T) in enumerate(zip(Ls, Ts)):
        v_star, k_star = optimal_VA(T=float(T), xi=xi, eta=eta, V_el=V_el, beta=beta, key_rate_fn=key_rate)
        V_star[i] = v_star
        K_max[i] = k_star

    # Find the first distance where the key rate is (numerically) zero.
    eps = 1e-6
    zero_idx = np.argmax(K_max <= eps) if np.any(K_max <= eps) else None
    if zero_idx is not None and np.any(K_max <= eps):
        L_zero = float(Ls[zero_idx])
        print(f"- First L where K_max <= {eps}: {L_zero:.1f} km (T={Ts[zero_idx]:.3g})")
    else:
        print(f"- K_max stayed above {eps} out to {L_max_km:.1f} km")

    print(f"- K_max(0 km)={K_max[0]:.4g}, K_max({L_max_km:.0f} km)={K_max[-1]:.4g}")

    plt.figure(figsize=(8, 5))
    plt.plot(Ls, K_max, marker="o", ms=3)
    plt.xlabel("Fiber distance L (km)")
    plt.ylabel(r"Max key rate $K_{\max}$ (bits/use)")
    plt.title(
        rf"Max key rate vs distance (alpha={alpha_db_per_km:.1f} dB/km, xi={xi:.3f}, eta={eta:.2f})"
    )
    plt.grid(True, ls="--", alpha=0.4)
    path = outdir / "optimizer_validation_key_rate_vs_distance.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    # Fixed detector/reconciliation settings (as used in earlier validation).
    eta = 0.6
    V_el = 0.01
    beta = 0.95

    plot_key_rate_unimodal_examples(eta=eta, V_el=V_el, beta=beta)
    plot_optimal_VA_vs_T(xi=0.01, eta=eta, V_el=V_el, beta=beta)

    # For the distance sweep, use a more ideal receiver to better illustrate the
    # long-distance cutoff.
    plot_max_key_rate_vs_distance(
        xi=0.01, eta=0.95, V_el=V_el, beta=beta, alpha_db_per_km=0.2, L_max_km=80.0
    )

