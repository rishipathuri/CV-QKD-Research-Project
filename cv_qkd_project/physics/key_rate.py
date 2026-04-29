from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cv_qkd_project.physics.covariance import (
    symplectic_eigenvalues,
    symplectic_eigenvalues_eve,
    build_covariance_matrix,
)


def g(x):
    """
    Bosonic entropy function g(x) in bits for symplectic eigenvalue x (SNU).

    In Gaussian CV-QKD security analysis, the von Neumann entropy of a
    single-mode thermal state can be written in terms of the symplectic
    eigenvalue x >= 1 (shot-noise units):

        g(x) = ((x+1)/2) * log2((x+1)/2) - ((x-1)/2) * log2((x-1)/2)

    Edge case:
    - x = 1 corresponds to the vacuum state and yields g(1) = 0.

    Parameters
    ----------
    x : float or np.ndarray
        Symplectic eigenvalue(s) in SNU. Physical covariance matrices yield
        x >= 1. The function supports NumPy arrays and broadcasts normally.

    Returns
    -------
    np.ndarray
        g(x) in bits, with the same broadcasted shape as x.
    """
    x = np.asarray(x, dtype=float)
    # Guard against tiny numerical excursions below 1
    x = np.maximum(x, 1.0)

    a = (x + 1.0) / 2.0
    b = (x - 1.0) / 2.0

    term_a = a * np.log2(a)

    # Compute b*log2(b) only where b>0 to avoid log2(0) warnings.
    term_b = np.zeros_like(b, dtype=float)
    mask = b > 0
    term_b[mask] = b[mask] * np.log2(b[mask])

    return term_a - term_b


def mutual_information(V_A, T, xi, eta, V_el):
    """
    Mutual information I_AB for GMCS CV-QKD with homodyne detection (bits/use).

    Uses the common Shannon formula:

        I_AB = 0.5 * log2(1 + SNR)

    with signal-to-noise ratio:

        SNR = (T * V_A) / (1 + T*xi + (1-eta)/eta + V_el/eta)

    All quantities are in shot-noise units (SNU) unless dimensionless.

    Parameters
    ----------
    V_A : float or np.ndarray
        Alice modulation variance (SNU). Supports NumPy arrays for vectorized
        evaluation over a V_A grid.
    T : float
        Channel transmittance (dimensionless).
    xi : float or np.ndarray
        Channel excess noise (SNU, input-referred). Can be scalar or an array
        broadcastable to the shape of V_A.
    eta : float
        Bob detection efficiency (dimensionless).
    V_el : float
        Bob electronic noise (SNU).

    Returns
    -------
    np.ndarray
        I_AB in bits per channel use, broadcasted to the shape of V_A.
    """
    V_A = np.asarray(V_A, dtype=float)
    xi = np.asarray(xi, dtype=float)
    denom = 1.0 + T * xi + (1.0 - eta) / eta + V_el / eta
    snr = (T * V_A) / denom
    return 0.5 * np.log2(1.0 + snr)


def holevo_bound(V_A, T, xi, eta, V_el):
    """
    Holevo bound chi_BE for reverse reconciliation (bits/use).

    Computes:

        chi_BE = g(lambda1) + g(lambda2) - g(lambda3) - g(lambda4)

    where (lambda1, lambda2) are the symplectic eigenvalues of the joint
    Alice–Bob covariance matrix, and (lambda3, lambda4) are the eigenvalues of
    the conditional state after Bob's homodyne measurement, as provided by
    `symplectic_eigenvalues_eve`.

    Supports vectorized evaluation over V_A by looping over the covariance
    matrix construction per point (since each CM is 4x4).

    Parameters
    ----------
    V_A : float or np.ndarray
        Alice modulation variance (SNU).
    T, eta, V_el : float
        Channel/detector parameters in the same conventions as elsewhere (SNU).
    xi : float or np.ndarray
        Channel excess noise (SNU, input-referred). Can be scalar or an array
        broadcastable to the shape of V_A.

    Returns
    -------
    np.ndarray
        chi_BE in bits per channel use, with the same shape as V_A.
    """
    V_A_arr = np.asarray(V_A, dtype=float)
    xi_arr = np.asarray(xi, dtype=float)
    V_A_arr, xi_arr = np.broadcast_arrays(V_A_arr, xi_arr)
    out = np.empty_like(V_A_arr, dtype=float)

    it = np.nditer(V_A_arr, flags=["multi_index"])
    for v in it:
        vA = float(v)
        this_xi = float(xi_arr[it.multi_index])
        CM = build_covariance_matrix(V_A=vA, T=T, xi=this_xi, eta=eta, V_el=V_el)
        l1, l2 = symplectic_eigenvalues(CM)
        l3, l4 = symplectic_eigenvalues_eve(V_A=vA, T=T, xi=this_xi, eta=eta, V_el=V_el)
        out[it.multi_index] = float(g(l1) + g(l2) - g(l3) - g(l4))

    return out


def key_rate(V_A, T, xi, eta, V_el, beta):
    """
    Devetak–Winter key rate K (bits/use), clipped at zero.

        K = beta * I_AB - chi_BE

    Parameters
    ----------
    V_A : float or np.ndarray
        Alice modulation variance (SNU). Supports vectorized evaluation.
    T : float
        Channel transmittance (dimensionless).
    xi : float or np.ndarray
        Channel excess noise (SNU, input-referred). Can be scalar or an array
        broadcastable to the shape of V_A.
    eta : float
        Bob detection efficiency (dimensionless).
    V_el : float
        Bob electronic noise (SNU).
    beta : float
        Reconciliation efficiency (dimensionless, 0..1).

    Returns
    -------
    np.ndarray
        Non-negative key rate in bits per channel use.
    """
    I_AB = mutual_information(V_A=V_A, T=T, xi=xi, eta=eta, V_el=V_el)
    chi = holevo_bound(V_A=V_A, T=T, xi=xi, eta=eta, V_el=V_el)
    K = beta * I_AB - chi
    return np.maximum(K, 0.0)


if __name__ == "__main__":
    # Validation plot: key rate vs V_A for several channel settings.
    V_A_grid = np.logspace(np.log10(1.01), np.log10(100.0), 200)

    eta = 0.6
    V_el = 0.01
    beta = 0.95

    cases = [
        (0.8, 0.01),
        (0.5, 0.01),
        (0.3, 0.02),
    ]

    plt.figure(figsize=(8, 5))
    for T, xi in cases:
        K = key_rate(V_A=V_A_grid, T=T, xi=xi, eta=eta, V_el=V_el, beta=beta)
        plt.plot(V_A_grid, K, label=f"T={T:.1f}, xi={xi:.3f}")

        # Sanity: key rate must be >= 0 after clipping
        assert np.all(K >= -1e-12)

    plt.xscale("log")
    plt.xlabel(r"Modulation variance $V_A$ (SNU)")
    plt.ylabel(r"Key rate $K$ (bits/use)")
    plt.title("CV-QKD key rate validation (homodyne, Gaussian model)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()

    out_path = Path("outputs/figures/key_rate_validation.png")
    os.makedirs(out_path.parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

