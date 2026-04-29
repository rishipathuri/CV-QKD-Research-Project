from __future__ import annotations

from typing import Callable

import numpy as np

from cv_qkd_project import config


def optimal_VA(
    T: float,
    xi: float,
    eta: float,
    V_el: float,
    beta: float,
    key_rate_fn: Callable[..., np.ndarray] | None = None,
) -> tuple[float, float]:
    """
    Brute-force search for the optimal modulation variance V_A on a fixed grid.

    This evaluates a key-rate function over the modulation variance grid
    `config.V_A_GRID` and returns:

        (V_A_star, K_max)

    Parameters
    ----------
    T : float
        Channel transmittance (dimensionless).
    xi : float
        Channel excess noise (SNU, input-referred).
    eta : float
        Bob detection efficiency (dimensionless).
    V_el : float
        Bob electronic noise (SNU).
    beta : float
        Reconciliation efficiency (dimensionless).
    key_rate_fn : callable, optional
        A function compatible with signature:
            key_rate_fn(V_A, T, xi, eta, V_el, beta) -> array_like
        If omitted, defaults to `cv_qkd_project.physics.key_rate.key_rate`.

        This hook exists so later you can pass a mismatch-aware key-rate model.

    Returns
    -------
    (float, float)
        Tuple `(optimal_VA, max_key_rate)` in SNU and bits/use, respectively.
    """
    if key_rate_fn is None:
        from cv_qkd_project.physics.key_rate import key_rate as key_rate_fn  # local import

    V_A_grid = config.V_A_GRID
    K = np.asarray(key_rate_fn(V_A=V_A_grid, T=T, xi=xi, eta=eta, V_el=V_el, beta=beta), dtype=float)

    # Guard against any numerical NaNs/Infs from extreme parameters.
    K = np.where(np.isfinite(K), K, -np.inf)
    idx = int(np.argmax(K))
    return float(V_A_grid[idx]), float(K[idx])


if __name__ == "__main__":
    # Small sanity check
    v_star, k_star = optimal_VA(T=0.5, xi=0.01, eta=0.6, V_el=0.01, beta=0.95)
    print("optimal_VA =", v_star)
    print("max_key_rate =", k_star)
    assert v_star >= config.V_A_MIN - 1e-12
    assert v_star <= config.V_A_MAX + 1e-12
    assert k_star >= -1e-12

