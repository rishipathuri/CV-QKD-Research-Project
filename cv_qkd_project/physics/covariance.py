from __future__ import annotations

import numpy as np


def build_covariance_matrix(V_A, T, xi, eta, V_el) -> np.ndarray:
    """
    Construct the 4x4 Alice–Bob covariance matrix for GMCS CV-QKD (SNU).

    This is the standard *entanglement-based* (EB) representation of a
    Gaussian-modulated coherent-state (GMCS) CV-QKD protocol, written in the
    ordering (x_A, p_A, x_B, p_B), with a two-mode covariance matrix:

        CM = [ A   C ]
             [ C^T B ]

    where each block is 2x2.

    Conventions used here (all variances in shot-noise units, SNU):
    - Alice's marginal variance is V = V_A + 1, i.e. modulation variance plus
      vacuum shot noise.
    - Channel transmittance is T (dimensionless).
    - Channel excess noise is xi (SNU, referred to channel input).
    - Bob's detector is modeled by an overall efficiency eta (dimensionless)
      and additive electronic noise V_el (SNU).

    Block definitions
    -----------------
    - Alice block:
        A = V * I_2
    - Correlation block (standard form for a two-mode squeezed state mapped
      through a lossy channel):
        C = diag( +c, -c ), where c = sqrt( T * (V^2 - 1) )
      This follows the "standard form" correlation structure in EB CV-QKD.
    - Bob block:
        B = V_B_meas * I_2
      where V_B_meas includes:
        - loss-induced vacuum \((1-T)\),
        - channel excess noise \(T*xi\) (input-referred),
        - detector inefficiency vacuum \((1-eta)\),
        - electronic noise \(V_el\).

    Notes
    -----
    If detector inefficiency is modeled as a beam-splitter mixing in vacuum,
    then it reduces Alice–Bob correlations by a factor of sqrt(eta) (and adds a
    vacuum contribution to Bob's marginal). Without this factor, the resulting
    CM can become non-physical (violating the uncertainty principle) when B is
    interpreted as the *measured* variance. This implementation therefore uses
    c = sqrt(eta * T * (V^2 - 1)).

    Parameters
    ----------
    V_A : float
        Alice modulation variance (SNU).
    T : float
        Channel transmittance (dimensionless).
    xi : float
        Channel excess noise (SNU), referred to channel input.
    eta : float
        Bob's (single) homodyne detector efficiency (dimensionless).
    V_el : float
        Bob's electronic noise (SNU).

    Returns
    -------
    np.ndarray
        4x4 covariance matrix CM in SNU, with ordering (x_A, p_A, x_B, p_B).
    """
    V_A = float(V_A)
    T = float(T)
    xi = float(xi)
    eta = float(eta)
    V_el = float(V_el)

    V = V_A + 1.0

    # Bob marginal (EB variance propagation, SNU):
    # - after channel: T*V + (1-T)*1 + T*xi
    # - after detection inefficiency: eta * V_B_pre + (1-eta)*1
    # - plus electronic noise: + V_el
    V_B_pre = T * V + (1.0 - T) * 1.0 + T * xi
    V_B_meas = eta * V_B_pre + (1.0 - eta) * 1.0 + V_el

    # Correlations reduced by channel loss and detector inefficiency.
    c = float(np.sqrt(max(0.0, eta * T * (V * V - 1.0))))

    A = np.array([[V, 0.0], [0.0, V]], dtype=float)
    B = np.array([[V_B_meas, 0.0], [0.0, V_B_meas]], dtype=float)
    C = np.array([[c, 0.0], [0.0, -c]], dtype=float)

    top = np.concatenate([A, C], axis=1)
    bottom = np.concatenate([C.T, B], axis=1)
    CM = np.concatenate([top, bottom], axis=0)
    return CM


def symplectic_eigenvalues(CM: np.ndarray) -> tuple[float, float]:
    """
    Compute the two symplectic eigenvalues of a 4x4 covariance matrix (SNU).

    For a two-mode Gaussian state with covariance matrix CM, the symplectic
    eigenvalues {nu1, nu2} can be obtained as the absolute values of the
    eigenvalues of:

        i * Omega * CM

    where Omega is the 4x4 symplectic form.

    Parameters
    ----------
    CM : np.ndarray
        4x4 covariance matrix in the ordering (x_A, p_A, x_B, p_B), in SNU.

    Returns
    -------
    (float, float)
        The two positive symplectic eigenvalues (nu1 <= nu2).
    """
    CM = np.asarray(CM, dtype=float)
    if CM.shape != (4, 4):
        raise ValueError(f"Expected CM shape (4,4), got {CM.shape}")

    Omega = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=float,
    )

    eigvals = np.linalg.eigvals(1j * Omega @ CM)
    absvals = np.sort(np.abs(eigvals))
    # For a physical 2-mode CM, abs(eigs) appear as [nu1, nu1, nu2, nu2].
    nu1 = float(absvals[0])
    nu2 = float(absvals[2])
    return (nu1, nu2)


def symplectic_eigenvalues_eve(V_A, T, xi, eta, V_el) -> tuple[float, float]:
    """
    Compute the additional symplectic eigenvalue(s) used for conditional Holevo terms.

    In reverse reconciliation with Bob homodyne detection, the Holevo quantity
    often uses:
    - The two symplectic eigenvalues of the pre-measurement Alice–Bob CM (nu1, nu2)
    - Additional eigenvalue(s) for the conditional state after Bob's measurement.

    This helper returns a pair (nu3, nu4) corresponding to the conditional
    symplectic spectrum used in many CV-QKD derivations.

    Implementation detail (homodyne conditioning)
    ---------------------------------------------
    Bob's homodyne measurement of (say) x_B can be modeled by conditioning the
    Alice subsystem on Bob's measured quadrature. For a 2-mode CM in block form:

        CM = [ A  C ]
             [ C^T B ]

    with 2x2 blocks, the conditional covariance of Alice given x_B is:

        A_cond = A - C * (Pi * B * Pi)^MP * C^T

    where Pi = diag(1, 0) projects onto the measured quadrature and ^MP is the
    Moore–Penrose pseudo-inverse.

    The conditional state is single-mode, so it has one symplectic eigenvalue:
        nu_cond = sqrt(det(A_cond))

    We return (nu_cond, 1.0) as a two-value tuple for convenience/compatibility
    with formulas written using two conditional eigenvalues; the second entry
    equals vacuum (1 SNU) corresponding to the eliminated quadrature.

    Parameters
    ----------
    V_A, T, xi, eta, V_el : float
        Same meaning and units as in `build_covariance_matrix` (SNU).

    Returns
    -------
    (float, float)
        (nu3, nu4) with nu3 >= 1 for physical states, and nu4 = 1.0.
    """
    CM = build_covariance_matrix(V_A=V_A, T=T, xi=xi, eta=eta, V_el=V_el)

    A = CM[:2, :2]
    B = CM[2:, 2:]
    C = CM[:2, 2:]

    Pi = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    BP = Pi @ B @ Pi
    BP_pinv = np.linalg.pinv(BP)
    A_cond = A - C @ BP_pinv @ C.T

    nu_cond = float(np.sqrt(max(0.0, np.linalg.det(A_cond))))
    return (nu_cond, 1.0)


if __name__ == "__main__":
    # Basic physicality checks: symplectic eigenvalues should satisfy nu >= 1 (SNU).
    params = dict(V_A=10.0, T=0.5, xi=0.01, eta=0.8, V_el=0.01)
    CM = build_covariance_matrix(**params)
    nu1, nu2 = symplectic_eigenvalues(CM)
    nu3, nu4 = symplectic_eigenvalues_eve(**params)

    print("Parameters:", params)
    print("CM:\n", CM)
    print("Symplectic eigenvalues (AB):", (nu1, nu2))
    print("Conditional eigenvalues (Eve helper):", (nu3, nu4))

    assert nu1 >= 1.0 - 1e-9
    assert nu2 >= 1.0 - 1e-9
    assert nu3 >= 1.0 - 1e-9
    assert nu4 >= 1.0 - 1e-9

