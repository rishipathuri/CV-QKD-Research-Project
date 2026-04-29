from __future__ import annotations

import numpy as np


def effective_eta(eta1, eta2):
    """
    Effective (mean) detection efficiency for a 2-arm homodyne receiver.

    Parameters
    ----------
    eta1, eta2 : float or np.ndarray
        Detection efficiencies of the two homodyne arms (dimensionless, 0..1).
        Supports NumPy broadcasting.

    Returns
    -------
    np.ndarray
        Mean efficiency (eta1 + eta2) / 2.
    """
    eta1 = np.asarray(eta1, dtype=float)
    eta2 = np.asarray(eta2, dtype=float)
    return (eta1 + eta2) / 2.0


def mismatch_noise(eta1, eta2):
    """
    Additional (effective) excess-noise contribution from efficiency mismatch (SNU).

    This is a lightweight side-channel model intended for simulation studies:
    mismatch increases an *effective* noise term approximately proportional to
    the square of the imbalance, normalized by the mean efficiency:

        xi_mismatch ∝ (eta1 - eta2)^2 / mean_eta

    We implement the proportionality with unit coefficient, returning:

        xi_mismatch = (eta1 - eta2)^2 / (mean_eta + eps)

    Parameters
    ----------
    eta1, eta2 : float or np.ndarray
        Efficiencies of the two homodyne arms (dimensionless, 0..1).

    Returns
    -------
    np.ndarray
        Additional noise term in SNU (non-negative).
    """
    eta1 = np.asarray(eta1, dtype=float)
    eta2 = np.asarray(eta2, dtype=float)
    mean_eta = effective_eta(eta1, eta2)
    eps = 1e-12
    return (eta1 - eta2) ** 2 / (mean_eta + eps)


def mismatch_detection(V_B, eta1, eta2, V_el):
    """
    Homodyne detection under efficiency mismatch (SNU).

    We model mismatch as:
    - An effective detection efficiency eta_eff = (eta1 + eta2)/2
    - An additive mismatch-induced noise term xi_mismatch (SNU)

    The measured variance is then:

        V_meas = eta_eff * V_B + (1 - eta_eff) * 1 + V_el + xi_mismatch

    Parameters
    ----------
    V_B : float or np.ndarray
        Bob pre-detection quadrature variance (SNU).
    eta1, eta2 : float or np.ndarray
        Two-arm efficiencies (dimensionless, 0..1).
    V_el : float or np.ndarray
        Electronic noise at detector output (SNU).

    Returns
    -------
    np.ndarray
        Biased measured variance (SNU), broadcast to the common shape.
    """
    V_B = np.asarray(V_B, dtype=float)
    V_el = np.asarray(V_el, dtype=float)
    eta_eff = effective_eta(eta1, eta2)
    xi_m = mismatch_noise(eta1, eta2)
    return eta_eff * V_B + (1.0 - eta_eff) * 1.0 + V_el + xi_m


if __name__ == "__main__":
    # Basic sanity checks
    eta1, eta2 = 0.7, 0.5
    eta_eff = effective_eta(eta1, eta2)
    xi_m = mismatch_noise(eta1, eta2)
    V_meas = mismatch_detection(V_B=6.0, eta1=eta1, eta2=eta2, V_el=0.01)
    print("eta_eff =", eta_eff)
    print("xi_mismatch =", xi_m)
    print("V_meas =", V_meas)
    assert eta_eff == (eta1 + eta2) / 2.0
    assert xi_m >= 0.0
    assert V_meas >= 1.0  # at least vacuum scale in SNU for reasonable params

