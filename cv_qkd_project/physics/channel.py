from __future__ import annotations

import numpy as np


def apply_channel(V_A, T, xi):
    """
    Apply a Gaussian lossy-noisy channel model (SNU).

    This implements a simple variance-propagation model for a coherent-state
    CV-QKD link where Alice uses Gaussian modulation and Bob receives the state
    through a channel characterized by:

    - Transmittance T (dimensionless, 0..1): captures optical loss.
    - Excess noise xi (SNU): noise in addition to vacuum noise, referred to the
      channel input (common convention in CV-QKD).

    The returned variance at Bob (before detection) is:

        V_B = T * V_A + 1 + T * xi

    where the constant 1 is the vacuum shot-noise contribution (SNU).

    Parameters
    ----------
    V_A : float or np.ndarray
        Alice modulation variance (SNU). This function is agnostic to whether
        this represents the total quadrature variance or the modulation-only
        term; it simply applies the formula provided.
    T : float or np.ndarray
        Channel transmittance (dimensionless). Can be scalar or array; standard
        NumPy broadcasting rules apply.
    xi : float or np.ndarray
        Channel excess noise (SNU) referred to the channel input. Can be scalar
        or array; standard NumPy broadcasting rules apply.

    Returns
    -------
    np.ndarray
        Bob's pre-detection output variance V_B (SNU), broadcast to the common
        shape of the inputs.
    """
    V_A = np.asarray(V_A, dtype=float)
    T = np.asarray(T, dtype=float)
    xi = np.asarray(xi, dtype=float)

    return T * V_A + 1.0 + T * xi


if __name__ == "__main__":
    # Sanity checks: scalar
    V_A = 10.0
    T = 0.5
    xi = 0.01
    V_B = apply_channel(V_A=V_A, T=T, xi=xi)
    print("Scalar sanity check:")
    print(f"V_A={V_A}, T={T}, xi={xi} -> V_B={V_B:.6f}")
    expected = T * V_A + 1.0 + T * xi
    assert abs(V_B - expected) < 1e-12
    assert V_B > 1.0  # should always be at least vacuum noise for T>=0

    # Sanity checks: vectorization / broadcasting
    V_A_vec = np.array([1.01, 2.0, 10.0])
    T_vec = np.array([0.1, 0.5, 0.9])
    xi_scalar = 0.02
    V_B_vec = apply_channel(V_A=V_A_vec, T=T_vec, xi=xi_scalar)
    print("\nVector sanity check:")
    print("V_A_vec =", V_A_vec)
    print("T_vec   =", T_vec)
    print("xi      =", xi_scalar)
    print("V_B_vec =", V_B_vec)
    assert V_B_vec.shape == V_A_vec.shape
    assert np.all(V_B_vec >= 1.0)  # vacuum contribution
    # With T in [0.1, 0.9] and V_A>=1.01, output should be within a reasonable range.
    assert np.all(V_B_vec <= 1.0 + np.max(T_vec) * np.max(V_A_vec) + np.max(T_vec) * xi_scalar + 1e-12)

