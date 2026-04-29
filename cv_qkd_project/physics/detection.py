from __future__ import annotations

import numpy as np


def homodyne_detect(V_B, eta, V_el):
    """
    Ideal homodyne detection model with finite efficiency and electronic noise (SNU).

    This function maps the pre-detection quadrature variance at Bob (V_B) to the
    measured variance after a homodyne detector characterized by:

    - Detection efficiency eta (dimensionless, 0..1): models optical coupling
      loss + photodiode quantum efficiency as an effective beam-splitter mixing
      in vacuum noise.
    - Electronic noise V_el (SNU): additive detector noise referred to the
      measurement output.

    The measured variance is:

        V_meas = eta * V_B + (1 - eta) * 1 + V_el

    where the constant 1 is vacuum shot-noise (SNU) injected by inefficiency.

    Parameters
    ----------
    V_B : float or np.ndarray
        Bob's pre-detection output variance (SNU), e.g. from a channel model.
        Can be scalar or array; standard NumPy broadcasting rules apply.
    eta : float or np.ndarray
        Overall detection efficiency (dimensionless). eta=1 is ideal (no vacuum
        mixing); eta<1 mixes in vacuum shot noise of 1 SNU. Can be scalar or
        array; standard NumPy broadcasting rules apply.
    V_el : float or np.ndarray
        Electronic noise variance (SNU) added at the detector output. Can be
        scalar or array; standard NumPy broadcasting rules apply.

    Returns
    -------
    np.ndarray
        Measured quadrature variance V_meas (SNU), broadcast to the common
        shape of the inputs.
    """
    V_B = np.asarray(V_B, dtype=float)
    eta = np.asarray(eta, dtype=float)
    V_el = np.asarray(V_el, dtype=float)

    return eta * V_B + (1.0 - eta) * 1.0 + V_el


if __name__ == "__main__":
    # Scalar sanity check
    V_B = 6.0
    eta = 0.8
    V_el = 0.01
    V_meas = homodyne_detect(V_B=V_B, eta=eta, V_el=V_el)
    print("Scalar sanity check:")
    print(f"V_B={V_B}, eta={eta}, V_el={V_el} -> V_meas={V_meas:.6f}")
    expected = eta * V_B + (1.0 - eta) * 1.0 + V_el
    assert abs(V_meas - expected) < 1e-12
    # For eta in [0,1] and V_el>=0, V_meas is at least vacuum+electronic noise
    assert V_meas >= 1.0 + V_el - 1e-12

    # Vectorization / broadcasting checks
    V_B_vec = np.array([1.0, 2.0, 10.0])
    eta_vec = np.array([0.5, 0.8, 0.95])
    V_el_scalar = 0.01
    V_meas_vec = homodyne_detect(V_B=V_B_vec, eta=eta_vec, V_el=V_el_scalar)
    print("\nVector sanity check:")
    print("V_B_vec    =", V_B_vec)
    print("eta_vec    =", eta_vec)
    print("V_el       =", V_el_scalar)
    print("V_meas_vec =", V_meas_vec)
    assert V_meas_vec.shape == V_B_vec.shape
    # Bounds: should lie between (vacuum+V_el) and (V_B+V_el) for eta in [0,1]
    assert np.all(V_meas_vec >= 1.0 + V_el_scalar - 1e-12)
    assert np.all(V_meas_vec <= V_B_vec + V_el_scalar + 1e-12)

