"""
Central configuration for the CV-QKD simulation project.

All values are in shot-noise units (SNU) unless noted otherwise.
"""

from __future__ import annotations

import numpy as np

# -------------------------
# Physical / system settings
# -------------------------

# Channel transmittance range (dimensionless)
T_MIN = 0.1
T_MAX = 0.9

# Channel excess noise range (SNU, referred to channel input)
XI_MIN = 0.001
XI_MAX = 0.05

# Detector efficiency mismatch bounds (dimensionless)
ETA_MIN = 0.5
ETA_MAX = 0.95

# Bob's electronic noise (SNU)
V_EL = 0.01

# Reconciliation efficiency (dimensionless)
BETA = 0.95

# Side-channel model strength
# --------------------------
# Scale factor applied to the detector-efficiency mismatch noise model.
# Larger values make mismatch more damaging; smaller values make it milder.
MISMATCH_NOISE_SCALE = 0.05

# -------------------------
# Optimization / grids
# -------------------------

# Modulation variance grid for V_A optimization (SNU)
V_A_GRID_SIZE = 200
V_A_MIN = 1.01
V_A_MAX = 100.0
V_A_GRID = np.logspace(np.log10(V_A_MIN), np.log10(V_A_MAX), V_A_GRID_SIZE)

# -------------------------
# Dataset / training settings
# -------------------------

# Total dataset size (number of samples)
DATASET_SIZE_N = 50_000

