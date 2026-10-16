"""Supply-demand equilibrium calculations for the Ntinda route."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve


def solve_ntinda_equilibrium() -> tuple[float, float]:
    """Solve Qd=120-0.02P and Qs=10+0.03P for equilibrium fare and quantity.

    Rearrangement gives 0.02P + Q = 120 and -0.03P + Q = 10.
    """
    coefficients = np.array([[0.02, 1.0], [-0.03, 1.0]], dtype=float)
    constants = np.array([120.0, 10.0], dtype=float)
    fare, quantity = solve(coefficients, constants)
    return float(fare), float(quantity)
