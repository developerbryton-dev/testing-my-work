"""
src/microgrid.py
Domain models for Kasese Health Centre Solar Micro-Grid Dispatch Planning.
"""

import statistics
from typing import Tuple, Dict, List, Optional
import numpy as np
import scipy.linalg
from scipy.optimize import nnls


class MicroGrid:
    """
    Models a 2-source (Solar x, Battery y) micro-grid dispatch planner.

    System Equations:
        3x + 2y = D1 (Daytime load)
        4x + 1y = D2 (Critical load)
    """

    def __init__(self, cost_solar: float = 150.0, cost_battery: float = 450.0) -> None:
        self.A = np.array([[3.0, 2.0], [4.0, 1.0]], dtype=float)
        self.cost_solar = float(cost_solar)
        self.cost_battery = float(cost_battery)

    @property
    def determinant(self) -> float:
        """Computes matrix determinant."""
        return float(np.linalg.det(self.A))

    @property
    def condition_number(self) -> float:
        """Computes matrix condition number."""
        return float(np.linalg.cond(self.A))

    def solve_day(self, d1: float, d2: float) -> Tuple[float, float, bool]:
        """
        Solves dispatch for a single day (D1, D2).
        Returns (x_solar, y_battery, is_feasible).
        """
        if d1 < 0 or d2 < 0:
            raise ValueError("Demands D1 and D2 must be non-negative.")

        b = np.array([d1, d2], dtype=float)
        x, y = scipy.linalg.solve(self.A, b)

        # Detect physical infeasibility (negative generation)
        if x < 0 or y < 0:
            # Fallback to Non-Negative Least Squares (NNLS)
            sol_nnls, _ = nnls(self.A, b)
            return float(sol_nnls[0]), float(sol_nnls[1]), False

        return float(x), float(y), True

    def solve_batch_loop(self, demands: np.ndarray) -> np.ndarray:
        """Solves 30 days using a standard Python loop."""
        num_days = demands.shape[1]
        results = np.zeros((2, num_days), dtype=float)
        for t in range(num_days):
            x, y, _ = self.solve_day(demands[0, t], demands[1, t])
            results[0, t] = x
            results[1, t] = y
        return results

    def solve_batch_vectorized(self, demands: np.ndarray) -> np.ndarray:
        """Solves 30 days in a single vectorized matrix call."""
        results = scipy.linalg.solve(self.A, demands)
        
        # Apply NNLS fix for any day that resulted in negative values
        for t in range(results.shape[1]):
            if results[0, t] < 0 or results[1, t] < 0:
                sol_nnls, _ = nnls(self.A, demands[:, t])
                results[:, t] = sol_nnls

        return results

    def compute_costs(self, x_solar: np.ndarray, y_battery: np.ndarray) -> np.ndarray:
        """Computes daily energy cost in UGX."""
        return (x_solar * self.cost_solar) + (y_battery * self.cost_battery)


class HybridMicroGrid(MicroGrid):
    """
    Extension: 3-source system adding Diesel Generator (z).
    """

    def __init__(
        self,
        cost_solar: float = 150.0,
        cost_battery: float = 450.0,
        cost_diesel: float = 800.0
    ) -> None:
        super().__init__(cost_solar, cost_battery)
        self.A3 = np.array([
            [3.0, 2.0, 1.0],
            [4.0, 1.0, 2.0],
            [1.0, 1.0, 1.0]
        ], dtype=float)
        self.cost_diesel = float(cost_diesel)

    def solve_hybrid_day(self, d1: float, d2: float, d3: float) -> Tuple[float, float, float]:
        """Solves 3x3 hybrid grid equation."""
        b = np.array([d1, d2, d3], dtype=float)
        
        if np.abs(np.linalg.det(self.A3)) < 1e-12:
            raise np.linalg.LinAlgError("System is singular or linearly dependent.")

        sol = scipy.linalg.solve(self.A3, b)
        if np.any(sol < 0):
            sol, _ = nnls(self.A3, b)

        return float(sol[0]), float(sol[1]), float(sol[2])