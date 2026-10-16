"""
tests/test_microgrid.py
Unit tests for MicroGrid and HybridMicroGrid classes.
"""

import pytest
import numpy as np
from src.microgrid import MicroGrid, HybridMicroGrid


def test_microgrid_properties_and_solve():
    grid = MicroGrid()
    assert np.isclose(grid.determinant, -5.0)
    # The L2 condition number of [[3, 2], [4, 1]] is 3 + 2*sqrt(2) ≈ 5.8284
    assert np.isclose(grid.condition_number, 5.8284, atol=1e-3)

    x, y, feasible = grid.solve_day(d1=12.0, d2=11.0)
    assert np.isclose(x, 2.0)
    assert np.isclose(y, 3.0)
    assert feasible is True


def test_infeasible_demand_handling():
    grid = MicroGrid()
    x, y, feasible = grid.solve_day(d1=1.0, d2=20.0)
    assert feasible is False
    assert x >= 0.0
    assert y >= 0.0


def test_invalid_input_exceptions():
    grid = MicroGrid()
    with pytest.raises(ValueError, match="Demands D1 and D2 must be non-negative"):
        grid.solve_day(d1=-10.0, d2=15.0)


def test_hybrid_microgrid_extension():
    hybrid = HybridMicroGrid()
    x, y, z = hybrid.solve_hybrid_day(d1=6.0, d2=7.0, d3=3.0)
    assert x >= 0.0 and y >= 0.0 and z >= 0.0