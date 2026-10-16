"""
tests/test_forecaster.py
Unit tests for UBOS District Population Forecaster.
"""

import pytest
import numpy as np
from src.forecaster import (
    DistrictPopulation,
    LinearForecaster,
    CAGRForecaster,
    FibonacciRatioForecaster,
    evaluate_metrics,
    calculate_classroom_needs
)


def test_district_population_validation():
    dp = DistrictPopulation("Kampala", [2015, 2016], [1200, 1250])
    assert len(dp) == 2

    with pytest.raises(ValueError, match="Mismatch"):
        DistrictPopulation("Err", [2015, 2016], [1200])

    with pytest.raises(ValueError, match="negative"):
        DistrictPopulation("Err", [2015, 2016], [-100, 1200])


def test_variance_ddof_difference():
    dp = DistrictPopulation("Test", [2015, 2016, 2017], [10, 20, 30])
    builtin_stats = dp.compute_stats_builtin()
    numpy_ddof0 = dp.compute_stats_numpy(ddof=0)
    numpy_ddof1 = dp.compute_stats_numpy(ddof=1)

    assert builtin_stats["variance"] == 100.0
    assert numpy_ddof0["variance"] == 200.0 / 3.0
    assert numpy_ddof1["variance"] == 100.0


def test_forecaster_fitting_and_predictions():
    years = np.array([2015, 2016, 2017, 2018])
    pop = np.array([100.0, 110.0, 120.0, 130.0])
    years_pred = np.array([2019, 2020])

    lin = LinearForecaster().fit(years, pop)
    np.testing.assert_allclose(lin.predict(years_pred), [140.0, 150.0])

    cagr = CAGRForecaster().fit(years, pop)
    assert cagr.predict(years_pred)[0] > 130.0

    fib = FibonacciRatioForecaster().fit(years, pop)
    assert len(fib.predict(years_pred)) == 2


def test_metrics_and_classroom_needs():
    metrics = evaluate_metrics([100, 200], [110, 190])
    assert metrics["MAE"] == 10.0
    assert np.isclose(metrics["RMSE"], 10.0)

    needs = calculate_classroom_needs(100.0, primary_age_pct=0.18, capacity=53)
    assert needs["pupils"] == 18000.0
    assert needs["classrooms"] == 340