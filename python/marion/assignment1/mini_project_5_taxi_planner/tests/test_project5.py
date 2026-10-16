import numpy as np
import pytest

from src.routes import Route
from src.forecasting import (
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    LinearTrendForecaster,
    SeasonalNaiveForecaster,
    mean_absolute_error,
    tune_alpha,
)
from src.equilibrium import solve_ntinda_equilibrium


def test_route_daily_and_total_revenue():
    route = Route("Test", np.array([2, 3, 4]), 1000)
    assert route.daily_revenue().tolist() == [2000.0, 3000.0, 4000.0]
    assert route.total_revenue() == 9000.0


def test_route_rejects_empty_passenger_counts():
    with pytest.raises(ValueError):
        Route("Empty", np.array([]), 1000)


def test_route_descriptive_statistics():
    stats = Route("Test", np.array([1, 2, 3]), 500).descriptive_statistics()
    assert stats["mean"] == 2
    assert stats["variance"] == pytest.approx(2 / 3)
    assert stats["standard_deviation"] == pytest.approx(np.sqrt(2 / 3))


def test_moving_average_and_empty_history():
    model = MovingAverageForecaster(3)
    assert model.predict_next([10, 20, 30, 40]) == 30
    with pytest.raises(ValueError):
        model.predict_next([])


def test_exponential_smoothing_alpha_one_returns_last_value():
    assert ExponentialSmoothingForecaster(1.0).predict_next([2, 4, 9]) == 9


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(0)


def test_linear_trend_predicts_next_linear_value():
    assert LinearTrendForecaster().predict_next([2, 4, 6]) == pytest.approx(8)


def test_seasonal_naive_requires_a_full_season():
    with pytest.raises(ValueError):
        SeasonalNaiveForecaster(7).predict_next([1, 2, 3])


def test_mae_and_alpha_grid_search():
    assert mean_absolute_error([1, 3], [2, 1]) == 1.5
    alpha, scores = tune_alpha([10, 12, 11, 14, 13, 15, 16], start_index=3,
                               alphas=[0.2, 0.5, 1.0])
    assert alpha in scores
    assert scores[alpha] == min(scores.values())


def test_equilibrium_values():
    fare, quantity = solve_ntinda_equilibrium()
    assert fare == pytest.approx(2200)
    assert quantity == pytest.approx(76)
