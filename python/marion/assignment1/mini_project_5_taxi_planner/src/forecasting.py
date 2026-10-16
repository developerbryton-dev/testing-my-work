"""Transparent forecasting methods and rolling-origin evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class Forecaster(ABC):
    """Base class for one-step-ahead passenger-count forecasters."""

    name: str = "Forecaster"

    @abstractmethod
    def predict_next(self, history: Sequence[float]) -> float:
        """Forecast the next observation using history only."""
        raise NotImplementedError


@dataclass
class MovingAverageForecaster(Forecaster):
    """Forecast using the mean of the most recent `window` observations."""

    window: int = 3
    name: str = "3-day moving average"

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")

    def predict_next(self, history: Sequence[float]) -> float:
        values = np.asarray(history, dtype=float)
        if values.size == 0:
            raise ValueError("history cannot be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("history must contain finite values")
        return float(np.mean(values[-self.window:]))


@dataclass
class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing with a configurable alpha in (0, 1]."""

    alpha: float = 0.3
    name: str = "simple exponential smoothing"

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1]")

    def predict_next(self, history: Sequence[float]) -> float:
        values = np.asarray(history, dtype=float)
        if values.size == 0:
            raise ValueError("history cannot be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("history must contain finite values")
        level = float(values[0])
        for observation in values[1:]:
            level = self.alpha * float(observation) + (1 - self.alpha) * level
        return level


@dataclass
class LinearTrendForecaster(Forecaster):
    """Fit a least-squares straight line to the available history."""

    name: str = "linear trend"

    def predict_next(self, history: Sequence[float]) -> float:
        values = np.asarray(history, dtype=float)
        if values.size == 0:
            raise ValueError("history cannot be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("history must contain finite values")
        if values.size == 1:
            return float(values[0])
        x = np.arange(values.size, dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        return float(intercept + slope * values.size)


@dataclass
class SeasonalNaiveForecaster(Forecaster):
    """Forecast using the observation one seasonal period ago."""

    season_length: int = 7
    name: str = "seasonal naive (7-day)"

    def __post_init__(self) -> None:
        if self.season_length < 1:
            raise ValueError("season_length must be at least 1")

    def predict_next(self, history: Sequence[float]) -> float:
        values = np.asarray(history, dtype=float)
        if values.size < self.season_length:
            raise ValueError("history must contain at least one full season")
        return float(values[-self.season_length])


def rolling_origin_predictions(
    values: Sequence[float],
    forecaster: Forecaster,
    start_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return actuals and one-step forecasts from `start_index` onward.

    At each origin, the model sees only earlier observations, avoiding leakage.
    `start_index` is zero-based; 3 means the first forecast is for day 4.
    """
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not 1 <= start_index < series.size:
        raise ValueError("start_index must be between 1 and len(values)-1")
    predictions = []
    actuals = []
    for index in range(start_index, series.size):
        predictions.append(forecaster.predict_next(series[:index]))
        actuals.append(series[index])
    return np.asarray(actuals), np.asarray(predictions)


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Compute MAE, rejecting empty or mismatched arrays."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.size == 0 or actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must be non-empty and same shape")
    if not np.all(np.isfinite(actual_array)) or not np.all(np.isfinite(predicted_array)):
        raise ValueError("actual and predicted must contain finite values")
    return float(np.mean(np.abs(actual_array - predicted_array)))


def tune_alpha(
    values: Sequence[float],
    start_index: int = 3,
    alphas: Iterable[float] | None = None,
) -> tuple[float, dict[float, float]]:
    """Choose alpha with lowest walk-forward MAE over the specified origins."""
    grid = list(alphas) if alphas is not None else list(np.linspace(0.1, 1.0, 10))
    if not grid:
        raise ValueError("alpha grid cannot be empty")
    scores: dict[float, float] = {}
    for alpha in grid:
        model = ExponentialSmoothingForecaster(float(alpha))
        actual, predicted = rolling_origin_predictions(values, model, start_index)
        scores[float(alpha)] = mean_absolute_error(actual, predicted)
    best_alpha = min(scores, key=scores.get)
    return best_alpha, scores
