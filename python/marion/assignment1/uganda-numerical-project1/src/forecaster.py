"""
src/forecaster.py
UBOS District Population Forecaster module.
"""

from abc import ABC, abstractmethod
import math
import statistics
from typing import Dict, List, Tuple, Union
import numpy as np


class DistrictPopulation:
    """Stores and validates district population historical data."""

    def __init__(self, name: str, years: Union[List[int], np.ndarray], population: Union[List[float], np.ndarray]) -> None:
        self.name = str(name)
        self.years = np.asarray(years, dtype=int)
        self.population = np.asarray(population, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if len(self.years) != len(self.population):
            raise ValueError(f"Mismatch: years length ({len(self.years)}) != population length ({len(self.population)}).")
        if np.any(self.population < 0):
            raise ValueError("Population values cannot be negative.")
        if len(self.years) < 2:
            raise ValueError("At least 2 data points are required.")

    def __len__(self) -> int:
        return len(self.years)

    def __repr__(self) -> str:
        return f"DistrictPopulation(name='{self.name}', points={len(self)}, range={self.years[0]}-{self.years[-1]})"

    def compute_stats_builtin(self) -> Dict[str, float]:
        pop_list = self.population.tolist()
        return {
            "mean": statistics.mean(pop_list),
            "median": statistics.median(pop_list),
            "variance": statistics.variance(pop_list),
            "stdev": statistics.stdev(pop_list)
        }

    def compute_stats_numpy(self, ddof: int = 0) -> Dict[str, float]:
        return {
            "mean": float(np.mean(self.population)),
            "median": float(np.median(self.population)),
            "variance": float(np.var(self.population, ddof=ddof)),
            "stdev": float(np.std(self.population, ddof=ddof))
        }

    def compute_yoy_growth(self) -> np.ndarray:
        return (self.population[1:] - self.population[:-1]) / self.population[:-1]

    def compute_cagr(self) -> float:
        n_years = len(self.years) - 1
        return float((self.population[-1] / self.population[0]) ** (1.0 / n_years) - 1.0)


class Forecaster(ABC):
    """Abstract Base Class for Population Forecasters."""

    def __init__(self) -> None:
        self.is_fitted = False
        self.years_train: np.ndarray = np.array([])
        self.pop_train: np.ndarray = np.array([])

    @abstractmethod
    def fit(self, years: np.ndarray, population: np.ndarray) -> "Forecaster":
        pass

    @abstractmethod
    def predict(self, years_to_predict: np.ndarray) -> np.ndarray:
        pass


class LinearForecaster(Forecaster):
    def __init__(self) -> None:
        super().__init__()
        self.slope = 0.0
        self.intercept = 0.0

    def fit(self, years: np.ndarray, population: np.ndarray) -> "LinearForecaster":
        self.years_train = np.asarray(years, dtype=float)
        self.pop_train = np.asarray(population, dtype=float)
        coeffs = np.polyfit(self.years_train, self.pop_train, deg=1)
        self.slope, self.intercept = coeffs[0], coeffs[1]
        self.is_fitted = True
        return self

    def predict(self, years_to_predict: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        t = np.asarray(years_to_predict, dtype=float)
        return self.slope * t + self.intercept


class CAGRForecaster(Forecaster):
    def __init__(self) -> None:
        super().__init__()
        self.cagr = 0.0
        self.base_year = 0.0
        self.base_pop = 0.0

    def fit(self, years: np.ndarray, population: np.ndarray) -> "CAGRForecaster":
        self.years_train = np.asarray(years, dtype=float)
        self.pop_train = np.asarray(population, dtype=float)
        n = len(self.years_train) - 1
        self.cagr = (self.pop_train[-1] / self.pop_train[0]) ** (1.0 / n) - 1.0
        self.base_year = self.years_train[-1]
        self.base_pop = self.pop_train[-1]
        self.is_fitted = True
        return self

    def predict(self, years_to_predict: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        t = np.asarray(years_to_predict, dtype=float)
        dt = t - self.base_year
        return self.base_pop * ((1.0 + self.cagr) ** dt)


class FibonacciRatioForecaster(Forecaster):
    def __init__(self, start_k: int = 10) -> None:
        super().__init__()
        self.start_k = start_k
        self.base_year = 0.0
        self.base_pop = 0.0

    @staticmethod
    def _fibonacci(n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    def fit(self, years: np.ndarray, population: np.ndarray) -> "FibonacciRatioForecaster":
        self.years_train = np.asarray(years, dtype=float)
        self.pop_train = np.asarray(population, dtype=float)
        self.base_year = self.years_train[-1]
        self.base_pop = self.pop_train[-1]
        self.is_fitted = True
        return self

    def predict(self, years_to_predict: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        t = np.asarray(years_to_predict, dtype=float)
        preds = []
        current = self.base_pop
        for i in range(len(t)):
            k = self.start_k + i
            ratio = self._fibonacci(k + 1) / self._fibonacci(k)
            current *= ratio
            preds.append(current)
        return np.array(preds)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def bootstrap_residuals(
    model: Forecaster,
    years_train: np.ndarray,
    pop_train: np.ndarray,
    years_forecast: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    model.fit(years_train, pop_train)
    fitted_train = model.predict(years_train)
    residuals = pop_train - fitted_train

    base_forecast = model.predict(years_forecast)
    bootstrap_paths = np.zeros((n_resamples, len(years_forecast)))

    for i in range(n_resamples):
        resampled_res = rng.choice(residuals, size=len(residuals), replace=True)
        synthetic_pop = fitted_train + resampled_res
        
        temp_model = model.__class__()
        temp_model.fit(years_train, synthetic_pop)
        
        noise = rng.choice(residuals, size=len(years_forecast), replace=True)
        bootstrap_paths[i, :] = temp_model.predict(years_forecast) + noise

    lower_bound = np.percentile(bootstrap_paths, 2.5, axis=0)
    upper_bound = np.percentile(bootstrap_paths, 97.5, axis=0)

    return base_forecast, lower_bound, upper_bound


def calculate_classroom_needs(
    pop_thousands: float, primary_age_pct: float = 0.18, capacity: int = 53
) -> Dict[str, float]:
    total_pop = pop_thousands * 1000.0
    pupils = total_pop * primary_age_pct
    classrooms = math.ceil(pupils / capacity)
    return {"pupils": pupils, "classrooms": classrooms}