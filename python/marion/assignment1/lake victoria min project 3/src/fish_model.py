"""
src/fish_model.py
Lake Victoria Fish Stock & Export Revenue Risk Model.
"""

import math
import statistics
from typing import Dict, List, Tuple
import numpy as np


class FishStock:
    """
    Simulates discrete logistic fish biomass dynamics with harvesting:
    N(t+1) = N(t) + r * N(t) * (1 - N(t) / K) - h * N(t)
    """

    def __init__(self, r: float = 0.4, K: float = 10000.0, initial_stock: float = 4000.0) -> None:
        if K <= 0.0:
            raise ValueError("Carrying capacity 'K' must be positive.")
        if initial_stock < 0.0:
            raise ValueError("Initial stock cannot be negative.")
            
        self.r = r
        self.K = K
        self.initial_stock = initial_stock

    @property
    def maximum_sustainable_yield(self) -> float:
        """Computes theoretical Maximum Sustainable Yield (MSY = r*K / 4)."""
        return (self.r * self.K) / 4.0

    def simulate(
        self, weeks: int = 52, harvest_rate: float = 0.10, closed_weeks: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates stock biomass (tonnes) and weekly harvest output (kg).
        """
        if not (0.0 <= harvest_rate <= 1.0):
            raise ValueError("Harvest rate 'h' must be between 0.0 and 1.0.")
            
        closed_weeks = set(closed_weeks) if closed_weeks else set()

        stocks = np.zeros(weeks + 1)
        harvests_kg = np.zeros(weeks)

        stocks[0] = self.initial_stock

        for t in range(weeks):
            current_N = stocks[t]
            h = 0.0 if t in closed_weeks else harvest_rate

            harvest_tonnes = h * current_N
            harvests_kg[t] = harvest_tonnes * 1000.0  # Convert tonnes to kg

            growth = self.r * current_N * (1.0 - current_N / self.K)
            next_N = max(0.0, current_N + growth - harvest_tonnes)
            stocks[t + 1] = next_N

        return stocks, harvests_kg


class PriceModel:
    """Simulates weekly fish price (UGX/kg) using a bounded random walk."""

    def __init__(
        self,
        initial_price: float = 12000.0,
        min_price: float = 9000.0,
        max_price: float = 16000.0,
        volatility: float = 500.0
    ) -> None:
        self.initial_price = initial_price
        self.min_price = min_price
        self.max_price = max_price
        self.volatility = volatility

    def simulate_path(self, weeks: int = 52, seed: int = None) -> np.ndarray:
        """Generates a single weekly price path bounded between min_price and max_price."""
        rng = np.random.default_rng(seed)
        prices = np.zeros(weeks)
        current = self.initial_price

        for t in range(weeks):
            shock = rng.normal(0, self.volatility)
            current = np.clip(current + shock, self.min_price, self.max_price)
            prices[t] = current

        return prices


class RiskAssessor:
    """Evaluates statistical risk metrics, CV classifications, and Monte Carlo VaR."""

    @staticmethod
    def calculate_stats(data: List[float]) -> Dict[str, float]:
        if len(data) < 2:
            raise ValueError("At least two data points are required.")
            
        mean_val = statistics.mean(data)
        stdev_val = statistics.stdev(data)
        var_val = statistics.variance(data)
        median_val = statistics.median(data)
        cv_val = stdev_val / mean_val if mean_val != 0 else 0.0

        return {
            "mean": mean_val,
            "median": median_val,
            "variance": var_val,
            "stdev": stdev_val,
            "cv": cv_val
        }

    @staticmethod
    def classify_risk(cv: float) -> str:
        """Classifies operational risk based on Coefficient of Variation (CV)."""
        if cv < 0.15:
            return "Low Risk"
        elif cv <= 0.30:
            return "Moderate Risk"
        else:
            return "High Risk"

    @staticmethod
    def run_monte_carlo_var(
        fish_stock: FishStock,
        price_model: PriceModel,
        harvest_rate: float = 0.10,
        weeks: int = 52,
        iterations: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42
    ) -> Tuple[float, np.ndarray]:
        """Runs Monte Carlo simulations and calculates Value-at-Risk (5% VaR)."""
        rng = np.random.default_rng(seed)
        annual_revenues = np.zeros(iterations)

        _, harvests_kg = fish_stock.simulate(weeks=weeks, harvest_rate=harvest_rate)

        for i in range(iterations):
            path_seed = rng.integers(0, 1_000_000)
            prices = price_model.simulate_path(weeks=weeks, seed=path_seed)
            weekly_revenue = harvests_kg * prices
            annual_revenues[i] = np.sum(weekly_revenue)

        alpha = 1.0 - confidence_level
        var_5pct = float(np.percentile(annual_revenues, alpha * 100))

        return var_5pct, annual_revenues