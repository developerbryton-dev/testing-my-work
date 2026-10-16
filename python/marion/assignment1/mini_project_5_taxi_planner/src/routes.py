"""Route revenue and descriptive-statistics models for the taxi planner."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pvariance, pstdev
from typing import Sequence

import numpy as np


@dataclass
class Route:
    """Represent one taxi route and its observed daily passenger counts.

    Args:
        name: Human-readable route name.
        passenger_counts: One passenger total per observed day.
        fare_ugx: Fare charged per passenger, in Ugandan shillings.
    """

    name: str
    passenger_counts: np.ndarray
    fare_ugx: float

    def __post_init__(self) -> None:
        """Validate and normalize route data."""
        self.passenger_counts = np.asarray(self.passenger_counts, dtype=float)
        if self.passenger_counts.ndim != 1:
            raise ValueError("passenger_counts must be a one-dimensional array")
        if self.passenger_counts.size == 0:
            raise ValueError("passenger_counts cannot be empty")
        if not np.all(np.isfinite(self.passenger_counts)):
            raise ValueError("passenger_counts must contain finite values")
        if np.any(self.passenger_counts < 0):
            raise ValueError("passenger_counts cannot be negative")
        if not np.isfinite(self.fare_ugx) or self.fare_ugx < 0:
            raise ValueError("fare_ugx must be finite and non-negative")

    def daily_revenue(self) -> np.ndarray:
        """Return observed daily gross revenue in UGX."""
        return self.passenger_counts * self.fare_ugx

    def total_revenue(self) -> float:
        """Return total gross revenue over all observed days in UGX."""
        return float(np.sum(self.daily_revenue()))

    def descriptive_statistics(self) -> dict[str, float]:
        """Return mean, population variance and population standard deviation.

        The statistics module is used as required by the assignment. Population
        measures are chosen because the supplied ten days are summarized as the
        observed period, rather than treated as a random sample.
        """
        values = self.passenger_counts.tolist()
        return {
            "mean": float(mean(values)),
            "variance": float(pvariance(values)),
            "standard_deviation": float(pstdev(values)),
        }

    def __str__(self) -> str:
        """Provide a concise readable route description."""
        return f"{self.name} (fare UGX {self.fare_ugx:,.0f})"
