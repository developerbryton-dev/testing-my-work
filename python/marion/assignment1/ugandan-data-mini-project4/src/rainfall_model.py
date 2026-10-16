"""
Rainfall Pattern & Crop Suitability Analyser.

This module contains reusable classes and functions for analysing
monthly rainfall patterns and crop suitability in Uganda.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine as scipy_cosine


MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


@dataclass
class Region:
    """Represent a region with 12 monthly rainfall observations.

    Parameters
    ----------
    name:
        Name of the region.
    rainfall:
        NumPy array containing January to December rainfall in mm.
    """

    name: str
    rainfall: np.ndarray

    def __post_init__(self) -> None:
        """Validate rainfall data after object creation."""
        self.rainfall = np.asarray(self.rainfall, dtype=float)

        if self.rainfall.shape != (12,):
            raise ValueError(
                "Rainfall must contain exactly 12 monthly values."
            )

        if np.any(self.rainfall < 0):
            raise ValueError(
                "Rainfall values cannot be negative."
            )

    def annual_total(self) -> float:
        """Return total annual rainfall in millimetres."""
        return float(np.sum(self.rainfall))

    def monthly_mean(self) -> float:
        """Return mean monthly rainfall in millimetres."""
        return float(np.mean(self.rainfall))

    def wettest_month(self) -> str:
        """Return the month with the highest rainfall."""
        index = int(np.argmax(self.rainfall))
        return MONTHS[index]

    def driest_month(self) -> str:
        """Return the month with the lowest rainfall."""
        index = int(np.argmin(self.rainfall))
        return MONTHS[index]

    def coefficient_of_variation(self) -> float:
        """Return the coefficient of variation of monthly rainfall.

        CV = sample standard deviation / mean.
        """
        mean = np.mean(self.rainfall)

        if mean == 0:
            return np.inf

        standard_deviation = np.std(
            self.rainfall,
            ddof=1
        )

        return float(standard_deviation / mean)

    def summary(self) -> Dict[str, float | str]:
        """Return the main rainfall statistics."""
        return {
            "annual_total": self.annual_total(),
            "monthly_mean": self.monthly_mean(),
            "wettest_month": self.wettest_month(),
            "driest_month": self.driest_month(),
            "cv": self.coefficient_of_variation(),
        }


@dataclass
class CropRule:
    """Represent a simplified monthly rainfall suitability rule.

    Parameters
    ----------
    crop:
        Name of the crop.
    minimum_rainfall:
        Lower monthly rainfall threshold in mm.
    maximum_rainfall:
        Upper monthly rainfall threshold in mm.

    Notes
    -----
    These thresholds are simplified screening assumptions. They are not
    direct monthly crop-water requirements from FAO.
    """

    crop: str
    minimum_rainfall: float
    maximum_rainfall: float

    def __post_init__(self) -> None:
        """Validate crop rainfall thresholds."""
        if self.minimum_rainfall < 0:
            raise ValueError(
                "Minimum rainfall cannot be negative."
            )

        if self.maximum_rainfall <= self.minimum_rainfall:
            raise ValueError(
                "Maximum rainfall must be greater than "
                "minimum rainfall."
            )

    def classify(self, rainfall: float) -> str:
        """Classify a month's rainfall for this crop.

        Returns
        -------
        str
            'Drought risk', 'Good', or 'Waterlogging risk'.
        """
        if rainfall < self.minimum_rainfall:
            return "Drought risk"

        if rainfall > self.maximum_rainfall:
            return "Waterlogging risk"

        return "Good"


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """Calculate cosine similarity between two vectors.

    Formula
    -------
    cosine similarity = dot(A, B) / (||A|| * ||B||)
    """
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            "Both vectors must have the same shape."
        )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "Cosine similarity is undefined for a zero vector."
        )

    return float(
        np.dot(a, b) / (norm_a * norm_b)
    )


def verify_cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> Dict[str, float]:
    """Compare our cosine similarity with SciPy's implementation."""
    ours = cosine_similarity(vector_a, vector_b)

    scipy_distance = scipy_cosine(
        vector_a,
        vector_b
    )

    scipy_similarity = 1 - scipy_distance

    return {
        "our_similarity": ours,
        "scipy_similarity": float(scipy_similarity),
        "absolute_difference": abs(
            ours - scipy_similarity
        ),
    }


def region_comparison(
    regions: List[Region]
) -> Dict[str, np.ndarray]:
    """Calculate pairwise regional comparison matrices.

    Returns
    -------
    dict
        Contains cosine similarity, Pearson correlation,
        and Euclidean distance matrices.
    """
    number_of_regions = len(regions)

    cosine_matrix = np.zeros(
        (number_of_regions, number_of_regions)
    )

    pearson_matrix = np.zeros(
        (number_of_regions, number_of_regions)
    )

    euclidean_matrix = np.zeros(
        (number_of_regions, number_of_regions)
    )

    for i in range(number_of_regions):

        for j in range(number_of_regions):

            a = regions[i].rainfall
            b = regions[j].rainfall

            cosine_matrix[i, j] = cosine_similarity(
                a,
                b
            )

            pearson_matrix[i, j] = np.corrcoef(
                a,
                b
            )[0, 1]

            euclidean_matrix[i, j] = np.linalg.norm(
                a - b
            )

    return {
        "cosine": cosine_matrix,
        "pearson": pearson_matrix,
        "euclidean": euclidean_matrix,
    }


def detect_rainy_seasons(
    region: Region,
    prominence: float = 20.0,
    distance: int = 2
) -> Dict[str, object]:
    """Detect rainfall peaks using scipy.signal.find_peaks.

    Parameters
    ----------
    region:
        Region to analyse.
    prominence:
        Minimum prominence required for a peak.
    distance:
        Minimum number of months between peaks.

    Returns
    -------
    dict
        Peak positions, months, rainfall amounts, and classification.
    """
    peaks, properties = find_peaks(
        region.rainfall,
        prominence=prominence,
        distance=distance
    )

    number_of_peaks = len(peaks)

    if number_of_peaks == 1:
        pattern = "Unimodal"

    elif number_of_peaks == 2:
        pattern = "Bimodal"

    elif number_of_peaks > 2:
        pattern = "Complex"

    else:
        pattern = "No clear peak"

    return {
        "region": region.name,
        "peak_indices": peaks,
        "peak_months": [
            MONTHS[index]
            for index in peaks
        ],
        "peak_rainfall": region.rainfall[peaks],
        "pattern": pattern,
        "prominences": properties.get(
            "prominences",
            np.array([])
        ),
    }


def suitability_table(
    regions: List[Region],
    crops: List[CropRule]
) -> Dict[str, Dict[str, List[str]]]:
    """Classify each month for every region and crop."""
    results: Dict[
        str,
        Dict[str, List[str]]
    ] = {}

    for region in regions:

        results[region.name] = {}

        for crop in crops:

            classifications = [
                crop.classify(rainfall)
                for rainfall in region.rainfall
            ]

            results[
                region.name
            ][
                crop.crop
            ] = classifications

    return results
