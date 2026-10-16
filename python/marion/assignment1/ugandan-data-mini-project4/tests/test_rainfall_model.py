import numpy as np
import pytest

from src.rainfall_model import (
    Region,
    CropRule,
    cosine_similarity,
)


def test_region_annual_total():
    rainfall = np.array([
        10, 20, 30, 40, 50, 60,
        70, 80, 90, 100, 110, 120
    ])

    region = Region(
        name="Test",
        rainfall=rainfall
    )

    assert region.annual_total() == 780


def test_region_empty_rainfall():
    with pytest.raises(ValueError):
        Region(
            name="Empty",
            rainfall=np.array([])
        )


def test_crop_rule_good_rainfall():
    crop = CropRule(
        crop="Maize",
        minimum_rainfall=50,
        maximum_rainfall=200
    )

    assert crop.classify(100) == "Good"


def test_crop_rule_drought_risk():
    crop = CropRule(
        crop="Maize",
        minimum_rainfall=50,
        maximum_rainfall=200
    )

    assert crop.classify(20) == "Drought risk"


def test_cosine_similarity_identical_vectors():
    rainfall = np.array([10, 20, 30])

    assert cosine_similarity(
        rainfall,
        rainfall
    ) == pytest.approx(1.0)
