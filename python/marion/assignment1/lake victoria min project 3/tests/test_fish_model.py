"""
tests/test_fish_model.py
Unit tests for FishStock, PriceModel, and RiskAssessor classes.
"""

import pytest
import numpy as np
from src.fish_model import FishStock, PriceModel, RiskAssessor


def test_fish_stock_msy_and_simulation():
    model = FishStock(r=0.4, K=10000.0, initial_stock=4000.0)
    assert model.maximum_sustainable_yield == 1000.0

    stocks, harvests = model.simulate(weeks=1, harvest_rate=0.10)
    assert np.isclose(harvests[0], 400000.0)
    assert np.isclose(stocks[1], 4560.0)


def test_price_model_bounds():
    price_model = PriceModel(initial_price=12000.0, min_price=9000.0, max_price=16000.0)
    prices = price_model.simulate_path(weeks=100, seed=42)
    assert np.all(prices >= 9000.0)
    assert np.all(prices <= 16000.0)


def test_risk_assessor_stats_and_classification():
    data = [10.0, 20.0, 30.0]
    stats = RiskAssessor.calculate_stats(data)
    assert stats["mean"] == 20.0
    assert stats["variance"] == 100.0
    assert stats["cv"] == 0.5
    assert RiskAssessor.classify_risk(stats["cv"]) == "High Risk"


def test_invalid_input_exceptions():
    with pytest.raises(ValueError, match="Carrying capacity 'K' must be positive"):
        FishStock(K=-100.0)

    with pytest.raises(ValueError, match="Harvest rate 'h' must be between 0.0 and 1.0"):
        model = FishStock()
        model.simulate(weeks=10, harvest_rate=1.5)