import numpy as np
import pandas as pd
import pytest

from backend.trend_research import simulate, study


def prices():
    dates = pd.bdate_range("2015-01-02", periods=1100)
    returns = np.r_[np.full(420, 0.001), np.full(180, -0.006), np.full(500, 0.002)]
    return dates, 100 * np.exp(np.cumsum(returns))


def test_buy_hold_matches_independent_net_price_ratio():
    dates, values = prices()
    start, end, bps = 400, 1050, 11
    result = simulate(dates, values, start, end, cost_bps=bps)
    expected = values[end] / values[start] * (1 - bps / 10000) / (1 + bps / 10000) - 1
    assert result["metrics"]["return"] == pytest.approx(expected)
    assert result["order_count"] == 2


def test_signal_delay_and_no_future_observation_leakage():
    dates, values = prices()
    end = 850
    a = simulate(dates, values, 400, end, months=10, delay=2)
    altered = values.copy()
    altered[end + 1 :] *= 100
    assert simulate(dates, altered, 400, end, months=10, delay=2) == a
    for order in a["orders"]:
        if order["signal_date"] == "predeclared test end":
            continue
        signal = dates.get_loc(pd.Timestamp(order["signal_date"]))
        executed = dates.get_loc(pd.Timestamp(order["date"]))
        assert executed >= signal + 2
    assert {o["side"] for o in a["orders"]} == {"buy", "sell"}


def test_study_aligns_comparisons_and_cost_stress():
    dates, values = prices()
    result = study(dates, values)
    variants = result["variants"]
    for value in variants.values():
        assert value["trace"][0]["date"] == result["evaluation_start"]
        assert value["trace"][-1]["date"] == result["evaluation_end"]
    assert (
        variants["10-month trend, double costs"]["metrics"]["return"] < variants["10-month trend"]["metrics"]["return"]
    )


def test_missing_duplicate_and_insufficient_history_are_rejected():
    dates, values = prices()
    missing = values.copy()
    missing[30] = np.nan
    with pytest.raises(ValueError, match="missing data"):
        study(dates, missing)
    with pytest.raises(ValueError, match="unique"):
        study(dates.insert(1, dates[0]), np.insert(values, 1, values[0]))
    with pytest.raises(ValueError, match="twelve completed"):
        study(dates[:100], values[:100])
