import numpy as np
import pandas as pd
import pytest

from backend.discovery import analyse_pair, bootstrap_correlations, init_pair_worker, pair_backtest, screen_pairs
from backend.discovery_data import DiscoveryConfig, parse_prices_csv


def test_native_bootstrap_thread_reproducibility_and_known_relationship():
    rng = np.random.default_rng(101)
    x = rng.normal(0, 0.01, 500)
    returns = np.column_stack([x, x, x * 0.8 + rng.normal(0, 0.005, 500)])
    pairs = np.array([[0, 1], [0, 2]], dtype=np.int32)
    a = bootstrap_correlations(returns, pairs, 250, 20, 42, 1)
    b = bootstrap_correlations(returns, pairs, 250, 20, 42, 8)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[0], [1, 1], atol=1e-12)
    assert 0.6 < a[1, 0] < a[1, 1] < 1


def test_selection_cannot_see_holdout():
    rng = np.random.default_rng(7)
    prices = np.cumsum(rng.normal(size=(800, 6)), axis=0)
    corr, pairs, _ = screen_pairs(prices, 560, 8)
    prices[560:] = rng.normal(size=(240, 6)) * 100
    other_corr, other_pairs, _ = screen_pairs(prices, 560, 8)
    np.testing.assert_array_equal(corr, other_corr)
    np.testing.assert_array_equal(pairs, other_pairs)


def test_cointegration_detects_synthetic_stationary_spread(tmp_path):
    rng = np.random.default_rng(18)
    common = np.cumsum(rng.normal(0, 0.01, 800))
    residual = np.zeros(800)
    for i in range(1, 800):
        residual[i] = 0.65 * residual[i - 1] + rng.normal(0, 0.002)
    logs = np.column_stack([common + residual, common])
    path = tmp_path / "training.npy"
    np.save(path, logs)
    init_pair_worker(path)
    result = analyse_pair((0, 1))
    assert result["cointegration_p"] < 0.001
    assert 0.9 < result["beta"] < 1.1
    assert result["half_life_days"] > 0


def test_signals_do_not_trade_the_return_that_generated_them():
    logs = np.zeros((20, 2))
    logs[10:, 0] = 1
    logs[11:, 0] = 0.95
    pair = {"a": 0, "b": 1, "beta": 1.0, "intercept": 0.0, "spread_std": 0.4, "test_note": ""}
    result = pair_backtest(logs, pair, 10, 0, 0)
    assert result["trace"][0]["equity"] == 1
    assert result["entries"] == 1
    assert result["holdout_return"] > 0


def test_csv_quality_and_universe_validation():
    with pytest.raises(ValueError, match="duplicate dates"):
        parse_prices_csv("Date,A,B,C\n2020-01-01,1,2,3\n2020-01-01,2,3,4\n")
    with pytest.raises(ValueError, match="positive"):
        parse_prices_csv("Date,A,B,C\n2020-01-01,1,-2,3\n")
    with pytest.raises(ValueError):
        DiscoveryConfig(tickers=["AAPL", "MSFT", "../../evil"])
    dates = pd.date_range("2020-01-01", periods=600, freq="B")
    data = pd.DataFrame({"A": np.arange(600) + 10, "B": np.arange(600) + 20, "C": np.arange(600) + 30}, index=dates)
    data.index.name = "Date"
    parsed = parse_prices_csv(data.to_csv())
    assert parsed.shape == (600, 3)


def test_near_identical_assets_share_an_exposure_group():
    from backend.discovery import exposure_clusters

    corr = np.array([[1, 0.999, 0.1], [0.999, 1, 0.1], [0.1, 0.1, 1]])
    groups = exposure_clusters(corr, ["ETF_A", "ETF_B", "OTHER"])
    assert ["ETF_A", "ETF_B"] in groups
