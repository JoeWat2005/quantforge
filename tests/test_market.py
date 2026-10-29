import json
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from backend import market_data
from backend.market_analysis import (
    analyze_basket,
    basket_backtest,
    calendar_positions,
    exchange_traded_candidate,
    generate_baskets,
    initialize_baskets,
    joint_factors,
    longest_complete,
    pairwise_tile,
    partial_tile,
)


@pytest.mark.parametrize("settings", [{"engine": "gpu"}, {"require_cpu_temperature": False}])
def test_market_rejects_unimplemented_engine_and_disabled_sensor_guard(settings):
    with pytest.raises(ValidationError):
        market_data.MarketConfig(**settings)


def test_missing_observation_correlation_matches_independent_pairs():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(120, 9))
    data[rng.random(data.shape) < 0.15] = np.nan
    corr, counts = pairwise_tile(data, 2, 7, 40)
    for a in range(2, 7):
        for b in range(9):
            mask = np.isfinite(data[:, a]) & np.isfinite(data[:, b])
            assert counts[a - 2, b] == mask.sum()
            assert corr[a - 2, b] == pytest.approx(np.corrcoef(data[mask, a], data[mask, b])[0, 1])


def test_reference_calendar_ignores_stray_dates_without_filling_missing_sessions():
    calendar = np.array(["2026-10-23", "2026-10-26", "2026-10-27"])
    observed = np.array(["2026-10-22", "2026-10-23", "2026-10-25", "2026-10-27", "2026-10-28"])
    positions, valid = calendar_positions(calendar, observed)
    np.testing.assert_array_equal(positions, [0, 2])
    np.testing.assert_array_equal(valid, [False, True, False, True, False])


@pytest.mark.parametrize("shape", [(100, 20), (30, 70)])
def test_precision_matches_direct_inverse_including_woodbury(shape):
    rng = np.random.default_rng(7)
    data = rng.normal(size=shape)
    loadings, fractions, model = joint_factors(data, 0.2)
    z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    precision = np.linalg.inv(0.8 * (z.T @ z / (len(z) - 1)) + 0.2 * np.eye(shape[1]))
    reference = -precision / np.sqrt(np.diag(precision)[:, None] * np.diag(precision)[None, :])
    result = partial_tile(model, 0, shape[1])
    mask = ~np.eye(shape[1], dtype=bool)
    np.testing.assert_allclose(result[mask], reference[mask], atol=1e-10)
    assert 0 < fractions.sum() <= 1
    assert loadings.shape == (shape[1], 5)


def test_three_asset_cointegration_and_no_training_lookahead(tmp_path):
    rng = np.random.default_rng(314)
    n = 1400
    x, y = np.cumsum(rng.normal(0, 0.01, size=(2, n)), axis=1)
    spread = rng.normal(0, 0.001, n)
    logs = np.array([4 + x, 4 + y, 4 + 0.5 * x + 0.5 * y + spread])
    path = tmp_path / "prices.npy"
    np.save(path, logs)
    initialize_baskets(path, 1000, 252)
    result = analyze_basket((0, 1, 2))
    assert result["status"] == "ok"
    assert result["nominal_rank"] == 1
    assert result["effective_legs"] == 3
    assert result["heldout_adf_p"] < 0.01
    logs[:, 1000:] = rng.normal(4, 0.2, size=(3, 400))
    alternate = tmp_path / "changed.npy"
    np.save(alternate, logs)
    initialize_baskets(alternate, 1000, 252)
    changed = analyze_basket((0, 1, 2))
    np.testing.assert_array_equal(result["weights"], changed["weights"])
    assert result["training_score"] == changed["training_score"]


def test_baskets_reproducible_and_unique():
    core = list(range(20))
    neighbors = [[((a + j + 1) % 20, 0.5) for j in range(10)] for a in core]
    a = generate_baskets(core, neighbors, [3, 5, 8], 200, 42)
    assert a == generate_baskets(core, neighbors, [3, 5, 8], 200, 42)
    assert len(set(a)) == len(a) == 200
    assert {len(b) for b in a} == {3, 5, 8}


def test_no_trading_before_signal_and_no_compression_of_gaps():
    logs = np.log(np.array([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0], [4.0, 1.0, 1.0]]))
    r = basket_backtest(logs, np.array([0.5, -0.25, -0.25]), 0, 0.25, 1, 2, 0, 0)
    assert r["entries"] == 1
    assert r["return"] == pytest.approx(0)
    data = np.ones((10, 3))
    data[3:5] = np.nan
    assert longest_complete(data) == (5, 10)


def test_provider_auth_is_header_only_and_error_is_redacted(monkeypatch):
    from backend import provider_budget

    monkeypatch.setattr(provider_budget, "acquire", lambda cost=1: None)
    monkeypatch.setattr(provider_budget, "configure", lambda credits: None)
    monkeypatch.setattr(market_data, "read_key", lambda: "private-test-key")
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(401, json={"code": 401, "message": "private-test-key rejected"})

    provider = market_data.TwelveClient()
    provider.client.close()
    provider.client = httpx.Client(
        base_url="https://api.twelvedata.com",
        headers={"Authorization": "apikey " + provider.key},
        transport=httpx.MockTransport(respond),
    )
    result = provider.request("/time_series", {"symbol": "SPY"})
    assert seen[0].headers["Authorization"] == "apikey private-test-key"
    assert "private-test-key" not in str(seen[0].url)
    assert "private-test-key" not in json.dumps(result)
    provider.client.close()


def test_cache_resume_preserves_completed_record_and_lease(monkeypatch, tmp_path):
    monkeypatch.setattr(market_data, "MARKET", tmp_path)
    db, _ = market_data.database("123abc")
    db.execute("INSERT INTO instruments(id,symbol,status,priority) VALUES ('id1','SPY','ok',1)")
    db.commit()
    db.close()
    db, _ = market_data.database("123abc")
    assert db.execute("SELECT status FROM instruments WHERE id='id1'").fetchone()[0] == "ok"
    db.close()
    with market_data.collection_lease() as first:
        assert first
        with market_data.collection_lease() as second:
            assert not second
    with market_data.collection_lease() as third:
        assert third


def test_provider_fund_share_classes_do_not_become_trading_candidates():
    assert not exchange_traded_candidate({"type": "ETF", "name": "Growth Fund Class A", "exchange": "NASDAQ"})
    assert exchange_traded_candidate({"type": "ETF", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE"})
    assert exchange_traded_candidate({"type": "Common Stock", "exchange": "NASDAQ"})
    assert not exchange_traded_candidate({"type": "Common Stock", "exchange": "OTC Markets"})


def test_full_analysis_starts_once_when_queue_finishes(monkeypatch, tmp_path):
    from backend import market_service

    monkeypatch.setattr(market_service.lab_store, "LAB", tmp_path / "lab")

    control = tmp_path / "policy.json"
    initial = {"auto_analyze": True, "analysis": market_data.MarketConfig(basket_trials=100000).model_dump()}
    market_data.atomic_json(control, initial)
    monkeypatch.setattr(market_service, "CONTROL", control)
    monkeypatch.setattr(market_service, "collection_state", lambda: {"status": "completed", "snapshot_id": "123abc"})
    starts = []

    class Stop:
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            return self.calls > 2

    class Controller:
        stop = Stop()
        lock = threading.RLock()
        process = None
        telemetry = {"cpu_temperature": 70}

        def start(self, config, module):
            starts.append((config, module))

    market_service.watch_market(Controller())
    assert len(starts) == 1
    assert starts[0][0].snapshot_id == "123abc"
    assert starts[0][0].basket_trials == 100000
    assert starts[0][1] == "backend.market_worker"


def test_atomic_status_retries_windows_file_sharing(monkeypatch, tmp_path):
    original = Path.replace
    attempts = []

    def briefly_locked(path, destination):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("Temporary sharing lock")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", briefly_locked)
    target = tmp_path / "status.json"
    market_data.atomic_json(target, {"completed": 123})
    assert len(attempts) == 3
    assert json.loads(target.read_text()) == {"completed": 123}


def test_flat_price_baseline_and_matched_benchmark_only_lose_modeled_costs():
    import pandas as pd

    from backend.market_worker import portfolio_experiment

    rows = [{"symbol": "SPY" if i == 0 else str(i), "type": "Common Stock", "exchange": "NASDAQ"} for i in range(8)]
    logs = np.full((8, 500), np.log(100.0))
    dates = np.array(pd.bdate_range("2020-01-01", periods=500).strftime("%Y-%m-%d"))
    r = portfolio_experiment(logs, [[i] for i in range(8)], np.full(8, 1e7), np.full(8, 0.01), rows, dates, 300, 10)
    assert r["status"] == "evaluated"
    assert -0.01 < r["return"] < 0
    assert r["matched_benchmark"]["return"] == pytest.approx(0.999 / 1.001 - 1)
    assert r["matched_benchmark"]["start"] == r["start"]
    assert r["matched_benchmark"]["end"] == r["end"]
