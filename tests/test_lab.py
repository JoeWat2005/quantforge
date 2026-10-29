from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backend import lab_store
from backend.lab_analysis import factor_residuals, feature_indices, metrics, schedule, training_model
from backend.lab_data import asof_metadata, import_prices, load_dataset, member_asof
from backend.lab_models import ManualDates, ResearchConfig
from backend.lab_portfolio import project_weights, run_portfolio


@pytest.fixture
def local_lab(monkeypatch, tmp_path):
    monkeypatch.setattr(lab_store, "LAB", tmp_path / "lab")
    return lab_store.LAB


def synthetic_input():
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2016-01-04", periods=2400)
    market = rng.normal(0.0002, 0.008, len(dates))
    data = {"Date": dates.strftime("%Y-%m-%d"), "SPY": 100 * np.exp(np.cumsum(market))}
    for i in range(16):
        symbol = f"S{i:02}"
        data[symbol] = 50 * np.exp(np.cumsum(0.6 * market + rng.normal(0.0001, 0.009, len(dates))))
        data[symbol + "_volume"] = np.full(len(dates), 1_000_000)
    data["SPY_volume"] = np.full(len(dates), 10_000_000)
    return pd.DataFrame(data).to_csv(index=False)


@pytest.fixture
def dataset(local_lab):
    return import_prices("Deterministic synthetic tests", synthetic_input(), True)


def test_walk_forward_disjoint_tests_and_purged_dates(dataset):
    body, days, _, _ = load_dataset(dataset["id"])
    config = ResearchConfig(dataset_id=body["id"])
    folds = schedule(days, config)
    assert len(folds) >= 3
    for fold in folds:
        assert fold["validation"][0] - fold["train"][1] - 1 >= config.gap_sessions
        assert fold["test"][0] - fold["validation"][1] - 1 >= config.gap_sessions
    assert all(a["test"][1] < b["test"][0] for a, b in zip(folds, folds[1:]))
    with pytest.raises(ValidationError):
        ResearchConfig(step_months=3, test_months=12)
    with pytest.raises(ValidationError):
        ResearchConfig(gap_sessions=5)


def test_basket_parallelism_obeys_cpu_ram_and_work_limits(monkeypatch):
    from backend import lab_worker

    monkeypatch.setattr(lab_worker, "effective_threads", lambda config: config.threads)
    config = ResearchConfig(threads=128, ram_gb=64)
    platform_limit = 61 if lab_worker.os.name == "nt" else 128
    assert lab_worker.basket_worker_count(config, 100000) == platform_limit
    assert lab_worker.basket_worker_count(config.model_copy(update={"threads": 8}), 100000) == 8
    assert lab_worker.basket_worker_count(config.model_copy(update={"ram_gb": 4}), 100000) == 10
    assert lab_worker.basket_worker_count(config, 33) == 2


def test_explicit_dates_preserved_and_bad_order_rejected(dataset):
    _, days, _, _ = load_dataset(dataset["id"])
    manual = ManualDates(
        train_start=date(2016, 1, 4),
        train_end=date(2018, 12, 31),
        validation_start=date(2019, 2, 1),
        validation_end=date(2019, 12, 31),
        test_start=date(2020, 2, 3),
        test_end=date(2020, 12, 31),
    )
    folds = schedule(days, ResearchConfig(split_mode="manual", manual_dates=manual))
    assert folds[0]["test_start"] == "2020-02-03"
    assert folds[0]["validation_start"] == "2019-02-01"
    with pytest.raises(ValidationError):
        ManualDates(**{**manual.model_dump(), "test_start": date(2018, 1, 1)})


def test_weekly_sampling_uses_common_session_not_asset_forward_fill():
    days = np.array(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-10"])
    np.testing.assert_array_equal(feature_indices(days, 0, 4, "weekly"), [1, 4])


def test_factor_adjustment_removes_common_market_component():
    rng = np.random.default_rng(12)
    market = rng.normal(size=2000)
    returns = np.column_stack([market + rng.normal(0, 0.1, 2000), 2 * market + rng.normal(0, 0.1, 2000)])
    residual, betas = factor_residuals(returns, market[:, None])
    assert np.corrcoef(returns.T)[0, 1] > 0.98
    assert abs(np.corrcoef(residual.T)[0, 1]) < 0.08
    np.testing.assert_allclose(betas, [[1, 2]], atol=0.02)


def test_all_training_choices_unchanged_by_future_prices(dataset):
    body, days, logs, volume = load_dataset(dataset["id"])
    config = ResearchConfig(dataset_id=body["id"], max_assets=12)
    fold = schedule(days, config)[0]
    first = training_model(body, days, logs, volume, fold, config)
    changed = np.array(logs)
    changed[:, fold["train"][1] + 1 :] += np.random.default_rng(2).normal(
        0, 3, changed[:, fold["train"][1] + 1 :].shape
    )
    second = training_model(body, days, changed, volume, fold, config)
    assert first["indices"] == second["indices"]
    np.testing.assert_array_equal(first["weights"], second["weights"])
    assert first["relationships"] == second["relationships"]


def portfolio_config(**kwargs):
    return ResearchConfig(
        commission_bps=0, spread_bps=0, slippage_bps=0, impact_coefficient=0, borrow_rate=0, max_weight=0.5, **kwargs
    )


def one_asset(prices, config, **kwargs):
    n = len(prices)
    model = {"weights": np.array([0.5]), "attributes": [{}], "turnover": np.array([1e8]), "volatility": np.array([0.2])}
    return run_portfolio(
        np.log(np.asarray(prices)[None, :]),
        np.array(pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")),
        [{"symbol": "X"}],
        model,
        0,
        n - 1,
        config,
        **kwargs,
    )


def test_portfolio_fixed_price_units_and_transaction_budget():
    result = one_asset([100, 120, 120], portfolio_config())
    assert result["metrics"]["return"] == pytest.approx(0.1)
    config = portfolio_config().model_copy(update={"commission_bps": 10})
    result = one_asset([100, 100, 100], config)
    assert result["metrics"]["return"] == pytest.approx((1 - 0.001 * 0.5) / (1 + 0.001 * 0.5) - 1)
    assert result["fees_fraction"] == pytest.approx(-result["metrics"]["return"])
    assert result["trades"][-1]["reason"] == "scheduled_end"


def test_signal_cannot_trade_at_its_observation_close():
    basket = {"indices": [0], "weights": [1.0], "training_mean": 0.0, "training_std": 1.0}
    result = one_asset(np.exp([2.5, 2.5, 3, 3]), portfolio_config(), family="baskets", baskets=[basket])
    assert result["daily_returns"][:2] == [0, 0]
    assert result["metrics"]["return"] == pytest.approx(-0.5 * (np.exp(0.5) - 1))
    assert result["trades"][0]["date"] == "2020-01-02"


def test_missing_held_prices_invalidate_not_disappear():
    result = one_asset([100, 101, np.nan, 102], portfolio_config())
    assert result["status"] == "invalid_missing_prices"
    assert result["metrics"]["return"] is None
    assert result["actual_end"] == "2020-01-02"


def test_risk_caps_scale_entire_basket_preserving_coefficients():
    config = ResearchConfig(max_gross=1, max_net=0.2, max_weight=0.15, sector_cap=0.2)
    raw = np.array([0.8, -0.5, 0.4])
    result = project_weights(raw, config, [{"sector": "A"}, {"sector": "A"}, {}])
    assert np.abs(result).sum() <= 1
    assert abs(result.sum()) <= 0.2
    assert np.abs(result).max() <= 0.15
    assert abs(result[0]) + abs(result[1]) <= 0.2 + 1e-12
    np.testing.assert_allclose(result / result[0], raw / raw[0])


def test_metadata_and_membership_are_asof_not_future():
    body = {
        "security": [
            {"symbol": "A", "security_id": "one", "known_at": "2020-01-01"},
            {"symbol": "A", "security_id": "two", "known_at": "2024-01-01"},
        ],
        "membership": [{"symbol": "A", "start": "2019-01-01", "end": "2022-01-01", "known_at": "2019-01-01"}],
    }
    assert asof_metadata(body, "A", "2021-01-01")["security_id"] == "one"
    assert member_asof(body, "A", "2021-01-01")
    assert not member_asof(body, "A", "2023-01-01")


def test_persistent_queue_and_restart_recovery(dataset):
    config = ResearchConfig(dataset_id=dataset["id"])
    value = lab_store.create(config)
    lab_store.queue(value["id"])
    assert lab_store.next_queued() == value["id"]
    with pytest.raises(ValueError):
        lab_store.queue(value["id"])
    lab_store.transition(value["id"], "running")
    lab_store.recover()
    assert lab_store.experiment(value["id"])["status"] == "interrupted"
    lab_store.queue(value["id"])
    assert lab_store.experiment(value["id"])["status"] == "queued"


def test_import_refuses_uncertain_prices_and_duplicate_headers(local_lab):
    with pytest.raises(ValueError):
        import_prices("x", synthetic_input(), False)
    text = synthetic_input().replace("S00,", "SPY,", 1)
    with pytest.raises(ValueError, match="duplicate"):
        import_prices("x", text, True)


def test_risk_summary_compounds_returns_and_does_not_hide_gaps():
    assert metrics([0.1, -0.1])["return"] == pytest.approx(-0.01)
    assert metrics([0.1, np.nan])["return"] is None


def test_paper_rejects_historical_backfill(dataset):
    from backend.lab_paper import create_paper, update_paper

    experiment = lab_store.create(ResearchConfig(dataset_id=dataset["id"]))
    lab_store.save_report(
        experiment["id"],
        {
            "config": experiment["config"],
            "deployment": {"asof": dataset["end"], "holdings": [{"symbol": "SPY", "weight": 0.5}]},
        },
    )
    lab_store.transition(experiment["id"], "completed")
    paper = create_paper(experiment["id"], "Forward only")
    result = update_paper(paper["id"], dataset["id"])
    assert result["ledger"] == []
    assert "No new" in result["update_note"]
    assert lab_store.paper(paper["id"])["ledger"] == []


def test_forward_paper_is_idempotent_and_rejects_changed_history(dataset):
    from backend.lab_paper import create_paper, update_paper

    experiment = lab_store.create(ResearchConfig(dataset_id=dataset["id"]))
    lab_store.save_report(
        experiment["id"],
        {
            "config": experiment["config"],
            "deployment": {"asof": "2024-12-31", "holdings": [{"symbol": "SPY", "weight": 0.15}]},
        },
    )
    lab_store.transition(experiment["id"], "completed")
    paper = create_paper(experiment["id"], "Historical fixture emulating an older creation date")
    paper["start_after"] = "2024-12-31"  # Test fixture only: the API cannot backdate creation.
    lab_store.save_paper(paper, "test_fixture")
    updated = update_paper(paper["id"], dataset["id"])
    assert updated["ledger"] and updated["ledger"][0]["date"] > "2024-12-31"
    again = update_paper(paper["id"], dataset["id"])
    assert len(again["ledger"]) == len(updated["ledger"])
    frame = pd.read_csv(__import__("io").StringIO(synthetic_input()))
    position = frame.index[frame["Date"] > "2024-12-31"][2]
    frame.loc[position, "SPY"] *= 1.4
    revised = import_prices("Deliberately inconsistent revision", frame.to_csv(index=False), True)
    with pytest.raises(ValueError, match="Revised data"):
        update_paper(paper["id"], revised["id"])
    assert lab_store.paper(paper["id"])["ledger"] == updated["ledger"]


def test_paper_provider_refresh_waits_for_real_forward_dates(dataset, monkeypatch):
    from backend import lab_feed
    from backend.lab_paper import create_paper

    experiment = lab_store.create(ResearchConfig(dataset_id=dataset["id"]))
    lab_store.save_report(
        experiment["id"],
        {
            "config": experiment["config"],
            "deployment": {"asof": dataset["end"], "holdings": [{"symbol": "SPY", "weight": 0.15}]},
        },
    )
    lab_store.transition(experiment["id"], "completed")
    paper = create_paper(experiment["id"], "No future prices fixture")

    def forbidden():
        raise AssertionError("A same-day paper refresh must not use provider credits")

    monkeypatch.setattr(lab_feed, "TwelveClient", forbidden)
    lab_feed.refresh(paper["id"])
    assert lab_feed.feed_status()["status"] == "completed"
    assert lab_store.paper(paper["id"])["ledger"] == []
    assert lab_feed.refresh_rate({"status": "running", "credits_per_minute": 55, "paced_requests_per_minute": 49}) == (
        6,
        5,
    )
    with pytest.raises(ValueError):
        lab_feed.refresh_rate({"status": "running", "credits_per_minute": 8, "paced_requests_per_minute": 7})


def test_full_catalogue_waiting_experiment_attaches_once(dataset, monkeypatch, tmp_path):
    import threading

    from backend import lab_service, warehouse

    monkeypatch.setattr(warehouse, "WAREHOUSE", tmp_path / "warehouse")

    value = lab_store.create(ResearchConfig(dataset_id=dataset["id"], await_snapshot_id="abc123"))
    lab_store.queue(value["id"])
    assert lab_store.experiment(value["id"])["status"] == "waiting_data"
    monkeypatch.setattr(lab_service, "collection_state", lambda: {"status": "completed", "snapshot_id": "abc123"})
    monkeypatch.setattr(lab_service, "capture_cache", lambda name: dataset)

    class Stop:
        count = 0

        def wait(self, seconds):
            self.count += 1
            return self.count > 2

    class Control:
        stop = Stop()
        lock = threading.RLock()
        process = None
        telemetry = {"cpu_temperature": 60}
        calls = []

        def start(self, config, module):
            self.calls.append(config)

    control = Control()
    lab_service.watch_research(control)
    assert len(control.calls) == 1
    assert control.calls[0].dataset_id == dataset["id"]
    assert control.calls[0].await_snapshot_id == ""


def test_test_window_failure_invalidates_combined_result_instead_of_dropping_fold():
    from backend.lab_worker import combine

    days = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    value = {
        "status": "invalid_missing_prices",
        "reason": "missing",
        "daily_returns": [0.1],
        "fees_fraction": 0,
        "borrow_fraction": 0,
        "turnover": 0,
        "trades": [],
    }
    checkpoint = {
        "fold": {"number": 1},
        "test": [0, 3],
        "books": {name: value for name in ["selected", "double_cost", "diversified", "equal_weight"]},
    }
    combined, _, _, _ = combine([checkpoint], days, ResearchConfig())
    assert combined["selected"]["metrics"]["return"] is None
    assert combined["selected"]["daily_returns"] == [0.1, None, None, None]
