import numpy as np
import pytest

from backend.config import Asset, RunConfig
from backend.engine import Engine, library_path
from backend.risk import StreamingRisk


def test_reproducible_across_threads_and_batches():
    config = RunConfig(paths=10000, steps=12, threads=1)
    a = Engine(config, "cpu").batch(0, 10000, True)
    other = Engine(config.model_copy(update={"threads": 8}), "cpu")
    b = np.concatenate([other.batch(0, 3333, True), other.batch(3333, 6667, True)])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, Engine(config.model_copy(update={"seed": 43}), "cpu").batch(0, 10000, True))


def test_analytic_lognormal_moments_and_correlation():
    config = RunConfig(threads=8, steps=4)
    scenarios = Engine(config, "cpu").batch(0, 200_000, True)
    logs = np.log(scenarios[:, 1:])
    means = np.array([a.mu - 0.5 * a.sigma**2 for a in config.assets]) * config.horizon
    vol = np.array([a.sigma for a in config.assets])
    expected_cov = np.asarray(config.correlation) * np.outer(vol, vol) * config.horizon
    # Six standard errors for marginal means; covariance tolerance below 1% annual variance.
    np.testing.assert_array_less(np.abs(logs.mean(axis=0) - means), 6 * vol / np.sqrt(len(logs)))
    np.testing.assert_allclose(np.cov(logs, rowvar=False), expected_cov, atol=0.0007)
    expected_value = sum(a.weight * np.exp(a.mu * config.horizon) for a in config.assets)
    assert abs(scenarios[:, 0].mean() - expected_value) < 0.004


def test_zero_volatility_exact_solution():
    config = RunConfig(assets=[Asset(name="Cash", weight=1, mu=0.03, sigma=0)], correlation=[[1]], horizon=2, steps=24)
    out = Engine(config, "cpu").batch(0, 100)
    np.testing.assert_allclose(out, np.exp(0.06), rtol=1e-14)


def test_histogram_var_and_cvar_bounds():
    rng = np.random.default_rng(91)
    ratios = np.exp(rng.normal(0.04, 0.2, 100000))
    risk = StreamingRisk(100000, 0.99)
    for batch in np.array_split(ratios, 11):
        risk.add(batch)
    result = risk.result()
    losses = np.sort((1 - ratios) * 100000)[::-1]
    exact_var = losses[999]
    assert result["var_bounds"][0] <= exact_var <= result["var_bounds"][1]
    assert abs(result["cvar"] - losses[:1000].mean()) <= result["cvar_error_bound"]
    assert sum(x["count"] for x in result["histogram"]) == len(ratios)
    assert result["histogram_overflow"] == 0


def test_invalid_portfolios_are_rejected():
    with pytest.raises(ValueError):
        RunConfig(correlation=[[1]])
    with pytest.raises(ValueError):
        RunConfig(assets=[Asset(name="Bad", weight=0.9, mu=0, sigma=0)], correlation=[[1]])
    with pytest.raises(ValueError):
        RunConfig(correlation=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    with pytest.raises(ValueError):
        RunConfig(ram_gb=float("nan"))


def test_histogram_out_of_range_fails_explicitly():
    risk = StreamingRisk(100000, 0.99)
    with pytest.raises(ValueError, match="histogram range"):
        risk.add(np.array([np.exp(21)]))


@pytest.mark.skipif(not library_path("gpu").exists(), reason="CUDA engine not compiled")
def test_cpu_gpu_agreement():
    config = RunConfig(steps=12, threads=8)
    cpu = Engine(config, "cpu").batch(0, 10000, True)
    gpu = Engine(config, "gpu").batch(0, 10000, True)
    np.testing.assert_allclose(cpu, gpu, rtol=1e-10, atol=1e-12)
