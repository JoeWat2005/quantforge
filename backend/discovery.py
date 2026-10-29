"""Relationship discovery with chronological evaluation and native bootstraps."""

import ctypes as ct
import math
import os
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import coint

from backend.engine import library_path

_TRAIN = None


def init_pair_worker(path):
    global _TRAIN
    _TRAIN = np.load(path, mmap_mode="r")


def analyse_pair(pair):
    a, b = pair
    x, y = _TRAIN[:, a], _TRAIN[:, b]
    beta, intercept = np.linalg.lstsq(np.column_stack([y, np.ones(len(y))]), x, rcond=None)[0]
    spread = x - beta * y - intercept
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistic, pvalue, _ = coint(x, y, trend="c", maxlag=1, autolag=None)
    valid = np.isfinite(statistic) and not caught
    prev = spread[:-1] - spread[:-1].mean()
    slope = float(np.dot(prev, np.diff(spread)) / max(np.dot(prev, prev), 1e-30))
    half_life = -math.log(2) / math.log(1 + slope) if -1 < slope < 0 else None
    return {
        "a": int(a),
        "b": int(b),
        "beta": float(beta),
        "intercept": float(intercept),
        "spread_std": float(spread.std(ddof=1)),
        "cointegration_p": float(pvalue) if valid else None,
        "half_life_days": half_life,
        "test_note": "" if valid else "Near duplicate or unstable test; no reliable cointegration p-value",
    }


def screen_pairs(log_prices, split, shortlist):
    returns = np.diff(log_prices[:split], axis=0)
    if (returns.std(axis=0) < 1e-8).any():
        raise ValueError("An asset has constant training prices; remove it before screening")
    corr = np.corrcoef(returns, rowvar=False)
    a, b = np.triu_indices(corr.shape[0], k=1)
    # Only the training window decides which pairs receive expensive tests.
    order = np.argsort(-corr[a, b], kind="stable")[:shortlist]
    return corr, np.column_stack([a[order], b[order]]).astype(np.int32), np.ascontiguousarray(returns)


def bootstrap_correlations(returns, pairs, repetitions, block, seed, threads):
    lib = ct.CDLL(str(library_path("research")))
    lib.qf_bootstrap.argtypes = [
        ct.POINTER(ct.c_double),
        ct.c_int,
        ct.c_int,
        ct.POINTER(ct.c_int),
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_uint64,
        ct.c_int,
        ct.POINTER(ct.c_double),
    ]
    lib.qf_bootstrap.restype = ct.c_int
    returns = np.ascontiguousarray(returns, dtype=np.float64)
    pairs = np.ascontiguousarray(pairs, dtype=np.int32)
    out = np.empty((len(pairs), repetitions))
    code = lib.qf_bootstrap(
        returns.ctypes.data_as(ct.POINTER(ct.c_double)),
        len(returns),
        returns.shape[1],
        pairs.ctypes.data_as(ct.POINTER(ct.c_int)),
        len(pairs),
        repetitions,
        block,
        seed,
        threads,
        out.ctypes.data_as(ct.POINTER(ct.c_double)),
    )
    if code:
        raise RuntimeError(f"Native bootstrap error {code}")
    return np.nanquantile(out, [0.025, 0.975], axis=1).T


def pair_backtest(log_prices, pair, split, cost_bps, borrow_rate):
    """Fixed training hedge/mean/std. Signals at close affect following returns."""
    a, b = pair["a"], pair["b"]
    beta = pair["beta"]
    spread = log_prices[:, a] - beta * log_prices[:, b] - pair["intercept"]
    scale = pair["spread_std"]
    z = spread / max(scale, 1e-12)
    pos = 0.0
    age = 0
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    trades = 0
    costs = 0.0
    pnl = []
    trace = []
    bankrupt = False
    enabled = beta > 0 and scale > 1e-8 and not pair.get("test_note")
    prices = np.exp(log_prices[:, [a, b]])
    units = np.zeros(2)
    rate = cost_bps / 10000
    for i in range(split, len(log_prices)):
        previous_wealth = wealth
        wealth += float(np.dot(units, prices[i] - prices[i - 1]))
        if pos:
            fee = float(np.dot(np.maximum(-units, 0), prices[i - 1])) * borrow_rate / 252
            wealth -= fee
            costs += fee
            age += 1
        if wealth <= 0:
            bankrupt = True
            wealth = 0.0
            units[:] = 0
        new_pos = pos
        if not enabled or bankrupt or i == len(log_prices) - 1:
            new_pos = 0.0
        elif pos and (abs(z[i]) < 0.5 or abs(z[i]) > 4 or age >= 20):
            new_pos = 0.0
        elif not pos and 2 <= abs(z[i]) <= 4:
            new_pos = -float(np.sign(z[i]))
        if new_pos != pos:
            exit_fee = rate * float(np.dot(np.abs(units), prices[i]))
            costs += exit_fee
            wealth = max(0.0, wealth - exit_fee)
            units[:] = 0.0
            if new_pos and wealth > 0:
                capital = wealth / (1 + rate)
                costs += wealth - capital
                wealth = capital
                # Fixed adjusted-price units between entry and exit: no
                # uncharged daily rebalancing is assumed.
                units = new_pos * capital * np.array([1.0, -beta]) / (1 + beta) / prices[i]
                trades += 1
                age = 0
        pos = new_pos
        peak = max(peak, wealth)
        max_drawdown = min(max_drawdown, wealth / peak - 1)
        pnl.append(wealth / previous_wealth - 1 if previous_wealth > 0 else 0.0)
        trace.append({"index": i, "z": float(z[i]), "equity": wealth})
    std = np.std(pnl, ddof=1) if len(pnl) > 1 else 0
    sharpe = float(np.mean(pnl) / std * np.sqrt(252)) if std > 1e-12 else None
    take = np.unique(np.linspace(0, max(0, len(trace) - 1), min(100, len(trace)), dtype=int))
    return {
        "holdout_return": wealth - 1,
        "holdout_max_drawdown": max_drawdown,
        "holdout_sharpe": sharpe,
        "entries": trades,
        "cost_fraction_sum": costs,
        "backtest_enabled": enabled,
        "bankrupt": bankrupt,
        "trace": [trace[i] for i in take],
    }


def rolling_correlations(x, y, window):
    if len(x) < window:
        return []
    values = []
    for end in range(window, len(x) + 1, 21):
        v = np.corrcoef(x[end - window : end], y[end - window : end])[0, 1]
        if np.isfinite(v):
            values.append(float(v))
    return values


def effective_threads(config):
    return min(config.threads, max(1, int((os.cpu_count() or 1) * config.cpu_ceiling / 100)))


def finite_correlation(x, y):
    if min(np.std(x), np.std(y)) < 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def exposure_clusters(correlation, names):
    distances = np.clip(1 - correlation, 0, 2)
    hierarchy = linkage(squareform(distances, checks=False), method="average")
    labels = fcluster(hierarchy, t=0.2, criterion="distance")
    groups = {}
    for label, name in zip(labels, names, strict=True):
        groups.setdefault(int(label), []).append(name)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))
