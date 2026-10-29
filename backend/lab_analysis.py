"""Chronological plans, training-only return models and portfolio diagnostics."""

import math
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from backend.lab_data import asof_metadata, member_asof
from backend.market_analysis import (
    exchange_traded_candidate,
    graph_groups,
    joint_factors,
    pairwise_tile,
    partial_tile,
    strongest_neighbors,
)

FACTORS = ["SPY", "XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE", "IWM", "TLT"]


def schedule(dates, config):
    days = np.asarray(dates, dtype="U10")
    start = str(config.start_date or days[0])
    end = str(config.end_date or days[-1])
    if start < days[0] or end > days[-1]:
        raise ValueError(f"Requested dates exceed dataset coverage {days[0]} → {days[-1]}")
    folds = []

    def add(bounds):
        tr, va, te = [np.flatnonzero((days >= a) & (days < b)) for a, b in bounds]
        va = va[config.gap_sessions :]
        te = te[config.gap_sessions :]
        if len(tr) < 252 or len(va) < 63 or len(te) < 63:
            raise ValueError("Each fold needs 252 training and 63 validation/test sessions after boundary gaps")
        folds.append(
            {
                "number": len(folds) + 1,
                "train": [int(tr[0]), int(tr[-1])],
                "validation": [int(va[0]), int(va[-1])],
                "test": [int(te[0]), int(te[-1])],
                **{
                    name + "_start": str(days[values[0]])
                    for name, values in [("train", tr), ("validation", va), ("test", te)]
                },
                **{
                    name + "_end": str(days[values[-1]])
                    for name, values in [("train", tr), ("validation", va), ("test", te)]
                },
            }
        )

    if config.split_mode == "manual":
        manual = config.manual_dates
        bounds = [
            (
                str(getattr(manual, name + "_start")),
                str(pd.Timestamp(getattr(manual, name + "_end")) + pd.Timedelta(days=1))[:10],
            )
            for name in ("train", "validation", "test")
        ]
        if bounds[0][0] < start or bounds[-1][1] > str(pd.Timestamp(end) + pd.Timedelta(days=1))[:10]:
            raise ValueError("Manual dates exceed the requested dataset period")
        # Explicit manual dates are exact; require the gap rather than silently moving them.
        positions = [np.flatnonzero((days >= a) & (days < b)) for a, b in bounds]
        if any(len(p) == 0 for p in positions):
            raise ValueError("A manual window contains no sessions")
        if (
            positions[1][0] - positions[0][-1] - 1 < config.gap_sessions
            or positions[2][0] - positions[1][-1] - 1 < config.gap_sessions
        ):
            raise ValueError("Manual windows need the configured number of excluded sessions between them")
        # add() removes implicit gaps in rolling mode only.
        # Build directly to preserve explicitly supplied endpoints.
        tr, va, te = positions
        if len(tr) < 252 or len(va) < 63 or len(te) < 63:
            raise ValueError("Manual windows need 252 training and 63 validation/test sessions")
        folds.append(
            {
                "number": 1,
                "train": [int(tr[0]), int(tr[-1])],
                "validation": [int(va[0]), int(va[-1])],
                "test": [int(te[0]), int(te[-1])],
                **{name + "_start": str(days[p[0]]) for name, p in zip(("train", "validation", "test"), positions)},
                **{name + "_end": str(days[p[-1]]) for name, p in zip(("train", "validation", "test"), positions)},
            }
        )
    else:
        base = pd.Timestamp(start)
        offset = 0
        while True:
            origin = base + pd.DateOffset(months=offset)
            train_end = origin + pd.DateOffset(years=config.train_years)
            val_end = train_end + pd.DateOffset(months=config.validation_months)
            test_end = val_end + pd.DateOffset(months=config.test_months)
            if test_end > pd.Timestamp(end) + pd.Timedelta(days=1):
                break
            if len(folds) == 30:
                raise ValueError("This plan exceeds 30 windows; increase the step or narrow the dates")
            train_start = base if config.split_mode == "expanding" else origin
            add(
                [
                    (str(train_start.date()), str(train_end.date())),
                    (str(train_end.date()), str(val_end.date())),
                    (str(val_end.date()), str(test_end.date())),
                ]
            )
            offset += config.step_months
    if not folds:
        raise ValueError(
            "No complete fold fits these dates. Reduce training/validation/test lengths or use a longer dataset."
        )
    return folds


def feature_indices(dates, start, end, frequency):
    indices = np.arange(start, end + 1)
    if frequency == "daily":
        return indices
    periods = pd.DatetimeIndex(dates[indices]).to_period("W-FRI")
    # Choose one common last reference session for each week, never each asset's last nonmissing value.
    return indices[np.r_[np.asarray(periods[1:] != periods[:-1]), True]]


def factor_residuals(training_returns, factor_returns, ridge=0.001):
    """Fit all coefficients and normalizations only on supplied training observations."""
    x = np.asarray(factor_returns, float)
    if x.shape[1] == 0:
        return training_returns.copy(), np.empty((0, training_returns.shape[1]))
    means = np.nanmean(x, axis=0)
    deviations = np.nanstd(x, axis=0, ddof=1)
    standardized = (x - means) / np.maximum(deviations, 1e-12)
    residual = np.full_like(training_returns, np.nan)
    coefficients = []
    for j, y in enumerate(training_returns.T):
        valid = np.isfinite(y) & np.isfinite(standardized).all(axis=1)
        a = np.column_stack([np.ones(valid.sum()), standardized[valid]])
        penalty = np.eye(a.shape[1]) * ridge
        penalty[0, 0] = 0
        beta = np.linalg.solve(a.T @ a + penalty, a.T @ y[valid])
        residual[valid, j] = y[valid] - a @ beta
        coefficients.append(beta[1:] / np.maximum(deviations, 1e-12))
    return residual, np.array(coefficients).T


def stationarity_p(series):
    if not np.isfinite(series).all() or len(series) < 80 or np.std(series) < 1e-10:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(adfuller(series, maxlag=1, autolag=None, regression="c")[1])
        return value if np.isfinite(value) else None
    except (ValueError, np.linalg.LinAlgError):
        return None


def half_life(spread):
    x = spread[:-1]
    y = spread[1:]
    x = x - x.mean()
    y = y - y.mean()
    phi = float(x @ y / max(float(x @ x), 1e-20))
    return -math.log(2) / math.log(phi) if 0 < phi < 1 else None


def training_model(body, dates, logs, volumes, fold, config):
    a, b = fold["train"]
    feature = feature_indices(dates, a, b, config.frequency)
    returns = np.diff(logs[:, feature], axis=1).T
    coverage = np.isfinite(returns).mean(axis=0)
    activity = (np.abs(np.nan_to_num(returns)) > 1e-10).sum(axis=0) / np.maximum(1, np.isfinite(returns).sum(axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vol = np.nanstd(returns, axis=0, ddof=1)
        turnover = np.nanmedian(np.exp(logs[:, a : b + 1]) * volumes[:, a : b + 1], axis=1)
    candidates = []
    exclusions = {"coverage_or_variance": 0, "universe_or_type": 0, "liquidity": 0, "duplicate": 0, "user_filter": 0}
    attrs = {}
    keys = set()
    signatures = set()
    ordered = sorted(
        range(len(body["records"])),
        key=lambda i: (-float(turnover[i]) if np.isfinite(turnover[i]) else 0, body["records"][i]["symbol"], i),
    )
    for i in ordered:
        row = body["records"][i]
        symbol = row["symbol"]
        if (config.symbols and symbol not in config.symbols) or symbol in config.exclude_symbols:
            exclusions["user_filter"] += 1
            continue
        if coverage[i] < config.minimum_coverage or not np.isfinite(vol[i]) or vol[i] < 1e-8 or activity[i] < 0.1:
            exclusions["coverage_or_variance"] += 1
            continue
        metadata = asof_metadata(body, symbol, str(dates[b]))
        attrs[i] = metadata
        if config.stocks_only and (metadata.get("instrument_type") or row.get("type")) == "ETF":
            exclusions["universe_or_type"] += 1
            continue
        established_fund = (
            row.get("type") == "ETF"
            and symbol in {"GLD", "IAU", "SLV", "PPLT", "PALL", "USO", "UNG", "DBC"}
            and "OTC" not in row.get("exchange", "").upper()
        )
        if not member_asof(body, symbol, str(dates[b])) or (
            body["source"] != "user_adjusted_csv" and not (exchange_traded_candidate(row) or established_fund)
        ):
            exclusions["universe_or_type"] += 1
            continue
        if (not np.isfinite(turnover[i]) and not config.allow_unknown_liquidity) or (
            np.isfinite(turnover[i]) and turnover[i] < config.minimum_turnover
        ):
            exclusions["liquidity"] += 1
            continue
        # Exact training-return aliases and supplied permanent security IDs cannot count twice.
        signature = np.nan_to_num(np.round(returns[:, i], 8), nan=999).tobytes()
        security = metadata.get("security_id") or row["id"]
        if security in keys or signature in signatures:
            exclusions["duplicate"] += 1
            continue
        keys.add(security)
        signatures.add(signature)
        candidates.append(i)
    selected = candidates[: config.max_assets]
    if len(selected) < 10:
        raise ValueError(
            f"Fold {fold['number']}: fewer than 10 training-eligible assets. Review coverage, membership and volume; CSV without volume needs the explicit unknown-liquidity assumption."
        )
    r = returns[:, selected]
    factors = []
    factor_indices = []
    wanted = [] if config.factor_mode == "raw" else (["SPY"] if config.factor_mode == "market" else FACTORS)
    for symbol in wanted:
        choices = [
            i
            for i, row in enumerate(body["records"])
            if row["symbol"] == symbol and coverage[i] >= 0.95 and vol[i] > 1e-8
        ]
        if choices:
            factor_indices.append(max(choices, key=lambda i: coverage[i]))
            factors.append(symbol)
    if wanted and "SPY" not in factors:
        raise ValueError("The selected factor mode requires SPY with adequate training coverage")
    residual, betas = factor_residuals(r, returns[:, factor_indices], config.ridge * 0.01)
    residual_std = np.nanstd(residual, axis=0)
    usable = np.isfinite(residual_std) & (residual_std > 1e-8)
    if usable.sum() < 2:
        raise ValueError("Fewer than two assets have measurable residual variation after factor adjustment")
    # A factor ETF explained almost exactly by itself is still usable for a long-only benchmark,
    # but gets a zero residual relationship rather than a fabricated correlation.
    standardized = np.nan_to_num((residual - np.nanmean(residual, axis=0)) / np.maximum(residual_std, 1e-8), nan=0)
    covariance = standardized.T @ standardized / max(1, len(standardized) - 1)
    denom = np.sqrt(np.maximum(np.diag(covariance), 1e-20))
    covariance = covariance / (denom[:, None] * denom[None, :])
    raw, _ = pairwise_tile(r, 0, len(selected), 104 if config.frequency == "weekly" else 252)
    loadings, fractions, precision = joint_factors(residual[:, usable], config.ridge)
    partial = np.zeros_like(covariance)
    locations = np.flatnonzero(usable)
    partial[np.ix_(locations, locations)] = partial_tile(precision, 0, len(locations))
    neighbors = strongest_neighbors(partial, 0, min(12, len(selected) - 1), absolute=True)
    groups = graph_groups(strongest_neighbors(raw, 0, min(12, len(selected) - 1)), 0.85)
    representatives = [
        max(group, key=lambda j: float(turnover[selected[j]]) if np.isfinite(turnover[selected[j]]) else 0)
        for group in groups
    ]
    representatives = sorted(
        representatives,
        key=lambda j: (
            not np.isfinite(turnover[selected[j]]),
            -float(turnover[selected[j]]) if np.isfinite(turnover[selected[j]]) else 0,
        ),
    )[: config.max_positions]
    daily = np.diff(logs[selected, a : b + 1], axis=1).T
    annual_vol = np.nanstd(daily, axis=0, ddof=1) * np.sqrt(252)
    weights = np.zeros(len(selected))
    inverse = 1 / np.maximum(annual_vol[representatives], 0.01)
    weights[representatives] = inverse / inverse.sum()
    cov = np.cov(np.nan_to_num(daily, nan=np.nanmean(daily, axis=0)), rowvar=False) * 252
    expected_vol = float(np.sqrt(max(0, weights @ cov @ weights)))
    weights *= min(1, config.volatility_target / max(expected_vol, 1e-8))
    factor_summary = []
    for k, fraction in enumerate(fractions):
        indices = np.argsort(-np.abs(loadings[:, k]))[:8]
        factor_summary.append(
            {
                "variance_fraction": float(fraction),
                "members": [
                    {"symbol": body["records"][selected[locations[j]]]["symbol"], "loading": float(loadings[j, k])}
                    for j in indices
                ],
            }
        )
    edges = []
    for i, row in enumerate(neighbors):
        for j, value in row:
            if i < j:
                edges.append(
                    {
                        "left": body["records"][selected[i]]["symbol"],
                        "right": body["records"][selected[j]]["symbol"],
                        "raw": float(raw[i, j]),
                        "residual": float(np.clip(covariance[i, j], -1, 1)),
                        "conditional": value,
                    }
                )
    edges.sort(key=lambda row: -abs(row["conditional"]))
    return {
        "indices": selected,
        "feature_indices": feature,
        "neighbors": neighbors,
        "weights": weights,
        "turnover": turnover[selected],
        "volatility": annual_vol,
        "attributes": [attrs[i] for i in selected],
        "exclusions": exclusions,
        "eligible_before_cap": len(candidates),
        "factor_symbols": factors,
        "factor_betas": betas,
        "relationships": edges[:100],
        "factors": factor_summary,
        "group_count": len(groups),
    }


def metrics(returns, cash_rate=0):
    r = np.asarray(returns, float)
    if not len(r) or not np.isfinite(r).all() or (r < -1).any():
        return {"status": "invalid", "return": None, "cagr": None, "sharpe": None, "max_drawdown": None}
    equity = np.cumprod(1 + r)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0
    draw = equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1
    tail = np.sort(r)[: max(1, int(np.ceil(0.05 * len(r))))]
    return {
        "status": "evaluated",
        "return": float(equity[-1] - 1),
        "cagr": float(equity[-1] ** (252 / len(r)) - 1),
        "sharpe": float((r.mean() - cash_rate / 252) / sd * np.sqrt(252)) if sd > 1e-12 else None,
        "volatility": sd * np.sqrt(252),
        "max_drawdown": float(draw.min()),
        "daily_var95": float(-np.quantile(r, 0.05)),
        "daily_cvar95": float(-tail.mean()),
        "sessions": len(r),
        "positive_day_fraction": float((r > 0).mean()),
    }


def uncertainty(returns, samples, block, seed):
    r = np.asarray(returns, float)
    if len(r) < 126 or not np.isfinite(r).all() or (r <= -1).any():
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        starts = rng.integers(0, len(r), int(np.ceil(len(r) / block)))
        indices = ((starts[:, None] + np.arange(block)) % len(r)).ravel()[: len(r)]
        values.append(float(np.expm1(np.log1p(r[indices]).mean() * 252)))
    return {
        "cagr_95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "samples": samples,
        "block_sessions": block,
        "note": "Conditional circular block-bootstrap uncertainty; does not correct strategy selection, survivorship or model error.",
    }
