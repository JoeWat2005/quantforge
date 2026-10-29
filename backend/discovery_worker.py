"""Isolated research worker. Uses the same external resource guards as Monte Carlo."""

import csv
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

from backend.discovery import (
    analyse_pair,
    bootstrap_correlations,
    effective_threads,
    exposure_clusters,
    finite_correlation,
    init_pair_worker,
    pair_backtest,
    rolling_correlations,
    screen_pairs,
)
from backend.discovery_data import DATA, DiscoveryConfig, load_prices
from backend.worker import emit


def run(config):
    started = time.perf_counter()

    def progress(percent, stage):
        elapsed = time.perf_counter() - started
        emit(
            type="progress",
            engine="discovery",
            completed=percent,
            paths=100,
            stage=stage,
            elapsed=elapsed,
            throughput=percent / max(elapsed, 0.001),
            results=None,
        )

    frame, metadata = load_prices(config, progress)
    log_prices = np.log(frame.to_numpy(dtype=float))
    names = frame.columns.tolist()
    split = int(len(frame) * config.train_fraction)
    if split < 252 or len(frame) - split < 100:
        raise ValueError("Need 252 training observations and 100 held-out observations")
    threads = effective_threads(config)
    progress(22, "Computing all-pairs training return correlations")
    with threadpool_limits(limits=threads):
        corr, pairs, returns = screen_pairs(log_prices, split, config.shortlist)
    total_pairs = len(names) * (len(names) - 1) // 2
    metadata.update(
        train_end=str(frame.index[split - 1].date()),
        holdout_start=str(frame.index[split].date()),
        train_rows=split,
        holdout_rows=len(frame) - split,
        all_pairs=total_pairs,
        tested_pairs=len(pairs),
        matrix_sha256=hashlib.sha256(log_prices.tobytes()).hexdigest(),
    )
    results = []
    for start in range(0, len(pairs), 32):
        group = pairs[start : start + 32]
        bounds = bootstrap_correlations(
            returns, group, config.bootstrap_repetitions, config.block_days, (config.seed + start) % (2**64), threads
        )
        for pair, bound in zip(group, bounds, strict=True):
            a, b = map(int, pair)
            rolling = rolling_correlations(returns[:, a], returns[:, b], config.rolling_days)
            results.append(
                {
                    "a": a,
                    "b": b,
                    "left": names[a],
                    "right": names[b],
                    "training_correlation": float(corr[a, b]),
                    "bootstrap_95": bound.tolist(),
                    "rolling_min": min(rolling) if rolling else None,
                    "rolling_max": max(rolling) if rolling else None,
                }
            )
        progress(
            23 + 42 * min(start + 32, len(pairs)) / len(pairs),
            f"Native block-bootstrap · {min(start + 32, len(pairs))}/{len(pairs)} pairs",
        )
    directory = DATA / "research"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.npy"):
        try:
            stale.unlink()
        except PermissionError:
            pass
    train_path = directory / (uuid.uuid4().hex + ".npy")
    np.save(train_path, log_prices[:split])
    workers = min(threads, 16, max(1, int(config.ram_gb / 0.5)), len(pairs))
    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=init_pair_worker, initargs=(str(train_path),)
        ) as pool:
            tasks = {pool.submit(analyse_pair, tuple(map(int, pair))): i for i, pair in enumerate(pairs)}
            for done, future in enumerate(as_completed(tasks), 1):
                i = tasks[future]
                test = future.result()
                results[i].update(test)
                p = test["cointegration_p"]
                results[i]["adjusted_p"] = min(1.0, p * total_pairs) if p is not None else None
                results[i].update(pair_backtest(log_prices, test, split, config.cost_bps, config.borrow_rate))
                if i >= 250:
                    results[i]["trace"] = []
                elif i >= 30:
                    full_trace = results[i]["trace"]
                    coarse = full_trace[::4]
                    if full_trace and coarse[-1] is not full_trace[-1]:
                        coarse.append(full_trace[-1])
                    results[i]["trace"] = coarse
                a, b = test["a"], test["b"]
                held = np.diff(log_prices[split - 1 :, [a, b]], axis=0)
                results[i]["holdout_correlation"] = finite_correlation(held[:, 0], held[:, 1])
                for point in results[i]["trace"]:
                    point["date"] = str(frame.index[point.pop("index")].date())
                progress(65 + 30 * done / len(pairs), f"Cointegration and held-out pair tests · {done}/{len(pairs)}")
    finally:
        train_path.unlink(missing_ok=True)
    # A separate training-only ranking for potential diversifiers of an anchor.
    anchor = names.index(config.anchor) if config.anchor in names else 0
    all_returns = np.diff(log_prices, axis=0)
    stress = returns[:, anchor] <= np.quantile(returns[:, anchor], 0.1)
    diversifiers = []
    for i in range(len(names)):
        if i == anchor:
            continue
        held = finite_correlation(all_returns[split - 1 :, anchor], all_returns[split - 1 :, i])
        stress_corr = finite_correlation(returns[stress, anchor], returns[stress, i])
        diversifiers.append(
            {
                "symbol": names[i],
                "training_correlation": float(corr[anchor, i]),
                "holdout_correlation": held,
                "stress_correlation": stress_corr,
                "training_annual_volatility": float(returns[:, i].std(ddof=1) * np.sqrt(252)),
            }
        )
    diversifiers.sort(key=lambda x: x["training_correlation"])
    cluster_labels = exposure_clusters(corr, names)
    report_id = train_path.stem
    csv_path = directory / (report_id + "_pairs.csv")
    columns = [
        "left",
        "right",
        "training_correlation",
        "rolling_min",
        "rolling_max",
        "holdout_correlation",
        "cointegration_p",
        "adjusted_p",
        "beta",
        "half_life_days",
        "holdout_return",
        "holdout_max_drawdown",
        "holdout_sharpe",
        "entries",
        "backtest_enabled",
        "test_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns + ["bootstrap_lower", "bootstrap_upper"], extrasaction="ignore"
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {**row, "bootstrap_lower": row["bootstrap_95"][0], "bootstrap_upper": row["bootstrap_95"][1]}
            )
    metadata["csv_file"] = csv_path.name
    result = {
        "metadata": metadata,
        "config": config.model_dump(),
        "pairs": results[:250],
        "diversifiers": diversifiers,
        "anchor": names[anchor],
        "clusters": cluster_labels,
        "heatmap_names": names[:24],
        "heatmap": corr[:24, :24].tolist(),
        "elapsed": time.perf_counter() - started,
        "native_bootstrap_resamples": len(pairs) * config.bootstrap_repetitions,
        "bootstrap_return_observations": len(pairs) * config.bootstrap_repetitions * len(returns),
        "threads": threads,
        "cointegration_workers": workers,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Research candidates, not buy/sell instructions. Correlation does not forecast returns.",
            "Shortlist, hedge ratios, thresholds and rankings use training data only. Later observations are evaluation only; repeated tuning against them invalidates that holdout.",
            "Adjusted p-values use Bonferroni over every possible universe pair, not just the shortlist. Engle-Granger assumes I(1) log price series; this assumption is not established by the screen.",
            "Pairs test uses fixed training coefficients, entry |z|>=2, exit |z|<0.5, stop |z|>4, 20-session holding limit, next-interval exposure, transaction costs and constant short borrow fees.",
            "Shorting availability, market impact, funding, taxes and historical index membership are not modeled. Statistical relationships can break.",
            "Confidence intervals are paired circular moving-block bootstrap estimates, conditional on this training sample and block length. These are descriptive pointwise intervals, not selection-adjusted or simultaneous-coverage guarantees.",
        ],
    }
    progress(99, "Saving reproducible research report")
    path = directory / "latest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    reports = sorted(directory.glob("*_pairs.csv"), key=lambda p: p.stat().st_mtime)
    for old in reports[:-20]:
        old.unlink()
    progress(100, "Research complete")
    emit(type="research_complete", discovery=result)
    emit(type="complete")


if __name__ == "__main__":
    try:
        emit(type="ready", pid=os.getpid())
        run(DiscoveryConfig.model_validate_json(input()))
    except Exception as exc:
        emit(type="error", error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
