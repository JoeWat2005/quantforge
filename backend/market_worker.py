"""Large, bounded, multi-asset research job; full-market or explicit cached preview."""

import csv
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from backend.discovery import effective_threads
from backend.market_analysis import (
    analyze_basket_batch,
    basket_backtest,
    calendar_positions,
    exchange_traded_candidate,
    generate_baskets,
    graph_groups,
    initialize_baskets,
    joint_factors,
    longest_complete,
    pairwise_tile,
    partial_tile,
    strongest_neighbors,
)
from backend.market_data import (
    BENCHMARK_SYMBOLS,
    MARKET,
    MarketConfig,
    atomic_json,
    collection_state,
    database,
    stock_scope,
)
from backend.worker import emit


def portfolio_experiment(logs, global_groups, training_liquidity, train_vol, rows, dates, split, cost_bps=10):
    """Predefined cluster-balanced inverse-volatility baseline, not an optimized buy list."""
    choices = []
    for group in global_groups:
        liquid = [
            i
            for i in group
            if training_liquidity[i] >= 1_000_000 and train_vol[i] > 1e-8 and exchange_traded_candidate(rows[i])
        ]
        if liquid:
            choices.append(max(liquid, key=lambda i: training_liquidity[i]))
    choices = sorted(choices, key=lambda i: -training_liquidity[i])[:32]
    if len(choices) < 5:
        return {
            "status": "insufficient_liquid_groups",
            "note": "Need five training-liquid exposure groups for this baseline",
        }
    target = 1 / np.maximum(train_vol[choices], 1e-8)
    target = target / target.sum()
    # Cap each holding, retaining any residual budget in cash.
    target = np.minimum(target, 0.1)
    prices = np.exp(np.asarray(logs[choices, :])).T
    first, last = longest_complete(prices[split:])
    first += split
    last += split
    if last - first < 126:
        return {
            "status": "insufficient_contiguous_holdout",
            "note": "Selected holdings do not have 126 complete consecutive held-out sessions",
        }
    units = np.zeros(len(choices))
    cash = equity = peak = 1.0
    drawdown = fees = 0.0
    rebalance_count = 0
    trace = []
    previous_month = None
    previous_equity = 1.0
    daily = []
    for i in range(first, last):
        equity = cash + float(units @ prices[i])
        month = str(dates[i])[:7]
        if month != previous_month and i < last - 1:
            values = units * prices[i]
            desired = equity * target
            fee = float(np.abs(desired - values).sum()) * cost_bps / 10000
            after = max(0.0, equity - fee)
            desired = after * target
            # Solve transaction-cost fixed point because fees reduce investable equity.
            for _ in range(4):
                fee = float(np.abs(desired - values).sum()) * cost_bps / 10000
                after = max(0.0, equity - fee)
                desired = after * target
            units = desired / prices[i]
            cash = after - float(desired.sum())
            equity = after
            fees += fee
            rebalance_count += 1
            previous_month = month
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
        daily.append(equity / previous_equity - 1 if previous_equity > 0 else 0.0)
        previous_equity = equity
        trace.append({"date": str(dates[i]), "equity": equity})
    ending_fee = float(np.abs(units * prices[last - 1]).sum()) * cost_bps / 10000
    equity = max(0.0, equity - ending_fee)
    fees += ending_fee
    daily[-1] = (1 + daily[-1]) * (equity / max(previous_equity, 1e-20)) - 1
    trace[-1]["equity"] = equity
    drawdown = min(drawdown, equity / peak - 1)
    take = np.unique(np.linspace(0, len(trace) - 1, min(120, len(trace)), dtype=int))
    sd = float(np.std(daily, ddof=1))
    benchmark = {"status": "unavailable"}
    candidates = [i for i, row in enumerate(rows) if row["symbol"] == "SPY"]
    if candidates:
        reference = np.exp(np.asarray(logs[candidates[0], first:last]))
        if np.isfinite(reference).all():
            equity_path = reference / reference[0] / (1 + cost_bps / 10000)
            equity_path[-1] *= 1 - cost_bps / 10000
            daily_reference = np.diff(np.r_[1.0, equity_path]) / np.r_[1.0, equity_path[:-1]]
            reference_sd = float(daily_reference.std(ddof=1))
            benchmark = {
                "status": "evaluated",
                "symbol": "SPY",
                "return": float(equity_path[-1] - 1),
                "max_drawdown": float(np.min(equity_path / np.maximum.accumulate(np.r_[1.0, equity_path])[1:] - 1)),
                "sharpe": float(daily_reference.mean() / reference_sd * np.sqrt(252)) if reference_sd > 1e-12 else None,
                "start": str(dates[first]),
                "end": str(dates[last - 1]),
                "note": "Matched observed dates, adjusted-price buy and hold, same entry/exit trading cost; no borrowing.",
            }
    return {
        "status": "evaluated",
        "method": "One training-liquid representative per training correlation component; inverse-volatility weights; 10% holding cap; monthly rebalance",
        "holdings": [
            {"symbol": rows[i]["symbol"], "exchange": rows[i]["exchange"], "weight": float(w)}
            for i, w in zip(choices, target, strict=True)
        ],
        "cash_weight": float(1 - target.sum()),
        "return": equity - 1,
        "max_drawdown": drawdown,
        "sharpe": float(np.mean(daily) / sd * np.sqrt(252)) if sd > 1e-12 else None,
        "fees_fraction_initial": fees,
        "rebalances": rebalance_count,
        "start": str(dates[first]),
        "end": str(dates[last - 1]),
        "trace": [trace[i] for i in take],
        "matched_benchmark": benchmark,
        "note": "Retrospective research baseline with a current-provider universe, not an unbiased investable performance claim. Cross-group correlations are not zero.",
    }


def run(config):
    started = time.perf_counter()

    def progress(percent, stage, **extra):
        elapsed = time.perf_counter() - started
        emit(
            type="progress",
            engine="market",
            completed=percent,
            paths=100,
            stage=stage,
            elapsed=elapsed,
            throughput=percent / max(elapsed, 0.001),
            results=None,
            **extra,
        )

    status = collection_state()
    snapshot = config.snapshot_id or status.get("snapshot_id")
    if not snapshot:
        raise ValueError("Start a Twelve Data collection before market analysis")
    db, directory = database(snapshot)
    instruments = [dict(row) for row in db.execute("SELECT * FROM instruments WHERE status='ok' ORDER BY symbol,id")]
    if stock_scope():
        instruments = [row for row in instruments if row["type"] != "ETF" or row["symbol"] in BENCHMARK_SYMBOLS]
    snapshot_total = db.execute("SELECT count(*) FROM instruments").fetchone()[0]
    db.close()
    if len(instruments) < 10:
        raise ValueError("At least ten downloaded histories are required for a multi-asset preview")
    scope = "configured_scope_attempted" if status.get("status") == "completed" else "cached_subset_preview"
    progress(1, f"Freezing {len(instruments):,} cached instruments · {scope}")
    samples = []
    date_pool = {}
    for i, row in enumerate(instruments):
        with np.load(directory / (row["id"] + ".npz"), allow_pickle=False) as data:
            dates = data["dates"]
            dates = date_pool.setdefault(hashlib.sha256(dates.tobytes()).digest(), dates)
            samples.append((dates, data["close"], data["volume"]))
        if i % 1000 == 0:
            progress(1 + 5 * (i + 1) / len(instruments), "Aligning historical prices without forward filling")
    anchors = [i for i, row in enumerate(instruments) if row["symbol"] == "SPY" and row["mic"] == "ARCX"]
    if not anchors:
        anchors = [i for i, row in enumerate(instruments) if row["symbol"] == "SPY"]
    if not anchors:
        raise ValueError("Download SPY first: its observed sessions provide the US alignment calendar")
    anchor = max(anchors, key=lambda i: instruments[i]["bars"])
    dates = np.array([day for day in samples[anchor][0] if datetime.fromisoformat(str(day)).weekday() < 5], dtype="U10")
    split = int(len(dates) * config.train_fraction)
    if split < config.minimum_overlap + 1 or len(dates) - split < 126:
        raise ValueError("Insufficient dates for the requested training and held-out windows")
    source_digest = hashlib.sha256(
        b"".join(
            Path(__file__).with_name(name).read_bytes()
            for name in ("market_worker.py", "market_analysis.py", "market_data.py")
        )
    ).hexdigest()
    fingerprint = hashlib.sha256(
        (
            "|".join(row["id"] for row in instruments)
            + "|"
            + str(config.model_dump())
            + "|"
            + str(dates[-1])
            + "|"
            + source_digest
        ).encode()
    ).hexdigest()[:24]
    output = MARKET / "analysis" / fingerprint
    output.mkdir(parents=True, exist_ok=True)
    required_disk = len(instruments) * len(dates) * 8 + len(instruments) ** 2 * 4
    if shutil.disk_usage(output).free < required_disk + 10 * 1024**3:
        raise ValueError("Insufficient free disk space for the matrix and 10 GiB disk reserve")
    logs = np.lib.format.open_memmap(
        output / "log_prices.npy", mode="w+", dtype="float64", shape=(len(instruments), len(dates))
    )
    logs[:] = np.nan
    liquidity = np.zeros(len(instruments))
    off_calendar_bars = 0
    for i, (ds, close, volume) in enumerate(samples):
        index, valid_dates = calendar_positions(dates, ds)
        off_calendar_bars += int((~valid_dates).sum())
        close, volume = close[valid_dates], volume[valid_dates]
        logs[i, index] = np.log(close)
        observed = index < split
        liquidity[i] = float(np.median(close[observed] * volume[observed])) if observed.any() else 0.0
    logs.flush()
    del samples, date_pool
    raw = np.diff(logs[:, :split], axis=1)
    counts = np.isfinite(raw).sum(axis=1)
    with np.errstate(invalid="ignore"):
        deviation = np.nanstd(raw, axis=1, ddof=1)
    active = (np.abs(np.nan_to_num(raw)) > 1e-10).sum(axis=1) / np.maximum(counts, 1)
    eligible = np.flatnonzero((counts >= config.minimum_overlap) & (deviation > 1e-8) & (active >= 0.1))
    if len(eligible) < 10:
        raise ValueError("Fewer than ten histories pass training-overlap, variance and active-return quality checks")
    training = np.asarray(raw[eligible, :].T)
    del raw
    n = len(eligible)
    tradable = np.array([exchange_traded_candidate(instruments[i]) and liquidity[i] >= 1_000_000 for i in eligible])
    core_local = np.flatnonzero((np.isfinite(training).mean(axis=0) >= 0.95) & tradable)
    if len(core_local) < 10:
        raise ValueError(
            "Fewer than ten exchange-traded, training-liquid instruments have 95% training coverage for the joint model"
        )
    core_global = eligible[core_local]
    threads = effective_threads(config)
    block = min(256, max(16, int(config.ram_gb * 1024**3 / (max(1, n) * 8 * 40))))
    matrix = np.memmap(output / "correlation.f32", dtype="float32", mode="w+", shape=(n, n))
    neighbors = []
    valid_pairs = 0
    with threadpool_limits(limits=threads):
        for start in range(0, n, block):
            end = min(n, start + block)
            corr, overlap = pairwise_tile(training, start, end, config.minimum_overlap)
            matrix[start:end] = corr
            for k, row in enumerate(corr):
                valid_pairs += int(np.isfinite(row[start + k + 1 :]).sum())
            neighbors.extend(strongest_neighbors(corr, start, config.neighbor_count))
            matrix.flush()
            progress(8 + 27 * end / n, f"Whole-universe pairwise-complete correlations · {end:,}/{n:,} assets")
        progress(36, f"Joint PCA and shrinkage precision · {len(core_local):,} core assets")
        loadings, fractions, precision = joint_factors(training[:, core_local], config.ridge)
        partial_neighbors = []
        for start in range(0, len(core_local), block):
            end = min(len(core_local), start + block)
            partial_neighbors.extend(
                strongest_neighbors(partial_tile(precision, start, end), start, config.neighbor_count, absolute=True)
            )
            progress(
                37 + 12 * end / len(core_local),
                f"Conditional relationships controlling for all {len(core_local):,} core assets",
            )
    del precision
    groups = graph_groups(neighbors)
    factors = []
    for j, ratio in enumerate(fractions):
        indices = np.argsort(-np.abs(loadings[:, j]))[:15]
        factors.append(
            {
                "factor": j + 1,
                "variance_fraction": float(ratio),
                "loadings": [
                    {
                        "symbol": instruments[core_global[i]]["symbol"],
                        "exchange": instruments[core_global[i]]["exchange"],
                        "loading": float(loadings[i, j]),
                    }
                    for i in indices
                ],
            }
        )
    edge_rows = []
    seen_edges = set()
    for a, row in enumerate(neighbors):
        for b, value in row:
            identity = tuple(sorted((a, b)))
            if identity not in seen_edges:
                seen_edges.add(identity)
                edge_rows.append(
                    {
                        "left": instruments[eligible[a]]["symbol"],
                        "right": instruments[eligible[b]]["symbol"],
                        "left_id": instruments[eligible[a]]["id"],
                        "right_id": instruments[eligible[b]]["id"],
                        "training_correlation": value,
                    }
                )
    edge_rows.sort(key=lambda row: -row["training_correlation"])
    conditional = []
    seen_edges = set()
    for a, row in enumerate(partial_neighbors):
        for b, value in row:
            identity = tuple(sorted((a, b)))
            if identity not in seen_edges:
                seen_edges.add(identity)
                ga, gb = core_global[a], core_global[b]
                conditional.append(
                    {
                        "left": instruments[ga]["symbol"],
                        "right": instruments[gb]["symbol"],
                        "partial_correlation": value,
                        "marginal_correlation": float(matrix[core_local[a], core_local[b]]),
                    }
                )
    conditional.sort(key=lambda row: -abs(row["partial_correlation"]))
    baskets = generate_baskets(
        core_global.tolist(), partial_neighbors, config.basket_sizes, config.basket_trials, config.seed
    )
    progress(50, f"Testing {len(baskets):,} multivariate baskets of {', '.join(map(str, config.basket_sizes))} assets")
    models = []
    # Windows ProcessPoolExecutor reserves handles and permits at most 61 workers.
    workers = min(61 if os.name == "nt" else 64, threads, max(1, int(config.ram_gb / 0.5)))
    batches = [baskets[i : i + 32] for i in range(0, len(baskets), 32)]
    iterator = iter(batches)
    done_count = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_baskets,
        initargs=(str(output / "log_prices.npy"), split, config.minimum_overlap),
    ) as pool:
        pending = set()
        for _ in range(workers * 2):
            batch = next(iterator, None)
            if batch is not None:
                pending.add(pool.submit(analyze_basket_batch, batch))
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                result = future.result()
                models.extend(result)
                done_count += len(result)
                batch = next(iterator, None)
                if batch is not None:
                    pending.add(pool.submit(analyze_basket_batch, batch))
            progress(
                50 + 40 * done_count / max(1, len(baskets)),
                f"Johansen multi-asset and fixed-vector held-out tests · {done_count:,}/{len(baskets):,} baskets",
            )
    valid = [model for model in models if model["status"] == "ok"]
    for model in valid:
        model["material_basket"] = (
            model["effective_legs"] == len(model["indices"])
            and model["mixed_sign"]
            and 0 < model["nominal_rank"] < len(model["indices"])
        )
    # Prevent a near-zero third leg from disguising an ordinary pair as a basket.
    valid.sort(key=lambda model: (not model["material_basket"], -model["training_score"]))
    for rank, model in enumerate(valid):
        model["symbols"] = [instruments[i]["symbol"] for i in model["indices"]]
        p = model["heldout_adf_p"]
        model["adjusted_heldout_adf_p"] = min(1.0, p * len(baskets)) if p is not None else None
        if (
            rank < 100
            and model["material_basket"]
            and model["mixed_sign"]
            and 0 < model["nominal_rank"] < len(model["indices"])
            and model["heldout_rows"] >= 126
        ):
            history = np.asarray(logs[model["indices"], :]).T
            test = basket_backtest(
                history,
                np.array(model["weights"]),
                model["training_mean"],
                model["training_std"],
                model["heldout_start"],
                model["heldout_end"],
                cost_bps=config.cost_bps,
                borrow=config.borrow_rate,
            )
            for point in test["trace"]:
                point["date"] = str(dates[point.pop("index")])
            model["strategy_test"] = test
        for key in ("training_start", "training_end", "heldout_start", "heldout_end"):
            index = model[key]
            model[key] = str(dates[index]) if 0 <= index < len(dates) else None
    progress(92, "Evaluating a predefined cluster-diversified research baseline after costs")
    global_groups = [[int(eligible[i]) for i in group] for group in groups]
    strategy = portfolio_experiment(
        logs, global_groups, liquidity, deviation, instruments, dates, split, config.cost_bps
    )
    summaries = []
    for group in global_groups[:60]:
        leaders = sorted(group, key=lambda i: -liquidity[i])[:25]
        summaries.append(
            {
                "size": len(group),
                "members": [
                    {"symbol": instruments[i]["symbol"], "exchange": instruments[i]["exchange"]} for i in leaders
                ],
            }
        )

    def write_csv(name, data, columns):
        path = output / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in data:
                clean = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                writer.writerow(clean)
        return name

    edge_csv = write_csv(
        "relationships.csv", edge_rows, ["left", "right", "left_id", "right_id", "training_correlation"]
    )
    basket_csv = write_csv(
        "baskets.csv",
        valid,
        [
            "symbols",
            "weights",
            "effective_legs",
            "material_basket",
            "training_score",
            "nominal_rank",
            "training_rows",
            "heldout_rows",
            "heldout_adf_p",
            "adjusted_heldout_adf_p",
            "training_start",
            "training_end",
            "heldout_start",
            "heldout_end",
        ],
    )
    metadata = {
        "scope": scope,
        "snapshot_id": snapshot,
        "analysis_id": fingerprint,
        "algorithm_sha256": source_digest,
        "session_calendar": "Observed SPY weekday sessions (provider reference, not an independent exchange calendar)",
        "off_calendar_bars_excluded": off_calendar_bars,
        "provider_catalogue_instruments": snapshot_total,
        "downloaded_at_start": len(instruments),
        "analysis_eligible": n,
        "joint_model_core": len(core_local),
        "excluded_training_quality": len(instruments) - n,
        "core_coverage_minimum": 0.95,
        "all_possible_pairs": n * (n - 1) // 2,
        "valid_pairwise_correlations": valid_pairs,
        "start": str(dates[0]),
        "train_end": str(dates[split - 1]),
        "holdout_start": str(dates[split]),
        "end": str(dates[-1]),
        "basket_candidates": len(baskets),
        "basket_successful_fits": len(valid),
        "material_baskets": sum(model["material_basket"] for model in valid),
        "basket_search_space": {
            str(k): str(math.comb(len(core_local), k)) for k in config.basket_sizes if len(core_local) >= k
        },
        "threads_requested": threads,
        "basket_processes": workers,
        "matrix_path": str(output / "correlation.f32"),
        "relationship_csv": edge_csv,
        "basket_csv": basket_csv,
    }
    result = {
        "metadata": metadata,
        "config": config.model_dump(),
        "factors": factors,
        "groups": summaries,
        "group_count": len(groups),
        "relationships": edge_rows[:300],
        "conditional_relationships": conditional[:300],
        "baskets": valid[:200],
        "strategy_baseline": strategy,
        "elapsed": time.perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "All eligible instruments enter the marginal correlation matrix. Basket combinations are sampled, not exhaustively enumerated.",
            "Only earlier training data select relationships, factors, baskets and coefficients. Later data are evaluation only. Human retuning against later results invalidates an untouched-test interpretation.",
            "Raw correlations use pairwise-complete returns with minimum overlap. The joint Gaussian ridge model uses a 95%-coverage core with missing training returns set to training means; partial correlation is conditional association, not causation.",
            "PCA factors describe common return variation and are not forecasts. Network groups are transitive connected components, so every pair within a group need not be strongly correlated.",
            "Johansen ranks use unadjusted 5% critical values and assume I(1) log prices. Fixed-vector held-out ADF p-values have a Bonferroni adjustment across every sampled basket. These are exploratory asymptotic diagnostics, not proof of profitable trading.",
            "Baskets with both coefficient signs, a nominal rank between zero and full rank, and every absolute normalized coefficient at least 2% are shown first. Within this group, ranking uses only the training trace score. This stops negligible extra legs from presenting a pair as a genuine many-asset relationship; all successful fits remain in CSV.",
            "Basket tests use longest contiguous complete training and evaluation segments; actual dates are recorded. No prices are forward filled for correlation or inference.",
            "Histories align to observed SPY weekday sessions. Off-calendar bars are excluded and counted; this reference depends on provider SPY coverage.",
            "The provider ETF endpoint also contains mutual fund share classes. All downloaded instruments can enter raw correlations, but joint factors, baskets and portfolio representatives require a listed common stock/ADR/REIT/ETN or an explicitly ETF-named fund, plus $1 million median training daily price-times-volume. This conservative name/type filter can exclude genuine ETFs and does not verify borrowability or execution liquidity.",
            "Strategy results are retrospective research with a present-day provider universe. Delisted coverage, historical constituents, liquidity, funding, execution constraints and survivorship bias limit investability.",
            "No orders are placed. A new forward paper-trading interval is required before treating a research hypothesis as validated.",
        ],
    }
    atomic_json(output / "report.json", result)
    atomic_json(MARKET / "latest.json", result)
    atomic_json(
        output / "instruments.json",
        [{"id": instruments[i]["id"], "symbol": instruments[i]["symbol"]} for i in eligible],
    )
    progress(100, "Multi-asset research complete")
    emit(type="market_complete", market=result)
    emit(type="complete")


if __name__ == "__main__":
    try:
        emit(type="ready", pid=os.getpid())
        run(MarketConfig.model_validate_json(input()))
    except Exception as exc:
        emit(type="error", error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
