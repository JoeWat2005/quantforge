"""Bounded walk-forward experiments, validation-only selection and fold checkpoints."""

import csv
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from backend import lab_store
from backend.discovery import effective_threads
from backend.lab_analysis import half_life, metrics, schedule, stationarity_p, training_model, uncertainty
from backend.lab_data import load_dataset
from backend.lab_models import ResearchConfig
from backend.lab_portfolio import benchmark, project_weights, run_portfolio
from backend.market_analysis import analyze_basket_batch, generate_baskets, initialize_baskets, longest_complete
from backend.market_data import atomic_json
from backend.worker import emit


def source_hash():
    return hashlib.sha256(
        b"".join(
            Path(__file__).with_name(name).read_bytes()
            for name in (
                "lab_worker.py",
                "lab_analysis.py",
                "lab_portfolio.py",
                "lab_models.py",
                "lab_data.py",
                "market_analysis.py",
                "warehouse.py",
            )
        )
    ).hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text("No records\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
            )


def basket_worker_count(config, candidate_count):
    # Windows ProcessPoolExecutor supports at most 61 processes. Each child uses
    # one BLAS thread; the parent watchdog still checks actual process-tree RSS.
    return min(
        61 if os.name == "nt" else 128,
        effective_threads(config),
        max(1, int(config.ram_gb / 0.4)),
        max(1, (candidate_count + 31) // 32),
    )


def fit_baskets(path, train, neighbors, config, progress):
    candidates = generate_baskets(
        list(range(len(train))), neighbors, config.basket_sizes, config.basket_trials, config.seed
    )
    np.save(path, train)
    workers = basket_worker_count(config, len(candidates))
    minimum = 104 if config.frequency == "weekly" else 252
    batches = iter([candidates[i : i + 32] for i in range(0, len(candidates), 32)])
    models = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_baskets, initargs=(str(path), train.shape[1], minimum)
    ) as pool:
        pending = set()
        for _ in range(workers * 2):
            batch = next(batches, None)
            if batch is not None:
                pending.add(pool.submit(analyze_basket_batch, batch))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                results = future.result()
                models.extend(results)
                done += len(results)
                batch = next(batches, None)
                if batch is not None:
                    pending.add(pool.submit(analyze_basket_batch, batch))
            progress(done / max(1, len(candidates)))
    return models, len(candidates), workers


def validation_score(result):
    if result["status"] != "evaluated":
        return -float("inf")
    m = result["metrics"]
    return m["return"] - 0.5 * abs(m["max_drawdown"])


def compact(result):
    # Daily returns and full ledger remain in checkpoints/CSV, bounded views in API reports.
    body = {key: value for key, value in result.items() if key not in ("trades", "daily_returns", "trace")}
    trace = result["trace"]
    take = np.unique(np.linspace(0, len(trace) - 1, min(180, len(trace)), dtype=int)) if trace else []
    body["trace"] = [trace[i] for i in take]
    body["trades"] = result["trades"][:300]
    body["trade_count"] = len(result["trades"])
    return body


def evaluate_fold(body, dates, logs, volumes, fold, config, directory, progress, capital_scales=None):
    with threadpool_limits(limits=effective_threads(config)):
        model = training_model(body, dates, logs, volumes, fold, config)
    ids = model["indices"]
    rows = [body["records"][i] for i in ids]
    panel = np.asarray(logs[ids, :])
    feature = model["feature_indices"]
    train = np.asarray(panel[:, feature])
    progress(
        0.12, f"Fold {fold['number']}: fitted training universe and {len(model['factor_symbols'])} observed ETF factors"
    )
    fitted, attempted, workers = fit_baskets(
        directory / f"training_{fold['number']}.npy",
        train,
        model["neighbors"],
        config,
        lambda p: progress(0.12 + 0.48 * p, f"Fold {fold['number']}: training-only multivariate tests"),
    )
    price_p = [stationarity_p(values) for values in train]
    usable = []
    for basket in fitted:
        if basket["status"] != "ok":
            continue
        basket["symbols"] = [rows[j]["symbol"] for j in basket["indices"]]
        a, b = longest_complete(train[basket["indices"]].T)
        spread = train[basket["indices"], a:b].T @ np.array(basket["weights"])
        basket["half_life_sessions"] = (
            half_life(spread) * (5 if config.frequency == "weekly" else 1) if half_life(spread) is not None else None
        )
        attrs = [model["attributes"][j] for j in basket["indices"]]
        borrow_ok = all(
            a.get("borrow_available", "").lower() != "false"
            and (config.allow_unverified_borrow or a.get("borrow_available", "").lower() == "true")
            for a in attrs
        )
        basket["material_basket"] = (
            basket["effective_legs"] == len(basket["indices"])
            and basket["mixed_sign"]
            and 0 < basket["nominal_rank"] < len(basket["indices"])
        )
        basket["training_price_stationarity_flag"] = any(
            price_p[j] is not None and price_p[j] < 0.01 for j in basket["indices"]
        )
        basket["candidate_for_trading"] = bool(
            basket["material_basket"]
            and borrow_ok
            and not basket["training_price_stationarity_flag"]
            and basket["half_life_sessions"] is not None
            and 0.5 <= basket["half_life_sessions"] <= 60
        )
        usable.append(basket)
    usable.sort(key=lambda row: (not row["candidate_for_trading"], -row["training_score"]))
    write_csv(directory / f"candidates_fold_{fold['number']}.csv", usable)
    chosen = []
    used = set()
    for basket in usable:
        if basket["candidate_for_trading"] and not used.intersection(basket["indices"]):
            chosen.append(basket)
            used.update(basket["indices"])
            if len(chosen) >= config.max_baskets:
                break
    va, vb = fold["validation"]
    ta, tb = fold["test"]
    options = []

    def simulate(start, end, book=None, **kwargs):
        effective = config
        if book and capital_scales:
            effective = config.model_copy(update={"capital": max(1, config.capital * capital_scales.get(book, 1))})
        return run_portfolio(panel, dates, rows, model, start, end, effective, dated_metadata=body, **kwargs)

    if config.strategy != "baskets":
        value = simulate(va, vb, capture=False)
        options.append({"family": "diversified", "entry": 2, "holding": 20, "result": value})
    if config.strategy != "diversified" and chosen:
        for entry in config.entry_grid:
            for holding in config.holding_grid:
                value = simulate(va, vb, family="baskets", baskets=chosen, entry=entry, holding=holding, capture=False)
                if value["executions"] >= 2:
                    options.append({"family": "baskets", "entry": entry, "holding": holding, "result": value})
    valid = [o for o in options if o["result"]["status"] == "evaluated"]
    selected = (
        max(valid, key=lambda o: validation_score(o["result"]))
        if valid
        else {"family": "cash", "entry": 2, "holding": 20, "result": None}
    )
    rule = {key: selected[key] for key in ("family", "entry", "holding")}
    progress(0.73, f"Fold {fold['number']}: validation froze {rule['family']} before opening test prices")
    kwargs = {"family": rule["family"], "baskets": chosen, "entry": rule["entry"], "holding": rule["holding"]}
    if rule["family"] == "cash":
        kwargs["fixed_weights"] = np.zeros(len(ids))
    selected_result = simulate(ta, tb, book="selected", **kwargs)
    stress = simulate(ta, tb, book="double_cost", multiplier=2, **kwargs)
    diversified = simulate(ta, tb, book="diversified")
    equal = np.zeros(len(ids))
    equal[: config.max_positions] = 1 / min(config.max_positions, len(ids))
    equal_weight = simulate(ta, tb, book="equal_weight", fixed_weights=equal)
    # Held-out diagnostics are attached after the rule is frozen and all test simulations finish.
    for basket in usable[:30]:
        values = panel[basket["indices"], ta : tb + 1].T
        a, b = longest_complete(values)
        basket["test_adf_p"] = stationarity_p(values[a:b] @ np.array(basket["weights"])) if b - a >= 126 else None
    public = {key: value for key, value in fold.items() if key not in ("train", "validation", "test")}
    public.update(
        assets=len(ids),
        eligible_before_cap=model["eligible_before_cap"],
        exclusions=model["exclusions"],
        factor_symbols=model["factor_symbols"],
        relationships=model["relationships"],
        factors=model["factors"],
        group_count=model["group_count"],
        basket_attempts=attempted,
        basket_fits=len(usable),
        basket_workers=workers,
        selected_rule=rule,
        validation_options=[
            {
                "family": o["family"],
                "entry": o["entry"],
                "holding": o["holding"],
                "score": validation_score(o["result"]) if o["result"]["status"] == "evaluated" else None,
                "metrics": o["result"]["metrics"],
                "status": o["result"]["status"],
            }
            for o in options
        ],
        fallback_cash=not bool(valid),
        baskets=usable[:30],
        holdings=[
            {
                "symbol": rows[i]["symbol"],
                "weight": float(w),
                "security_id": model["attributes"][i].get("security_id"),
                "sector": model["attributes"][i].get("sector"),
            }
            for i, w in enumerate(project_weights(model["weights"], config, model["attributes"]))
            if w > 0
        ],
        results={
            name: compact(result)
            for name, result in [
                ("selected", selected_result),
                ("double_cost", stress),
                ("diversified", diversified),
                ("equal_weight", equal_weight),
            ]
        },
    )
    books = {
        "selected": selected_result,
        "double_cost": stress,
        "diversified": diversified,
        "equal_weight": equal_weight,
    }
    progress(1, f"Fold {fold['number']} complete; test results were not used for its selection")
    return {"fold": public, "books": books, "test": fold["test"]}


def combine(checkpoints, dates, config):
    start = checkpoints[0]["test"][0]
    end = checkpoints[-1]["test"][1]
    output = {}
    ledger = []
    for name in ("selected", "double_cost", "diversified", "equal_weight"):
        returns = np.full(end - start + 1, config.cash_rate / 252)
        errors = []
        fees = borrow = turnover = 0
        for checkpoint in checkpoints:
            a, b = checkpoint["test"]
            value = checkpoint["books"][name]
            r = value["daily_returns"]
            returns[a - start : a - start + len(r)] = r
            if value["status"] != "evaluated":
                returns[a - start + len(r) : b - start + 1] = np.nan
                errors.append(value["reason"])
            fees += value["fees_fraction"]
            borrow += value["borrow_fraction"]
            turnover += value["turnover"]
            if name == "selected":
                ledger.extend([{**trade, "fold": checkpoint["fold"]["number"]} for trade in value["trades"]])
        equity = np.cumprod(1 + returns)
        take = np.unique(np.linspace(0, len(returns) - 1, min(500, len(returns)), dtype=int))
        output[name] = {
            "metrics": metrics(returns, config.cash_rate),
            "errors": errors,
            "fees_sum_fold_equity": fees,
            "borrow_sum_fold_equity": borrow,
            "turnover_sum_fold_equity": turnover,
            "trace": [
                {"date": str(dates[start + i]), "equity": float(equity[i])} for i in take if np.isfinite(equity[i])
            ],
            "daily_returns": [float(r) if np.isfinite(r) else None for r in returns],
        }
    return output, ledger, start, end


def run(config):
    started = time.perf_counter()
    directory = lab_store.report_path(config.experiment_id).parent
    directory.mkdir(parents=True, exist_ok=True)

    def progress(percent, stage):
        elapsed = time.perf_counter() - started
        emit(
            type="progress",
            engine="research",
            completed=percent,
            paths=100,
            stage=stage,
            elapsed=elapsed,
            throughput=percent / max(elapsed, 0.001),
            results=None,
        )

    progress(1, "Verifying frozen input hashes and observed prices")
    body, dates, logs, volumes = load_dataset(config.dataset_id)
    estimated = logs.nbytes + volumes.nbytes
    if estimated > config.ram_gb * 1024**3 * 0.6:
        raise ValueError("Dataset arrays exceed 60% of the RAM budget; raise the ceiling or import a smaller universe")
    if shutil.disk_usage(directory).free < 10 * 1024**3 + estimated * 2:
        raise ValueError("Insufficient disk space for experiment files plus 10 GiB reserve")
    folds = schedule(dates, config)
    digest = source_hash()
    checkpoint_file = directory / "checkpoint.json"
    key = hashlib.sha256((config.model_dump_json() + body["input_sha256"] + digest).encode()).hexdigest()
    checkpoints = []
    if checkpoint_file.exists():
        saved = json.loads(checkpoint_file.read_text())
        if saved.get("key") == key:
            checkpoints = saved["completed_folds"]
    resumed = len(checkpoints)
    for i, fold in enumerate(folds):
        if i < resumed:
            continue
        capital_scales = {name: 1.0 for name in ("selected", "double_cost", "diversified", "equal_weight")}
        previous_end = folds[0]["test"][0] - 1
        for previous in checkpoints:
            gap = previous["test"][0] - previous_end - 1
            for name in capital_scales:
                value = previous["books"][name]["metrics"].get("return")
                capital_scales[name] *= (1 + config.cash_rate / 252) ** max(0, gap) * (
                    1 + value if value is not None else 1
                )
            previous_end = previous["test"][1]
        gap = fold["test"][0] - previous_end - 1
        for name in capital_scales:
            capital_scales[name] *= (1 + config.cash_rate / 252) ** max(0, gap)
        checkpoint = evaluate_fold(
            body,
            dates,
            logs,
            volumes,
            fold,
            config,
            directory,
            lambda fraction, stage: progress(3 + 88 * (i + fraction) / len(folds), stage),
            capital_scales,
        )
        checkpoints.append(checkpoint)
        atomic_json(checkpoint_file, {"key": key, "completed_folds": checkpoints, "updated": lab_store.now()})
    progress(93, "Combining non-overlapping tests, cash gaps, stress costs and block uncertainty")
    aggregate, ledger, start, end = combine(checkpoints, dates, config)
    spy = [i for i, row in enumerate(body["records"]) if row["symbol"] == "SPY"]
    if not spy:
        raise ValueError("SPY benchmark is missing")
    bench = benchmark(logs[spy[0]], dates, start, end, config)
    if bench["status"] != "evaluated":
        bench["daily_returns"] = [None] * (end - start + 1)
    aggregate["spy"] = {**bench, "trace": bench["trace"][:: max(1, len(bench["trace"]) // 500)]}
    total_trials = sum(c["fold"]["basket_attempts"] for c in checkpoints)
    for checkpoint in checkpoints:
        for basket in checkpoint["fold"]["baskets"]:
            p = basket.get("test_adf_p")
            basket["adjusted_test_adf_p"] = min(1, p * total_trials) if p is not None else None
    selected_returns = aggregate["selected"]["daily_returns"]
    interval = (
        uncertainty(selected_returns, config.bootstrap_samples, config.block_sessions, config.seed)
        if None not in selected_returns
        else None
    )
    # Latest fit is for a new future paper interval, never retroactively added to test performance.
    end_index = int(np.searchsorted(dates, str(config.end_date or dates[-1]), side="right") - 1)
    import pandas as pd

    begin = int(
        np.searchsorted(dates, str((pd.Timestamp(dates[end_index]) - pd.DateOffset(years=config.train_years)).date()))
    )
    with threadpool_limits(limits=effective_threads(config)):
        deployed = training_model(
            body, dates, logs, volumes, {"train": [begin, end_index], "number": "forward"}, config
        )
    allocations = project_weights(deployed["weights"], config, deployed["attributes"])
    deployment = {
        "asof": str(dates[end_index]),
        "method": "Frozen long-only training-group inverse-volatility portfolio; monthly paper rebalance",
        "cost_inputs": {
            "attributes": [attr for attr, w in zip(deployed["attributes"], allocations) if w > 0],
            "turnover": [
                float(v) if np.isfinite(v) else None for v, w in zip(deployed["turnover"], allocations) if w > 0
            ],
            "volatility": [float(v) for v, w in zip(deployed["volatility"], allocations) if w > 0],
        },
        "holdings": [
            {
                "symbol": body["records"][i]["symbol"],
                "weight": float(weight),
                "id": body["records"][i]["id"],
                "exchange": body["records"][i].get("exchange", ""),
                "mic": body["records"][i].get("mic", ""),
            }
            for i, weight in zip(deployed["indices"], allocations)
            if weight > 0
        ],
    }
    result = {
        "version": 2,
        "experiment_id": config.experiment_id,
        "created": lab_store.now(),
        "config": config.model_dump(mode="json"),
        "dataset": {
            k: v for k, v in body.items() if k not in ("records", "dates", "security", "membership", "parquet_files")
        },
        "input_sha256": body["input_sha256"],
        "algorithm_sha256": digest,
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "threadpoolctl")},
        "elapsed": time.perf_counter() - started,
        "resumed_folds": resumed,
        "folds": [c["fold"] for c in checkpoints],
        "aggregate": aggregate,
        "uncertainty": interval,
        "deployment": deployment,
        "trade_count": len(ledger),
        "trades": ledger[:500],
        "test_start": str(dates[start]),
        "test_end": str(dates[end]),
        "basket_attempts": total_trials,
        "rule_evaluations": sum(len(c["fold"]["validation_options"]) for c in checkpoints),
        "notes": [
            "Every fold rebuilds its universe, factor model, groups and basket coefficients using training data only. A declared validation rule grid selects the family and settings; test outcomes never select that fold.",
            "Weekly mode changes common observation dates for relationship fitting; execution and costs still use daily reference closes. Weekly sampling supplies fewer observations, not automatically stronger evidence.",
            "Return correlations avoid price-level common trends. ETF residual controls are observed risk proxies, not a causal model or a complete removal of inflation. CPI-adjusted performance and fundamental factors are not currently modeled.",
            "Raw correlations use pairwise-complete training returns. Joint covariance/PCA/precision models standardize observed training returns and fill their small remaining gaps with training means, never future values or filled prices. Known liquid gold/commodity trusts are permitted alongside explicitly ETF-named funds; fund type and historical tradability still require review.",
            "The current provider catalogue has survivor, alias and membership limitations. Supplied historical membership/security metadata are used only when known by the fit date, but their provenance is not independently certified.",
            "Boundary gaps cover the longest declared holding period. Windows close positions at their predeclared end; inter-test gaps hold cash and are included in aggregate returns. Overlapping test windows are rejected.",
            "Signals observed at a close execute at the next close. Adjusted closes are an execution approximation; OHLC, order books, actual fills, taxes and corporate-action cash flows are not reconstructed.",
            "Commission, half-spread, slippage, training-volume impact, cash/funding and short borrow are modeled. Dated user metadata can change spread, borrow rate and availability. Unknown short availability is hypothetical when enabled.",
            "Missing held/ordered prices or known unavailable borrowing invalidate the affected test and overall performance; no future-survivor substitution or fictitious exit is allowed. Maximum exposures may drift between closes; breaches schedule next-close deleveraging.",
            "Basket entry rebalances the aggregate portfolio to coefficient-derived weights, preserving common caps. These are trading approximations to a log-price relation, not a perfectly replicating derivative.",
            "Training price ADF and nominal Johansen rank are diagnostics; I(1) and stable-regime assumptions are not proven. Test spread ADF adjustments cover attempted baskets, not every possible research choice or strategy profitability.",
            "Validation grid selection and repeated research can overfit. Bootstrap intervals are conditional noise estimates. Already-inspected historical tests remain development evidence; only a new forward interval is untouched.",
            "The deployment allocation is fitted after retrospective evaluation to all allowed recent history for a new paper interval. It is not used to improve historical performance and is not an instruction to buy.",
        ],
    }
    write_csv(directory / "trades.csv", ledger)
    write_csv(
        directory / "folds.csv",
        [
            {
                k: v
                for k, v in c["fold"].items()
                if k.endswith("_start")
                or k.endswith("_end")
                or k in ("number", "assets", "basket_attempts", "selected_rule")
            }
            for c in checkpoints
        ],
    )
    write_csv(
        directory / "daily_returns.csv",
        [
            {"date": str(dates[start + i]), **{name: value["daily_returns"][i] for name, value in aggregate.items()}}
            for i in range(end - start + 1)
        ],
    )
    lab_store.save_report(config.experiment_id, result)
    progress(100, "Walk-forward experiment complete; all dates, settings and trade costs saved")
    emit(
        type="lab_complete",
        summary={
            "folds": len(folds),
            "basket_attempts": total_trials,
            "elapsed": result["elapsed"],
            "metrics": aggregate["selected"]["metrics"],
        },
    )
    emit(type="complete")


if __name__ == "__main__":
    try:
        emit(type="ready", pid=os.getpid())
        run(ResearchConfig.model_validate_json(input()))
    except Exception as exc:
        emit(type="error", error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
