"""Reproducible, single-instrument monthly trend research; never sends broker orders."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.lab_analysis import metrics


def validate_prices(dates, prices):
    days = pd.DatetimeIndex(dates)
    values = np.asarray(prices, dtype=float)
    if len(days) != len(values) or len(days) < 2:
        raise ValueError("Dates and prices need matching nonempty histories")
    if days.hasnans or days.has_duplicates or not days.is_monotonic_increasing:
        raise ValueError("Dates must be valid, unique and increasing")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be observed, finite and positive; missing data is not filled")
    return days, values


def simulate(dates, prices, start, end, months=None, cost_bps=11.0, delay=1):
    """A signal at close t executes at close t+delay, before earning the next interval."""
    days, values = validate_prices(dates, prices)
    if not 0 <= start < end < len(days):
        raise ValueError("Evaluation needs at least two observed sessions within the history")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps <= 1000 or not isinstance(delay, int) or delay < 1:
        raise ValueError("Use finite costs of 0–1000 bps and an execution delay of at least one session")
    if months is not None and (not isinstance(months, int) or months < 2):
        raise ValueError("The moving average needs at least two monthly closes")
    frame = pd.DataFrame({"date": days[: end + 1], "price": values[: end + 1], "index": np.arange(end + 1)})
    # The final observed month cannot supply a later execution inside this input.
    monthly = frame.groupby(frame.date.dt.to_period("M")).last().iloc[:-1]
    signals = {}
    if months is None:
        signals[start] = (True, "predeclared buy-and-hold")
    else:
        average = monthly.price.rolling(months).mean()
        for j in range(months - 1, len(monthly)):
            row = monthly.iloc[j]
            execution = int(row["index"]) + delay
            if start <= execution <= end:
                signals[execution] = (float(row.price) > float(average.iloc[j]), str(row.date.date()))
        available = monthly.loc[monthly["index"] + delay <= start]
        if len(available) < months:
            raise ValueError("Evaluation starts before the required monthly lookback and execution delay")
        # Initialize the sleeve using only a signal whose execution was already due.
        row = available.iloc[-1]
        signals[start] = (float(row.price) > float(available.price.iloc[-months:].mean()), str(row.date.date()))
    units, cash, previous = 0.0, 1.0, 1.0
    returns, trace, orders, exposures = [], [], [], []
    rate = cost_bps / 10000
    for index in range(start, end + 1):
        target, signal_date = signals.get(index, (None, None))
        if index == end:
            target, signal_date = False, "predeclared test end"
        side, notional, charge = None, 0.0, 0.0
        if target is True and units == 0:
            notional = cash / (1 + rate)
            charge = cash - notional
            units, cash, side = notional / values[index], 0.0, "buy"
        elif target is False and units > 0:
            notional = units * values[index]
            charge = notional * rate
            cash, units, side = cash + notional - charge, 0.0, "sell"
        if side:
            orders.append(
                {
                    "date": str(days[index].date()),
                    "signal_date": signal_date,
                    "side": side,
                    "notional_fraction": notional,
                    "cost_fraction": charge,
                }
            )
        equity = cash + units * values[index]
        returns.append(equity / previous - 1)
        exposures.append(units * values[index] / equity)
        trace.append({"date": str(days[index].date()), "equity": equity})
        previous = equity
    return {
        "metrics": metrics(returns),
        "order_count": len(orders),
        "mean_end_of_day_stock_exposure": float(np.mean(exposures)),
        "orders": orders,
        "trace": trace,
    }


def study(dates, prices, start=None, end=None, cost_bps=11.0):
    days, values = validate_prices(dates, prices)
    end_index = len(days) - 1 if end is None else int(days.searchsorted(pd.Timestamp(end), side="right") - 1)
    monthly = (
        pd.Series(np.arange(end_index + 1), index=days[: end_index + 1])
        .groupby(days[: end_index + 1].to_period("M"))
        .last()
        .iloc[:-1]
    )
    if len(monthly) < 12:
        raise ValueError("Provide at least twelve completed months plus an evaluation period")
    # Twelve-month variants have one-session delay; the extra-delay variant needs ten months.
    first = int(monthly.iloc[11]) + 1
    begin = max(first, 0 if start is None else int(days.searchsorted(pd.Timestamp(start))))
    variants = {
        "Buy and hold": simulate(days, values, begin, end_index, cost_bps=cost_bps),
        "10-month trend": simulate(days, values, begin, end_index, 10, cost_bps),
        "10-month trend, double costs": simulate(days, values, begin, end_index, 10, cost_bps * 2),
        "8-month sensitivity": simulate(days, values, begin, end_index, 8, cost_bps),
        "12-month sensitivity": simulate(days, values, begin, end_index, 12, cost_bps),
        "10-month trend, extra delay": simulate(days, values, begin, end_index, 10, cost_bps, 2),
    }
    return {
        "evaluation_start": str(days[begin].date()),
        "evaluation_end": str(days[end_index].date()),
        "cost_bps_per_side": cost_bps,
        "cash_rate": 0,
        "status": "Historical research only; no broker orders or future-return guarantee",
        "notes": [
            "Signals use completed month ends; execution uses later observed daily closes.",
            "No leverage, shorting, cash interest, personal taxes or CFD financing are modelled.",
            "Costs are assumptions, not historical bid/ask prices; currency follows the input.",
            "All variants share dates; nearby lookbacks are sensitivity checks, not winner selection.",
        ],
        "variants": variants,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Date plus a positive adjusted-price column; no missing values")
    source.add_argument("--dataset-id", help="An existing local Research Studio frozen dataset ID")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--cost-bps", type=float, default=11)
    parser.add_argument("--output", type=Path, default=Path("runs/trend-research"))
    args = parser.parse_args()
    if args.csv:
        frame = pd.read_csv(args.csv)
        if "Date" not in frame or args.symbol not in frame:
            raise ValueError("CSV needs Date and the selected symbol's adjusted-price column")
        dates, prices = pd.to_datetime(frame.Date, errors="raise"), frame[args.symbol].to_numpy(float)
        provenance = {"source": str(args.csv), "symbol": args.symbol}
        from backend.lab_data import file_hash

        provenance["input_sha256"] = file_hash(args.csv)
    else:
        from backend.lab_data import load_dataset

        body, dates, logs, _ = load_dataset(args.dataset_id)
        matches = [i for i, row in enumerate(body["records"]) if row["symbol"] == args.symbol]
        if len(matches) != 1:
            raise ValueError("Choose a symbol with exactly one verified listing in the dataset")
        prices = np.exp(logs[matches[0]])
        provenance = {"source": body["name"], "symbol": args.symbol, "input_sha256": body["input_sha256"]}
    result = {**study(dates, prices, args.start, args.end, args.cost_bps), **provenance}
    from backend.lab_data import file_hash

    result["algorithm_sha256"] = file_hash(Path(__file__))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    summary = [
        {"variant": name, **value["metrics"], "orders": value["order_count"]}
        for name, value in result["variants"].items()
    ]
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    print(pd.DataFrame(summary)[["variant", "cagr", "max_drawdown", "orders"]].to_string(index=False))
    print("Saved", args.output / "report.json")


if __name__ == "__main__":
    main()
