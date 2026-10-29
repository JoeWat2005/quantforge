"""Daily close ledger with delayed execution, common portfolio caps and explicit missing-data failure."""

import numpy as np

from backend.lab_analysis import metrics
from backend.lab_data import asof_metadata


def assumed_value(attributes, name, fallback):
    value = attributes.get(name)
    return fallback if value is None or value == "" else float(value)


def project_weights(weights, config, attributes=None):
    result = np.asarray(weights, float).copy()
    gross = float(np.abs(result).sum())
    net = abs(float(result.sum()))
    largest = float(np.abs(result).max(initial=0))
    scale = min(
        1,
        config.max_gross / max(gross, 1e-20),
        config.max_net / max(net, 1e-20),
        config.max_weight / max(largest, 1e-20),
    )
    if attributes:
        sectors = {a.get("sector") for a in attributes if a.get("sector")}
        for sector in sectors:
            exposure = sum(abs(result[i]) for i, a in enumerate(attributes) if a.get("sector") == sector)
            scale = min(scale, config.sector_cap / max(exposure, 1e-20))
    return result * scale


def cost_of_orders(orders, prices, turnover, volatility, attributes, config, multiplier=1):
    notional = np.abs(orders * prices)
    spreads = np.array([assumed_value(a, "spread_bps", config.spread_bps) for a in attributes])
    adv = np.where(np.isfinite(turnover) & (turnover > 0), turnover, max(config.minimum_turnover, 1e6))
    impact = (
        config.impact_coefficient
        * np.nan_to_num(volatility, nan=0.3)
        / np.sqrt(252)
        * np.sqrt(notional * config.capital / adv)
    )
    rate = (config.commission_bps + spreads / 2 + config.slippage_bps) / 10000 + impact
    return notional * rate * multiplier


def run_portfolio(
    logs,
    dates,
    rows,
    model,
    start,
    end,
    config,
    family="diversified",
    baskets=None,
    entry=2,
    holding=20,
    multiplier=1,
    capture=True,
    fixed_weights=None,
    dated_metadata=None,
    close_at_end=True,
):
    """Prices are adjusted closes. A signal observed at t is executed at t+1 close.

    Long-only allocation is known before the window and enters at its first close.
    All positions close at the predeclared final close. Missing held/ordered marks
    invalidate the window; securities are not dropped using later availability.
    """
    n = logs.shape[0]
    prices = np.exp(np.asarray(logs))
    attributes = model["attributes"]
    units = np.zeros(n)
    cash = equity = peak = 1.0
    fee_total = borrow_total = turnover_total = 0.0
    positions = np.zeros(len(baskets or []), int)
    ages = np.zeros_like(positions)
    pending = None
    previous_prices = None
    previous_equity = 1.0
    previous_month = None
    halted = False
    executions = 0
    records = []
    daily = []
    trace = []
    breaches = 0
    max_gross = max_net = max_name = 0.0
    weights = np.array(fixed_weights if fixed_weights is not None else model["weights"], float)
    if family != "baskets":
        pending = project_weights(weights, config, attributes)
    status = "evaluated"
    reason = None
    for index in range(start, end + 1):
        if dated_metadata is not None:
            attributes = [
                {**model["attributes"][j], **asof_metadata(dated_metadata, row["symbol"], str(dates[index]))}
                for j, row in enumerate(rows)
            ]
        today = prices[:, index]
        needed = np.abs(units) > 1e-15
        if pending is not None:
            needed |= np.abs(pending) > 1e-15
        if not np.isfinite(today[needed]).all():
            status = "invalid_missing_prices"
            reason = f"Missing held or ordered price on {dates[index]}; no fictitious exit or survivor substitution"
            break
        short_needed = (units < 0) | ((pending < 0) if pending is not None else False)
        if any(short_needed[j] and a.get("borrow_available", "").lower() == "false" for j, a in enumerate(attributes)):
            status = "invalid_borrow_unavailable"
            reason = f"Known unavailable borrowing on {dates[index]}; no invented short fill"
            break
        safe_prices = np.where(np.isfinite(today), today, 1.0)
        if previous_prices is not None:
            cash += max(cash - float(np.maximum(-units, 0) @ previous_prices), 0) * config.cash_rate / 252
            cash -= max(-cash, 0) * config.funding_rate / 252 * multiplier
            rates = np.array([assumed_value(a, "borrow_rate", config.borrow_rate) for a in attributes])
            charge = float(np.maximum(-units, 0) * previous_prices @ rates) / 252 * multiplier
            cash -= charge
            borrow_total += charge
        equity = cash + float(units @ safe_prices)
        if equity <= 0:
            # Insolvency is a complete loss, never an unbounded negative NAV.
            units[:] = 0
            cash = equity = 0
            pending = None
            halted = True
        if index == end and close_at_end:
            pending = np.zeros(n)
        if pending is not None and equity > 0:
            target = project_weights(pending, config, attributes)
            after = equity
            for _ in range(6):
                target_units = after * target / safe_prices
                orders = target_units - units
                costs = cost_of_orders(
                    orders, safe_prices, model["turnover"], model["volatility"], attributes, config, multiplier
                )
                after = max(0.0, equity - float(costs.sum()))
            target_units = after * target / safe_prices
            orders = target_units - units
            costs = cost_of_orders(
                orders, safe_prices, model["turnover"], model["volatility"], attributes, config, multiplier
            )
            cash -= float(orders @ safe_prices) + float(costs.sum())
            executions += int((np.abs(orders) > 1e-12).sum())
            fee_total += float(costs.sum())
            turnover_total += float(np.abs(orders * safe_prices).sum())
            if capture:
                for j in np.flatnonzero(np.abs(orders) > 1e-12):
                    records.append(
                        {
                            "date": str(dates[index]),
                            "symbol": rows[j]["symbol"],
                            "side": "buy" if orders[j] > 0 else "sell",
                            "units_per_initial_currency": float(orders[j]),
                            "price": float(safe_prices[j]),
                            "notional": float(abs(orders[j] * safe_prices[j]) * config.capital),
                            "cost": float(costs[j] * config.capital),
                            "reason": "scheduled_end"
                            if index == end and close_at_end
                            else "delayed_signal_or_rebalance",
                        }
                    )
            units = target_units
            equity = max(0.0, cash + float(units @ safe_prices))
            pending = None
        current = units * safe_prices / max(equity, 1e-20)
        gross = float(np.abs(current).sum())
        net = abs(float(current.sum()))
        largest = float(np.abs(current).max(initial=0))
        max_gross = max(max_gross, gross)
        max_net = max(max_net, net)
        max_name = max(max_name, largest)
        peak = max(peak, equity)
        if equity / peak - 1 <= -config.drawdown_stop:
            halted = True
            pending = np.zeros(n)
        month = str(dates[index])[:7]
        if not halted and index < end:
            if family == "baskets":
                changed = False
                for k, basket in enumerate(baskets or []):
                    ids = basket["indices"]
                    observed = logs[ids, index]
                    if not np.isfinite(observed).all():
                        continue
                    z = float(
                        (observed @ np.array(basket["weights"]) - basket["training_mean"])
                        / max(basket["training_std"], 1e-12)
                    )
                    ages[k] += int(positions[k] != 0)
                    if positions[k] and (abs(z) < config.exit_z or abs(z) > config.stop_z or ages[k] >= holding):
                        positions[k] = 0
                        ages[k] = 0
                        changed = True
                    elif not positions[k] and entry <= abs(z) <= config.stop_z:
                        positions[k] = -int(np.sign(z))
                        changed = True
                if changed:
                    desired = np.zeros(n)
                    for position, basket in zip(positions, baskets or []):
                        desired[basket["indices"]] += (
                            position * np.array(basket["weights"]) / max(1, len(baskets or []))
                        )
                    pending = project_weights(desired, config, attributes)
            elif previous_month is not None and month != previous_month:
                # Month boundary observed today schedules the next close, not today's close.
                pending = project_weights(weights, config, attributes)
            if gross > config.max_gross * 1.02 or net > config.max_net * 1.02 or largest > config.max_weight * 1.02:
                breaches += 1
                pending = project_weights(current, config, attributes)
        previous_month = month
        previous_prices = safe_prices.copy()
        change = equity / previous_equity - 1 if previous_equity > 0 else 0.0
        daily.append(float(change))
        previous_equity = equity
        trace.append({"date": str(dates[index]), "equity": float(equity), "gross": gross, "net": float(current.sum())})
    summary = metrics(daily, config.cash_rate)
    if status != "evaluated":
        summary = {"status": status, "return": None, "cagr": None, "sharpe": None, "max_drawdown": None}
    return {
        "status": status,
        "reason": reason,
        "metrics": summary,
        "daily_returns": daily,
        "trace": trace,
        "trades": records,
        "fees_fraction": fee_total,
        "borrow_fraction": borrow_total,
        "turnover": turnover_total,
        "max_gross": max_gross,
        "max_net": max_net,
        "max_name": max_name,
        "drift_deleveraging_events": breaches,
        "drawdown_stop_triggered": halted,
        "start": str(dates[start]),
        "end": str(dates[end]),
        "actual_end": trace[-1]["date"] if trace else None,
        "executions": executions,
    }


def benchmark(logs, dates, start, end, config):
    prices = np.exp(np.asarray(logs[start : end + 1]))
    if not np.isfinite(prices).all():
        return {"status": "invalid_missing_prices", "daily_returns": [], "metrics": metrics([]), "trace": []}
    cost = (config.commission_bps + config.spread_bps / 2 + config.slippage_bps) / 10000
    nav = prices / prices[0] / (1 + cost)
    nav[-1] *= 1 - cost
    # SPY buy-and-hold comparison omits impact and has no borrow/funding exposure.
    r = np.diff(np.r_[1.0, nav]) / np.r_[1.0, nav[:-1]]
    return {
        "status": "evaluated",
        "metrics": metrics(r, config.cash_rate),
        "daily_returns": r.tolist(),
        "trace": [{"date": str(day), "equity": float(v)} for day, v in zip(dates[start : end + 1], nav)],
    }
