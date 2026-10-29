"""Market-wide tiled correlations, joint factors and multi-asset basket tests."""

import math
import warnings

import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_LOG = None
_SPLIT = 0
_MINIMUM = 252


def calendar_positions(calendar, observed):
    """Match to reference sessions without filling or collapsing missing sessions."""
    positions = np.searchsorted(calendar, observed)
    valid = positions < len(calendar)
    valid[valid] &= calendar[positions[valid]] == observed[valid]
    return positions[valid], valid


def exchange_traded_candidate(row):
    """Conservative research screen; the provider ETF endpoint also lists mutual funds."""
    if "OTC" in str(row.get("exchange", "")).upper() or str(row.get("mic", "")).upper().startswith("OTC"):
        return False
    kind = row.get("type", "")
    if kind == "ETF":
        name = str(row.get("name", "")).upper()
        return "ETF" in name or "EXCHANGE TRADED" in name or "EXCHANGE-TRADED" in name
    return kind in {"Common Stock", "American Depositary Receipt", "Depositary Receipt", "REIT", "Exchange-Traded Note"}


def longest_complete(values):
    valid = np.isfinite(values).all(axis=1)
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if not len(starts):
        return 0, 0
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


def pairwise_tile(returns, start, end, minimum):
    """Exact pairwise-complete Pearson estimates; missing observations are not filled."""
    mask = np.isfinite(returns).astype(np.float64)
    clean = np.nan_to_num(returns, nan=0.0)
    a = clean[:, start:end]
    m = mask[:, start:end]
    count = m.T @ mask
    sx = a.T @ mask
    sy = m.T @ clean
    sxx = (a * a).T @ mask
    syy = m.T @ (clean * clean)
    cross = a.T @ clean
    safe = np.maximum(count, 1)
    numerator = cross - sx * sy / safe
    variance_x = sxx - sx * sx / safe
    variance_y = syy - sy * sy / safe
    denominator = np.sqrt(np.maximum(0, variance_x * variance_y))
    corr = np.divide(numerator, denominator, out=np.full_like(cross, np.nan), where=denominator > 1e-20)
    corr[count < minimum] = np.nan
    return np.clip(corr, -1, 1), count


def joint_factors(returns, ridge, component_count=5):
    """Ridge Gaussian precision using Woodbury when assets exceed observations."""
    means = np.nanmean(returns, axis=0)
    std = np.nanstd(returns, axis=0, ddof=1)
    z = np.nan_to_num((returns - means) / np.maximum(std, 1e-12), nan=0.0)
    t, n = z.shape
    d = t - 1
    if n <= t:
        covariance = z.T @ z / d
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1][:component_count]
        loadings = vectors[:, order]
        fractions = values[order] / max(float(np.trace(covariance)), 1e-20)
        precision = np.linalg.inv((1 - ridge) * covariance + ridge * np.eye(n))
        diag = np.maximum(np.diag(precision), 1e-12)
        return loadings, fractions, {"precision": precision, "diag": diag}
    gram = z @ z.T
    values, vectors = np.linalg.eigh(gram / d)
    order = np.argsort(values)[::-1][:component_count]
    loadings = z.T @ vectors[:, order] / np.sqrt(np.maximum(values[order] * d, 1e-20))
    fractions = values[order] / max(float(np.trace(gram) / d), 1e-20)
    c = (1 - ridge) / d
    factor = cho_factor(np.eye(t) + c / ridge * gram, lower=True, check_finite=False)[0]
    transformed = solve_triangular(factor, z, lower=True, check_finite=False)
    coefficient = c / (ridge * ridge)
    diag = np.maximum(1 / ridge - coefficient * np.sum(transformed * transformed, axis=0), 1e-12)
    return loadings, fractions, {"transformed": transformed, "coefficient": coefficient, "diag": diag}


def partial_tile(model, start, end):
    if "precision" in model:
        values = -model["precision"][start:end]
    else:
        u = model["transformed"]
        values = model["coefficient"] * (u[:, start:end].T @ u)
    result = values / np.sqrt(model["diag"][start:end, None] * model["diag"][None, :])
    return np.clip(result, -1, 1)


def strongest_neighbors(matrix, start, k, absolute=False):
    neighbors = []
    for offset, row in enumerate(matrix):
        scores = np.abs(row) if absolute else row.copy()
        scores = np.nan_to_num(scores, nan=-np.inf)
        scores[start + offset] = -np.inf
        take = min(k, len(scores) - 1)
        indices = np.argpartition(scores, -take)[-take:]
        indices = indices[np.argsort(-scores[indices])]
        neighbors.append([(int(j), float(row[j])) for j in indices if np.isfinite(row[j])])
    return neighbors


def graph_groups(neighbors, threshold=0.85):
    parents = list(range(len(neighbors)))

    def find(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a

    for a, row in enumerate(neighbors):
        for b, value in row:
            if value >= threshold:
                aa, bb = find(a), find(b)
                if aa != bb:
                    parents[bb] = aa
    groups = {}
    for a in range(len(neighbors)):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def generate_baskets(core, neighbors, sizes, trials, seed):
    sizes = [size for size in sizes if size <= len(core)]
    if not sizes:
        return []
    capacity = sum(math.comb(len(core), size) for size in sizes)
    target = min(max(trials, len(core)), capacity)
    rng = np.random.default_rng(seed)
    selected = set()
    for attempt in range(target * 10):
        size = sizes[attempt % len(sizes)]
        anchor = attempt % len(core)
        if attempt % 2 == 0:
            choices = [j for j, _ in neighbors[anchor] if j != anchor]
            pool = list(dict.fromkeys(choices))
            if len(pool) < size - 1:
                draw = rng.choice(len(core) - 1, size - 1, replace=False)
                others = (draw + (draw >= anchor)).tolist()
            else:
                others = rng.choice(pool, size - 1, replace=False).tolist()
        else:
            draw = rng.choice(len(core) - 1, size - 1, replace=False)
            others = (draw + (draw >= anchor)).tolist()
        selected.add(tuple(sorted([core[anchor], *(core[j] for j in others)])))
        if len(selected) >= target:
            break
    return sorted(selected)


def initialize_baskets(path, split, minimum):
    global _LOG, _SPLIT, _MINIMUM
    _LOG = np.load(path, mmap_mode="r")
    _SPLIT = split
    _MINIMUM = minimum


def analyze_basket(indices):
    prices = np.asarray(_LOG[list(indices), :]).T
    a, b = longest_complete(prices[:_SPLIT])
    if b - a < _MINIMUM:
        return {"indices": list(indices), "status": "insufficient_contiguous_training"}
    training = prices[a:b]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            test = coint_johansen(training, det_order=0, k_ar_diff=1)
        # New NumPy versions may return complex-typed eigensolutions with exactly
        # zero imaginary parts. Accept only numerically real solutions; preserve
        # rejection of genuine complex or otherwise unstable fits.
        eigenvalues = np.real_if_close(test.eig, tol=1000)
        vectors = np.real_if_close(test.evec, tol=1000)
        unsafe_warnings = [w for w in caught if not issubclass(w.category, np.exceptions.ComplexWarning)]
        if (
            unsafe_warnings
            or np.iscomplexobj(eigenvalues)
            or np.iscomplexobj(vectors)
            or not np.isfinite(test.lr1).all()
            or not np.isfinite(vectors).all()
            or np.any(eigenvalues < 0)
            or np.any(eigenvalues >= 1)
        ):
            return {"indices": list(indices), "status": "unstable_johansen"}
        vector = vectors[:, 0]
        vector = vector / max(np.abs(vector).sum(), 1e-20)
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        spread = training @ vector
        deviation = float(spread.std(ddof=1))
        if deviation < 1e-9:
            return {"indices": list(indices), "status": "degenerate_spread"}
        rank = 0
        for i in range(len(indices)):
            if test.lr1[i] <= test.cvt[i, 1]:
                break
            rank += 1
        c, d = longest_complete(prices[_SPLIT:])
        heldout_p = None
        if d - c >= 126:
            future = prices[_SPLIT + c : _SPLIT + d] @ vector
            if future.std() > 1e-9:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    heldout_p = float(adfuller(future, maxlag=1, regression="c", autolag=None)[1])
        return {
            "indices": list(indices),
            "status": "ok",
            "weights": vector.tolist(),
            "training_mean": float(spread.mean()),
            "training_std": deviation,
            "training_rows": b - a,
            "training_start": a,
            "training_end": b - 1,
            "trace_statistic": float(test.lr1[0]),
            "trace_95_critical": float(test.cvt[0, 1]),
            "training_score": float(test.lr1[0] / test.cvt[0, 1]),
            "nominal_rank": rank,
            "heldout_adf_p": heldout_p,
            "heldout_start": _SPLIT + c,
            "heldout_end": _SPLIT + d - 1,
            "heldout_rows": d - c,
            "mixed_sign": bool((vector > 0).any() and (vector < 0).any()),
            "effective_legs": int((np.abs(vector) >= 0.02).sum()),
        }
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return {"indices": list(indices), "status": "numerical_failure"}


def analyze_basket_batch(baskets):
    return [analyze_basket(basket) for basket in baskets]


def basket_backtest(logs, weights, mean, deviation, start, end, cost_bps=10, borrow=0.03):
    """Fixed adjusted-price units and next-interval execution; normalized initial equity."""
    prices = np.exp(logs)
    z = (logs @ weights - mean) / max(deviation, 1e-12)
    units = np.zeros(len(weights))
    equity = peak = 1.0
    position = age = entries = 0
    dd = fees = 0.0
    trace = []
    pnl = []
    for i in range(start, end + 1):
        previous = equity
        if position:
            equity += float(units @ (prices[i] - prices[i - 1]))
        charge = float(np.maximum(-units, 0) @ prices[i - 1]) * borrow / 252 if position else 0.0
        equity = max(0.0, equity - charge)
        fees += charge
        age += int(position != 0)
        target = position
        if equity <= 0 or i == end or (position and (abs(z[i]) < 0.5 or abs(z[i]) > 4 or age >= 20)):
            target = 0
        elif position == 0 and 2 <= abs(z[i]) <= 4:
            target = -int(np.sign(z[i]))
        if target != position:
            charge = float(np.abs(units) @ prices[i]) * cost_bps / 10000
            equity = max(0.0, equity - charge)
            fees += charge
            units[:] = 0
            if target and equity > 0:
                capital = equity / (1 + cost_bps / 10000)
                fees += equity - capital
                equity = capital
                units = target * capital * weights / prices[i]
                entries += 1
                age = 0
        position = target
        peak = max(peak, equity)
        dd = min(dd, equity / peak - 1)
        pnl.append(equity / previous - 1 if previous > 0 else 0.0)
        trace.append({"index": i, "equity": equity, "z": float(z[i])})
    take = np.unique(np.linspace(0, len(trace) - 1, min(80, len(trace)), dtype=int))
    std = np.std(pnl, ddof=1)
    return {
        "return": equity - 1,
        "max_drawdown": dd,
        "sharpe": float(np.mean(pnl) / std * np.sqrt(252)) if std > 1e-12 else None,
        "entries": entries,
        "fees_fraction_initial": fees,
        "trace": [trace[i] for i in take],
    }
