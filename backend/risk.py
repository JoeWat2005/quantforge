"""Bounded streaming risk statistics, with explicit quantile approximation bounds."""

import numpy as np


class StreamingRisk:
    def __init__(self, value: float, confidence: float):
        self.value, self.confidence = value, confidence
        self.edges = np.linspace(-20, 20, 32769)
        self.counts = np.zeros(32768, dtype=np.int64)
        self.sums = np.zeros(32768)
        self.n = 0
        self.total = 0.0
        self.overflow = 0
        self.sample = []

    def add(self, ratios: np.ndarray):
        logs = np.log(ratios)
        self.overflow += int(((logs < -20) | (logs >= 20)).sum())
        if self.overflow:
            raise ValueError(
                "Scenario exceeds log-wealth histogram range [-20,20]; reduce model horizon or volatility. Risk bounds would be invalid."
            )
        indices = np.clip(np.searchsorted(self.edges, logs, side="right") - 1, 0, 32767)
        self.counts += np.bincount(indices, minlength=32768)
        self.sums += np.bincount(indices, weights=(1 - ratios) * self.value, minlength=32768)
        self.n += len(ratios)
        self.total += float(ratios.sum())
        if len(self.sample) < 256:
            self.sample.extend(ratios[: 256 - len(self.sample)].tolist())

    def result(self):
        if not self.n:
            return None
        tail = self.n * (1 - self.confidence)
        cumulative = np.cumsum(self.counts)
        b = min(int(np.searchsorted(cumulative, tail, side="left")), 32767)
        before = int(cumulative[b - 1]) if b else 0
        fraction = (tail - before) / max(1, int(self.counts[b]))
        low = self.value * (1 - np.exp(self.edges[b + 1]))
        high = self.value * (1 - np.exp(self.edges[b]))
        cvar = (float(self.sums[:b].sum()) + fraction * float(self.sums[b])) / tail
        # Fixed chart bins; include all histogram counts in the visible range.
        occupied = np.flatnonzero(self.counts)
        groups = np.array_split(np.arange(occupied[0], occupied[-1] + 1), min(70, occupied[-1] - occupied[0] + 1))
        histogram = [
            {"value": float(np.exp(self.edges[g[len(g) // 2]]) * self.value), "count": int(self.counts[g].sum())}
            for g in groups
        ]
        return {
            "n": self.n,
            "mean": self.total / self.n * self.value,
            "var": float((low + high) / 2),
            "var_bounds": [float(low), float(high)],
            "cvar": cvar,
            "cvar_error_bound": float(high - low),
            "confidence": self.confidence,
            "histogram": histogram,
            "sample": self.sample,
            "histogram_overflow": self.overflow,
            "method": "Log-wealth histogram; fractional boundary-bin CVaR. Bounds valid only without overflow.",
        }
