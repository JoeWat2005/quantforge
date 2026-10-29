"""Validated simulation and resource policy; all rates are illustrative inputs."""

import os
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


class Asset(BaseModel):
    name: str = Field(max_length=40)
    weight: float = Field(ge=0, le=1, allow_inf_nan=False)
    mu: float = Field(ge=-0.5, le=0.5, allow_inf_nan=False)
    sigma: float = Field(ge=0, le=1, allow_inf_nan=False)


DEFAULT_ASSETS = [
    Asset(name="US equities", weight=0.45, mu=0.07, sigma=0.18),
    Asset(name="Government bonds", weight=0.25, mu=0.035, sigma=0.07),
    Asset(name="Gold", weight=0.15, mu=0.04, sigma=0.16),
    Asset(name="Technology", weight=0.15, mu=0.09, sigma=0.28),
]
DEFAULT_CORR = [[1, -0.15, 0.05, 0.8], [-0.15, 1, 0.1, -0.1], [0.05, 0.1, 1, 0.05], [0.8, -0.1, 0.05, 1]]


class RunConfig(BaseModel):
    paths: int = Field(default=1_000_000, ge=100, le=100_000_000_000)
    steps: int = Field(default=12, ge=1, le=4096)
    horizon: float = Field(default=1, gt=0, le=30, allow_inf_nan=False)
    initial_value: float = Field(default=100_000, gt=0, le=1e12, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**64 - 1)
    engine: Literal["cpu", "gpu", "compare"] = "cpu"
    threads: int = Field(default=min(64, os.cpu_count() or 1), ge=1, le=os.cpu_count() or 1)
    cpu_ceiling: int = Field(default=90, ge=1, le=100)
    gpu_duty: int = Field(default=90, ge=1, le=100)
    vram_gb: float = Field(default=8, ge=0.125, le=14, allow_inf_nan=False)
    batch_size: int = Field(default=131_072, ge=1024, le=4_194_304)
    ram_gb: float = Field(default=8, ge=0.25, le=224, allow_inf_nan=False)
    reserve_gb: float = Field(default=24, ge=4, le=128, allow_inf_nan=False)
    memory_stress: bool = False
    cpu_temp_limit: float = Field(default=85, ge=50, le=90, allow_inf_nan=False)
    gpu_temp_limit: float = Field(default=80, ge=50, le=85, allow_inf_nan=False)
    require_cpu_temperature: bool = False
    max_seconds: int = Field(default=3600, ge=1, le=86400)
    confidence: float = Field(default=0.99, ge=0.9, le=0.9999, allow_inf_nan=False)
    assets: list[Asset] = Field(
        default_factory=lambda: [x.model_copy() for x in DEFAULT_ASSETS], min_length=1, max_length=16
    )
    correlation: list[list[float]] = Field(default_factory=lambda: [row[:] for row in DEFAULT_CORR])

    @model_validator(mode="after")
    def validate_portfolio(self):
        if abs(sum(a.weight for a in self.assets) - 1) > 1e-8:
            raise ValueError("Portfolio weights must sum to one")
        c = np.asarray(self.correlation, dtype=float)
        n = len(self.assets)
        if c.shape != (n, n) or not np.isfinite(c).all():
            raise ValueError("Correlation dimensions must match assets and contain finite values")
        if not np.allclose(c, c.T, atol=1e-12) or not np.allclose(c.diagonal(), 1, atol=1e-12) or (np.abs(c) > 1).any():
            raise ValueError("Correlation must be symmetric with unit diagonal and entries in [-1,1]")
        try:
            np.linalg.cholesky(c)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Correlation must be positive definite") from exc
        return self


PRESETS = {
    "quick": {"paths": 1_000_000, "steps": 12, "ram_gb": 8, "require_cpu_temperature": False},
    "heavy": {
        "paths": 100_000_000,
        "steps": 252,
        "ram_gb": 32,
        "threads": os.cpu_count() or 1,
        "cpu_ceiling": 95,
        "require_cpu_temperature": True,
    },
    "extreme": {
        "paths": 1_000_000_000,
        "steps": 252,
        "ram_gb": 192,
        "threads": os.cpu_count() or 1,
        "cpu_ceiling": 100,
        "require_cpu_temperature": True,
    },
}
