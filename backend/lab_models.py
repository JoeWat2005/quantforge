"""Validated, serializable research specifications; dates are never inferred from results."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ManualDates(BaseModel):
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    @model_validator(mode="after")
    def chronological(self):
        if not (
            self.train_start
            < self.train_end
            < self.validation_start
            < self.validation_end
            < self.test_start
            < self.test_end
        ):
            raise ValueError("Dates must follow training → validation → test, without overlap")
        return self


class ResearchConfig(BaseModel):
    engine: Literal["cpu"] = "cpu"
    require_cpu_temperature: Literal[True] = True
    seed: int = Field(default=42, ge=0, le=2**64 - 1)
    cpu_ceiling: int = Field(default=100, ge=1, le=100)
    reserve_gb: float = Field(default=24, ge=24, le=128, allow_inf_nan=False)
    cpu_temp_limit: float = Field(default=85, ge=50, le=85, allow_inf_nan=False)
    gpu_temp_limit: float = Field(default=80, ge=50, le=85, allow_inf_nan=False)
    experiment_id: str = Field(default="", pattern=r"^[a-f0-9]{0,32}$")
    dataset_id: str = Field(default="", pattern=r"^[a-f0-9]{0,32}$")
    await_snapshot_id: str = Field(default="", pattern=r"^[a-f0-9]{0,32}$")
    name: str = Field(default="Market relationship experiment", min_length=1, max_length=100)
    start_date: date | None = None
    end_date: date | None = None
    split_mode: Literal["rolling", "expanding", "manual"] = "rolling"
    manual_dates: ManualDates | None = None
    train_years: int = Field(default=3, ge=2, le=15)
    validation_months: int = Field(default=12, ge=3, le=36)
    test_months: int = Field(default=12, ge=3, le=36)
    step_months: int = Field(default=12, ge=3, le=36)
    gap_sessions: int = Field(default=20, ge=1, le=126)
    frequency: Literal["daily", "weekly"] = "daily"
    factor_mode: Literal["raw", "market", "market_sectors"] = "market_sectors"
    max_assets: int = Field(default=300, ge=10, le=2000)
    stocks_only: bool = True
    symbols: list[str] = Field(default_factory=list, max_length=5000)
    exclude_symbols: list[str] = Field(default_factory=list, max_length=5000)
    basket_trials: int = Field(default=1000, ge=10, le=100000)
    basket_sizes: list[int] = Field(default_factory=lambda: [3, 5, 8], min_length=1, max_length=6)
    minimum_coverage: float = Field(default=0.95, ge=0.9, le=1, allow_inf_nan=False)
    minimum_turnover: float = Field(default=1e6, ge=0, le=1e10, allow_inf_nan=False)
    allow_unknown_liquidity: bool = False
    allow_unverified_borrow: bool = True
    ridge: float = Field(default=0.2, ge=0.05, le=0.8, allow_inf_nan=False)
    strategy: Literal["compare", "diversified", "baskets"] = "compare"
    entry_grid: list[float] = Field(default_factory=lambda: [1.5, 2, 2.5], min_length=1, max_length=4)
    holding_grid: list[int] = Field(default_factory=lambda: [10, 20], min_length=1, max_length=3)
    exit_z: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    stop_z: float = Field(default=4, ge=3, le=8, allow_inf_nan=False)
    max_positions: int = Field(default=12, ge=5, le=32)
    max_baskets: int = Field(default=5, ge=1, le=10)
    max_gross: float = Field(default=1, ge=0.1, le=2, allow_inf_nan=False)
    max_net: float = Field(default=1, ge=0.05, le=2, allow_inf_nan=False)
    max_weight: float = Field(default=0.15, ge=0.02, le=0.5, allow_inf_nan=False)
    volatility_target: float = Field(default=0.12, ge=0.03, le=0.4, allow_inf_nan=False)
    sector_cap: float = Field(default=0.35, ge=0.1, le=1, allow_inf_nan=False)
    drawdown_stop: float = Field(default=0.35, ge=0.05, le=0.9, allow_inf_nan=False)
    capital: float = Field(default=100000, ge=1000, le=1e9, allow_inf_nan=False)
    commission_bps: float = Field(default=1, ge=0, le=100, allow_inf_nan=False)
    spread_bps: float = Field(default=5, ge=0, le=500, allow_inf_nan=False)
    slippage_bps: float = Field(default=2, ge=0, le=100, allow_inf_nan=False)
    impact_coefficient: float = Field(default=0.1, ge=0, le=2, allow_inf_nan=False)
    borrow_rate: float = Field(default=0.03, ge=0, le=2, allow_inf_nan=False)
    cash_rate: float = Field(default=0, ge=0, le=0.15, allow_inf_nan=False)
    funding_rate: float = Field(default=0.06, ge=0, le=0.5, allow_inf_nan=False)
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    block_sessions: int = Field(default=20, ge=5, le=126)
    ram_gb: float = Field(default=32, ge=4, le=192, allow_inf_nan=False)
    threads: int = Field(default=32, ge=1, le=128)
    max_seconds: int = Field(default=7200, ge=30, le=86400)

    @field_validator("symbols", "exclude_symbols")
    @classmethod
    def symbol_list(cls, values):
        import re

        clean = sorted(set(v.strip().upper() for v in values if v.strip()))
        if any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-^=]{0,23}", v) for v in clean):
            raise ValueError("Invalid symbol in universe filter")
        return clean

    @model_validator(mode="after")
    def validate_research(self):
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must precede end date")
        if self.split_mode == "manual" and self.manual_dates is None:
            raise ValueError("Manual mode requires all six training, validation and test dates")
        if self.step_months < self.test_months:
            raise ValueError("Walk-forward steps must be at least the test length, preventing overlapping test returns")
        if any(k < 3 or k > 8 for k in self.basket_sizes):
            raise ValueError("Basket sizes must be 3–8")
        if any(not 1 <= h <= 60 for h in self.holding_grid) or self.gap_sessions < max(self.holding_grid):
            raise ValueError(
                "Holding periods must be 1–60 sessions and boundary gaps must cover the longest holding period"
            )
        if any(not 1 <= z <= 3 or z >= self.stop_z for z in self.entry_grid):
            raise ValueError("Entry thresholds must be 1–3 and below the stop threshold")
        self.basket_sizes = sorted(set(self.basket_sizes))
        self.entry_grid = sorted(set(self.entry_grid))
        self.holding_grid = sorted(set(self.holding_grid))
        return self


class ResourcePolicy(BaseModel):
    ram_ceiling_gb: float = Field(default=192, ge=8, le=192, allow_inf_nan=False)
    thread_ceiling: int = Field(default=128, ge=1, le=128)
    cpu_stop: float = Field(default=85, ge=50, le=85, allow_inf_nan=False)
    os_reserve_gb: float = Field(default=24, ge=24, le=128, allow_inf_nan=False)
    duration_ceiling: int = Field(default=21600, ge=60, le=86400)
