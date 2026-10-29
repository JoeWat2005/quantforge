"""Real adjusted daily prices, provenance and explicit quality checks."""

import csv
import hashlib
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from backend.config import RunConfig
from backend.engine import ROOT

DATA = Path(os.environ.get("QF_DATA_DIR", str(ROOT / "data"))).resolve()
UNIVERSE = "SPY QQQ IWM DIA VTI VOO IVV RSP VTV VUG SCHD XLK XLF XLV XLE XLI XLP XLY XLU XLB XLRE TLT IEF SHY TIP LQD HYG GLD IAU SLV VNQ AAPL MSFT NVDA AMZN GOOGL META AVGO ORCL CRM AMD INTC CSCO JPM BAC GS MS V MA XOM CVX COP UNH JNJ MRK ABBV PFE PG KO PEP WMT COST HD MCD DIS NFLX CAT DE HON GE RTX UPS FDX NKE SBUX TMO ABT DHR LIN NEE DUK SO PLD AMT BRK-B BLK SCHW CME IBM INTU ADBE AMAT MU QCOM TXN ADI GILD AMGN BMY MDT LOW TGT".split()


class DiscoveryConfig(RunConfig):
    source: str = Field(default="yahoo", pattern="^(yahoo|csv)$")
    tickers: list[str] = Field(default_factory=lambda: UNIVERSE[:], min_length=3, max_length=1000)
    years: int = Field(default=5, ge=3, le=20)
    dataset_id: str = Field(default="", max_length=64, pattern="^[a-f0-9]*$")
    train_fraction: float = Field(default=0.7, ge=0.5, le=0.8, allow_inf_nan=False)
    shortlist: int = Field(default=100, ge=1, le=5000)
    bootstrap_repetitions: int = Field(default=500, ge=50, le=10000)
    block_days: int = Field(default=20, ge=2, le=63)
    rolling_days: int = Field(default=126, ge=21, le=252)
    cost_bps: float = Field(default=10, ge=0, le=100, allow_inf_nan=False)
    borrow_rate: float = Field(default=0.03, ge=0, le=0.5, allow_inf_nan=False)
    anchor: str = Field(default="SPY", max_length=12)
    ram_gb: float = Field(default=8, ge=2, le=64, allow_inf_nan=False)
    require_cpu_temperature: bool = True
    max_seconds: int = Field(default=600, ge=1, le=86400)

    @field_validator("tickers")
    @classmethod
    def valid_tickers(cls, values):
        values = list(dict.fromkeys(x.strip().upper().replace(".", "-") for x in values))
        if len(values) < 3 or any(not re.fullmatch(r"[A-Z][A-Z0-9^-]{0,11}", x) for x in values):
            raise ValueError("Provide at least three distinct US ticker symbols, up to 12 characters each")
        return values


def parse_prices_csv(raw: str):
    """Wide format: Date,AAPL,MSFT,...; values must be positive adjusted prices."""
    header = next(csv.reader(io.StringIO(raw)), [])
    if not header or header[0].strip().lower() != "date" or len(header) < 4:
        raise ValueError("CSV must start Date,TICKER1,TICKER2,TICKER3,... with adjusted daily prices")
    if len(header) > 1001 or len(set(x.strip().upper() for x in header)) != len(header):
        raise ValueError("CSV must have unique column names and at most 1000 assets")
    frame = pd.read_csv(io.StringIO(raw), index_col=0)
    if len(frame) > 10000:
        raise ValueError("CSV supports at most 10,000 dates")
    frame.index = pd.to_datetime(frame.index, errors="raise", utc=True).normalize().tz_localize(None)
    if frame.index.isna().any():
        raise ValueError("Every row needs a valid date")
    if frame.index.duplicated().any():
        raise ValueError("CSV has duplicate dates")
    frame.columns = [str(x).strip().upper() for x in frame.columns]
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.^-]{0,11}", name) for name in frame.columns):
        raise ValueError("CSV asset headers must be US ticker symbols, up to 12 characters")
    frame = frame.apply(pd.to_numeric, errors="raise").sort_index()
    if ((frame <= 0) | np.isinf(frame)).any().any():
        raise ValueError("Prices must be positive finite numbers or empty cells")
    return frame


def _fetch_ticker(symbol, years):
    directory = DATA / "price_cache"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol}_{years}y.json"
    if path.exists() and time.time() - path.stat().st_mtime < 86400:
        return json.loads(path.read_text(encoding="utf-8"))
    response = httpx.get(
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"period1": int(time.time() - years * 365.25 * 86400), "period2": int(time.time()), "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    if response.status_code == 429:
        raise RuntimeError("Provider rate limit (429). Stop downloading and use cached data or CSV; retry later.")
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    if result["meta"].get("currency") != "USD":
        raise ValueError("Only USD-priced assets are supported in this US universe")
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adjusted is None:
        raise ValueError("Provider did not return adjusted close prices")
    today = datetime.now(timezone.utc).date().isoformat()
    rows = [
        (datetime.fromtimestamp(t, timezone.utc).date().isoformat(), float(p))
        for t, p in zip(result["timestamp"], adjusted, strict=True)
        if p is not None
        and np.isfinite(p)
        and p > 0
        and datetime.fromtimestamp(t, timezone.utc).date().isoformat() < today
    ]
    data = {
        "symbol": symbol,
        "rows": rows,
        "source": "Yahoo Finance chart adjusted close (unofficial endpoint)",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "currency": "USD",
    }
    if sum(p.stat().st_size for p in directory.glob("*.json")) + len(json.dumps(data)) > 256 * 1024**2:
        raise ValueError("Automatic price cache reached 256 MiB; manage data/price_cache before downloading more")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data), encoding="utf-8")
    temp.replace(path)
    return data


def load_prices(config, progress):
    failures = []
    provenance = []
    if config.source == "csv":
        path = DATA / "imports" / (config.dataset_id + ".csv")
        if not config.dataset_id or not path.is_file():
            raise ValueError("Import a CSV before selecting CSV source")
        frame = parse_prices_csv(path.read_text(encoding="utf-8"))
        provenance = [
            {
                "source": "User-supplied CSV; adjustment and currency declared by user",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ]
    else:
        series = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_fetch_ticker, ticker, config.years): ticker for ticker in config.tickers}
            for done, future in enumerate(as_completed(futures), 1):
                ticker = futures[future]
                try:
                    data = future.result()
                    series[ticker] = pd.Series(dict(data["rows"]), name=ticker, dtype=float)
                    provenance.append({k: v for k, v in data.items() if k != "rows"})
                except RuntimeError:
                    for pending in futures:
                        pending.cancel()
                    raise
                except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                    failures.append({"symbol": ticker, "reason": str(exc)[:160]})
                progress(20 * done / len(futures), f"Loading adjusted prices · {done}/{len(futures)}")
        if len(series) < 3:
            raise ValueError(
                "Fewer than three assets downloaded. Import licensed adjusted-price CSV data or try later."
            )
        frame = pd.DataFrame(series)
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index().reindex(columns=sorted(series))
    original_rows = len(frame)
    coverage = frame.notna().mean()
    dropped = coverage[coverage < 0.95].index.tolist()
    frame = frame.drop(columns=dropped).dropna()
    if len(frame.columns) < 3 or len(frame) < 504:
        raise ValueError(
            "Need at least three assets with 504 overlapping complete daily observations; sparse assets are excluded"
        )
    if (np.diff(np.log(frame.to_numpy()), axis=0).std(axis=0) < 1e-8).any():
        raise ValueError("A constant-price asset has no measurable return correlation; remove it")
    return frame, {
        "source": config.source,
        "start": str(frame.index[0].date()),
        "end": str(frame.index[-1].date()),
        "rows": len(frame),
        "assets": len(frame.columns),
        "excluded_sparse_assets": dropped,
        "rows_removed_for_missing_values": original_rows - len(frame),
        "download_failures": failures,
        "provenance": provenance,
        "adjustment_note": "Provider-adjusted closing prices; verify corporate actions before relying on results. No forward filling.",
        "selection_warning": "User-chosen present-day universe; no point-in-time membership or delisted-stock coverage. Survivorship/selection bias is possible.",
    }
