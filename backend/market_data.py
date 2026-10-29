"""Twelve Data local cache and resumable US-universe collector. Never export keys."""

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
import numpy as np
import psutil
from pydantic import Field, field_validator

from backend.config import RunConfig
from backend.discovery_data import DATA, UNIVERSE
from backend.engine import ROOT
from backend.telemetry import GIB

MARKET = DATA / "market"
STATUS = MARKET / "collection.json"
BENCHMARK_SYMBOLS = {"SPY", "XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE", "IWM", "TLT"}


def stock_scope():
    path = MARKET / "scope.json"
    return json.loads(path.read_text()).get("stocks_only", True) if path.exists() else True


class MarketConfig(RunConfig):
    engine: Literal["cpu"] = "cpu"
    years: int = Field(default=10, ge=3, le=15)
    snapshot_id: str = Field(default="", pattern="^[a-f0-9]{0,32}$")
    basket_trials: int = Field(default=20000, ge=10, le=200000)
    basket_sizes: list[int] = Field(default_factory=lambda: [3, 5, 8], min_length=1, max_length=6)
    neighbor_count: int = Field(default=12, ge=3, le=50)
    minimum_overlap: int = Field(default=252, ge=126, le=1000)
    train_fraction: float = Field(default=0.8, ge=0.6, le=0.9, allow_inf_nan=False)
    ridge: float = Field(default=0.2, ge=0.05, le=0.8, allow_inf_nan=False)
    cost_bps: float = Field(default=10, ge=0, le=100, allow_inf_nan=False)
    borrow_rate: float = Field(default=0.03, ge=0, le=0.5, allow_inf_nan=False)
    ram_gb: float = Field(default=64, ge=8, le=192, allow_inf_nan=False)
    threads: int = Field(default=os.cpu_count() or 1, ge=1, le=os.cpu_count() or 1)
    cpu_ceiling: int = Field(default=100, ge=1, le=100)
    require_cpu_temperature: Literal[True] = True
    max_seconds: int = Field(default=14400, ge=1, le=86400)

    @field_validator("basket_sizes")
    @classmethod
    def valid_sizes(cls, values):
        if any(size < 3 or size > 8 for size in values):
            raise ValueError("Basket sizes must be between 3 and 8")
        return sorted(set(values))


def atomic_json(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(body, allow_nan=False), encoding="utf-8")
    # Windows readers and antivirus can briefly deny replacement of an open file.
    # Keep the old complete snapshot visible and retry instead of losing the queue.
    for attempt in range(25):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 24:
                raise
            time.sleep(min(0.1, 0.01 * (attempt + 1)))


@contextmanager
def collection_lease(path=None):
    """OS-released lease prevents duplicate collectors, including after forced stops."""
    lock_path = path or (MARKET / "collector.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_key():
    value = os.environ.get("TWELVEDATA_API_KEY", "")
    if not value:
        path = ROOT / ".secrets" / "twelvedata.key"
        value = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    if not value or "\n" in value or "\r" in value:
        raise ValueError("Twelve Data API key is not configured on this PC")
    return value


def collection_state():
    if not STATUS.exists():
        return {
            "status": "not_started",
            "key_configured": bool(os.environ.get("TWELVEDATA_API_KEY") or (ROOT / ".secrets/twelvedata.key").exists()),
        }
    state = json.loads(STATUS.read_text(encoding="utf-8"))
    pid = state.get("pid")
    if state.get("status") in ("running", "waiting") and pid:
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - state.get("process_created", 0)) > 0.1:
                state["status"] = "paused"
        except psutil.NoSuchProcess:
            state["status"] = "paused"
    return state


def database(snapshot_id):
    directory = MARKET / "snapshots" / snapshot_id
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "catalog.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS instruments(
      id TEXT PRIMARY KEY,symbol TEXT,name TEXT,exchange TEXT,mic TEXT,type TEXT,currency TEXT,
      status TEXT DEFAULT 'pending',bars INTEGER DEFAULT 0,first_date TEXT,last_date TEXT,
      median_turnover REAL DEFAULT 0,bytes INTEGER DEFAULT 0,error TEXT DEFAULT '',priority INTEGER DEFAULT 0)""")
    if "attempts" not in [row[1] for row in db.execute("PRAGMA table_info(instruments)")]:
        db.execute("ALTER TABLE instruments ADD COLUMN attempts INTEGER DEFAULT 0")
    db.execute("CREATE INDEX IF NOT EXISTS queue_order ON instruments(status,priority,symbol,id)")
    return db, directory


def prioritize_queue(db):
    db.execute("""UPDATE instruments SET priority=
      (CASE WHEN type='ETF' THEN 10000
       WHEN type IN ('Common Stock','American Depositary Receipt','REIT','Depositary Receipt','Limited Partnership')
       THEN 1000 ELSE 5000 END)+(CASE WHEN exchange='OTC' THEN 20000 ELSE 0 END)
       WHERE status='pending' AND priority>=1000""")
    db.commit()


class TwelveClient:
    def __init__(self):
        self.key = read_key()
        self.client = httpx.Client(
            base_url="https://api.twelvedata.com", headers={"Authorization": "apikey " + self.key}, timeout=60
        )
        self.limit = 55
        self.next_request = 0.0

    def request(self, path, params=None, credit_cost=1):
        delay = self.next_request - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        from backend.provider_budget import acquire, configure

        try:
            state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
            if state.get("credits_per_minute"):
                configure(state["credits_per_minute"])
        except (OSError, ValueError):
            pass
        acquire(credit_cost)
        self.next_request = time.monotonic() + 60 * credit_cost / max(1, int(self.limit * 0.9))
        try:
            response = self.client.get(path, params=params or {})
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("Twelve Data connection or JSON response failed") from exc
        if response.status_code >= 400 or body.get("status") == "error":
            code = int(body.get("code", response.status_code))
            message = str(body.get("message", "Provider request failed")).replace(self.key, "[REDACTED]")[:300]
            return {"error_code": code, "error": message}
        if path == "/api_usage" and body.get("plan_limit"):
            configure(body["plan_limit"])
        return body


def catalogue(client):
    records = []
    counts = {}
    for endpoint in ("stocks", "etfs"):
        cache = MARKET / (endpoint + "_catalog.json")
        if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
            body = json.loads(cache.read_text(encoding="utf-8"))
        else:
            body = client.request("/" + endpoint, {"country": "United States", "include_delisted": "false"})
            if "error_code" in body:
                if endpoint == "etfs":
                    counts["etf_catalog_error"] = body["error"]
                    continue
                raise RuntimeError(body["error"])
            atomic_json(cache, body)
        data = body.get("data", [])
        advertised = body.get("count", len(data))
        if advertised > len(data):
            raise RuntimeError(f"{endpoint} catalogue is truncated; expected {advertised}, received {len(data)}")
        counts[endpoint] = len(data)
        for row in data:
            if row.get("country") not in ("United States", "US"):
                continue
            identity = "|".join(str(row.get(k, "")) for k in ("symbol", "exchange", "mic_code"))
            item = {**row, "id": hashlib.sha256(identity.encode()).hexdigest()[:24]}
            item["type"] = row.get("type", "ETF" if endpoint == "etfs" else "Stock")
            records.append(item)
    unique = {row["id"]: row for row in records}
    return list(unique.values()), counts


def collect(years=10):
    MARKET.mkdir(parents=True, exist_ok=True)
    old = collection_state()
    if old.get("status") in ("running", "waiting"):
        return
    client = TwelveClient()
    usage = client.request("/api_usage")
    if "error_code" in usage:
        raise RuntimeError(usage["error"])
    client.limit = int(usage.get("plan_limit", 55))
    if old.get("snapshot_id") and old.get("years") == years and old.get("status") not in ("completed",):
        end = old["end_date"]
        start = old["start_date"]
        snapshot_id = old["snapshot_id"]
    else:
        end = datetime.now(timezone.utc).date().isoformat()
        start = (datetime.fromisoformat(end) - timedelta(days=round(years * 365.25))).date().isoformat()
        snapshot_id = hashlib.sha256((start + "|" + end + "|all|US").encode()).hexdigest()[:24]
    state = {
        "status": "running",
        "pid": os.getpid(),
        "process_created": psutil.Process().create_time(),
        "snapshot_id": snapshot_id,
        "years": years,
        "start_date": start,
        "end_date": end,
        "plan": usage.get("plan_category"),
        "credits_per_minute": client.limit,
        "paced_requests_per_minute": max(1, int(client.limit * 0.9)),
        "stage": "Loading full US stock and ETF catalogues",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "auto_analyze": True,
        "auto_resume": True,
        "completed": 0,
        "unavailable": 0,
        "pending": 0,
        "total": 0,
    }
    atomic_json(STATUS, state)
    records, counts = catalogue(client)
    db, directory = database(snapshot_id)
    priority = {ticker: i for i, ticker in enumerate(UNIVERSE)}
    for row in records:
        db.execute(
            """INSERT OR IGNORE INTO instruments(id,symbol,name,exchange,mic,type,currency,status,priority)
           VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                row["id"],
                row["symbol"],
                row.get("name", ""),
                row.get("exchange", ""),
                row.get("mic_code", ""),
                row["type"],
                row.get("currency", ""),
                "pending" if row.get("currency") == "USD" else "non_usd",
                priority.get(row["symbol"], 1000) + (20000 if row.get("exchange") == "OTC" else 0),
            ),
        )
    db.commit()
    if stock_scope():
        placeholders = ",".join("?" for _ in BENCHMARK_SYMBOLS)
        db.execute(
            "UPDATE instruments SET status='deferred_scope' WHERE type='ETF' AND status='pending' AND symbol NOT IN ("
            + placeholders
            + ")",
            list(BENCHMARK_SYMBOLS),
        )
        db.commit()
        state["scope"] = "US stocks; existing ETF cache and required market references retained"
    prioritize_queue(db)
    state["catalog_counts"] = counts
    state["coverage_note"] = (
        "Current provider US catalogue, including OTC and ETFs. Excludes provider-delisted catalogue entries; not a historical point-in-time universe."
    )

    def publish(stage=None):
        totals = {row[0]: row[1] for row in db.execute("SELECT status,count(*) FROM instruments GROUP BY status")}
        state.update(
            total=sum(totals.values()),
            completed=totals.get("ok", 0),
            pending=totals.get("pending", 0),
            unavailable=sum(v for k, v in totals.items() if k not in ("ok", "pending")),
            counts=totals,
            updated_utc=datetime.now(timezone.utc).isoformat(),
        )
        state["estimated_minutes_remaining"] = state["pending"] / max(1, int(client.limit * 0.9))
        if stage:
            state["stage"] = stage
        atomic_json(STATUS, state)

    publish(f"Downloading {years}-year adjusted daily histories; completed files are reusable")
    while True:
        if psutil.virtual_memory().available < 24 * GIB:
            state["status"] = "waiting"
            publish("Waiting for the 24 GiB OS memory reserve")
            time.sleep(5)
            continue
        if psutil.Process().memory_info().rss > 2 * GIB:
            state["status"] = "paused"
            publish("Collector reached its 2 GiB process memory ceiling")
            return
        row = db.execute(
            "SELECT * FROM instruments WHERE status='pending' ORDER BY priority,symbol,id LIMIT 1"
        ).fetchone()
        if row is None:
            break
        if db.execute("SELECT coalesce(sum(bytes),0) FROM instruments").fetchone()[0] > 16 * GIB:
            state["status"] = "paused"
            publish("Historical cache reached its 16 GiB ceiling")
            return
        params = {
            "symbol": row["symbol"],
            "interval": "1day",
            "outputsize": 5000,
            "order": "asc",
            "start_date": start,
            "end_date": end,
            "adjust": "all",
            "country": "United States",
        }
        if row["mic"]:
            params["mic_code"] = row["mic"]
        elif row["exchange"]:
            params["exchange"] = row["exchange"]
        state["last_symbol"] = row["symbol"]
        state["status"] = "running"
        try:
            body = client.request("/time_series", params)
        except RuntimeError:
            db.execute("UPDATE instruments SET attempts=attempts+1 WHERE id=?", (row["id"],))
            if row["attempts"] >= 2:
                db.execute(
                    "UPDATE instruments SET status='transient_failed',error=? WHERE id=?",
                    ("Provider connection failed on three attempts", row["id"]),
                )
            db.commit()
            state["status"] = "waiting"
            publish("Provider connection failed; retrying this symbol after 60 seconds")
            time.sleep(60)
            continue
        code = body.get("error_code")
        if code == 429:
            state["status"] = "waiting"
            publish("Provider quota reached; resuming when the quota resets")
            daily = usage.get("plan_daily_limit")
            if daily:
                tomorrow = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                    days=1
                )
                time.sleep(max(65, (tomorrow - datetime.now(timezone.utc)).total_seconds() + 5))
            else:
                time.sleep(65)
            continue
        if code == 401:
            state.update(status="blocked", auto_resume=False)
            publish("Twelve Data rejected the API key; update the private key and resume")
            return
        if code:
            db.execute("UPDATE instruments SET status='unavailable',error=? WHERE id=?", (body["error"], row["id"]))
        else:
            meta = body.get("meta", {})
            if meta.get("currency") != "USD" or meta.get("symbol") != row["symbol"]:
                db.execute(
                    "UPDATE instruments SET status='unavailable',error=? WHERE id=?",
                    ("Provider symbol/currency metadata did not match request", row["id"]),
                )
                db.commit()
                publish()
                continue
            parsed = []
            for bar in body.get("values", []):
                try:
                    date = bar["datetime"][:10]
                    close = float(bar["close"])
                    volume = float(bar.get("volume", 0) or 0)
                    if start <= date < end and np.isfinite(close) and close > 0:
                        extra = []
                        for column in ("open", "high", "low"):
                            value = float(bar[column]) if bar.get(column) is not None else np.nan
                            extra.append(value if np.isfinite(value) and value > 0 else np.nan)
                        parsed.append((date, close, volume, *extra))
                except (ValueError, KeyError, TypeError):
                    continue
            parsed = sorted({p[0]: p for p in parsed}.values())
            if not parsed:
                db.execute(
                    "UPDATE instruments SET status='unavailable',error=? WHERE id=?",
                    ("No usable adjusted daily bars", row["id"]),
                )
            else:
                path = directory / (row["id"] + ".npz")
                np.savez_compressed(
                    path,
                    dates=np.array([p[0] for p in parsed], dtype="U10"),
                    close=np.array([p[1] for p in parsed]),
                    volume=np.array([p[2] for p in parsed]),
                    open=np.array([p[3] for p in parsed]),
                    high=np.array([p[4] for p in parsed]),
                    low=np.array([p[5] for p in parsed]),
                )
                turnover = float(np.median([p[1] * p[2] for p in parsed[-60:]]))
                db.execute(
                    """UPDATE instruments SET status='ok',bars=?,first_date=?,last_date=?,
                    median_turnover=?,bytes=?,error='' WHERE id=?""",
                    (len(parsed), parsed[0][0], parsed[-1][0], turnover, path.stat().st_size, row["id"]),
                )
        db.commit()
        publish(f"Downloading {years}-year adjusted daily histories; completed files are reusable")
    state.update(status="completed", auto_resume=False)
    publish("Full catalogue queue finished; unavailable histories remain explicitly excluded")
    db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()
    try:
        with collection_lease() as acquired:
            if acquired:
                collect(args.years)
    except Exception as exc:
        state = collection_state()
        state.update(status="failed", stage=str(exc)[:400])
        atomic_json(STATUS, state)
        raise SystemExit(1) from None
