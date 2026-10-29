"""Bounded price browsing and Parquet exports; no arbitrary SQL endpoint."""

import uuid
from datetime import date
from typing import Annotated, Literal

import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import Field

from backend.warehouse import WAREHOUSE, catalog, info, query_prices, search
from backend.warehouse_jobs import migration_state, start_migration, stop_migration

router = APIRouter(prefix="/api/warehouse", tags=["Stock price warehouse"])
Identity = Annotated[str, Field(pattern=r"^[a-f0-9]{24}$")]


@router.get("")
def overview():
    return {**info(), "migration": migration_state()}


@router.get("/instruments")
def instruments(q: str = "", asset_class: Literal["equity", "etf", ""] = "equity", limit: int = 50):
    return search(q[:100], asset_class, min(limit, 100))


def read_prices(identity, start, end):
    with catalog() as db:
        item = db.execute("SELECT * FROM instruments WHERE id=?", (identity,)).fetchone()
    if not item:
        raise HTTPException(404, "Instrument is not stored in the local warehouse")
    if start and end and start > end:
        raise HTTPException(400, "Start date must precede end date")
    return dict(item), query_prices([identity], start, end, limit=10000)


@router.get("/prices/{identity}")
def prices(identity: Identity, start: date | None = None, end: date | None = None):
    import time

    clock = time.perf_counter()
    item, table = read_prices(identity, start, end)
    item.pop("metadata", None)
    return {
        "instrument": item,
        "bars": table.to_pylist(),
        "query_seconds": time.perf_counter() - clock,
        "note": "Provider-adjusted daily prices. NULL OHLC fields indicate a legacy close-only source. This is not a historical fill record.",
    }


@router.get("/prices/{identity}/parquet")
def export_prices(identity: Identity, start: date | None = None, end: date | None = None):
    item, table = read_prices(identity, start, end)
    directory = WAREHOUSE / "exports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + ".parquet")
    pq.write_table(table, path, compression="zstd")
    # Keep a bounded set of small browser exports; the canonical lake is never pruned here.
    files = sorted(directory.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    for old in files[:-50]:
        old.unlink(missing_ok=True)
    return FileResponse(
        path, filename=f"quantforge-{item['symbol']}-daily.parquet", media_type="application/vnd.apache.parquet"
    )


@router.post("/sync")
def start_sync():
    return start_migration()


@router.post("/pause")
def stop_sync():
    return stop_migration()
