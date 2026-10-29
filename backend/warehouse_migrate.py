"""Resumable conversion of the existing cache; raw inputs remain intact."""

import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import psutil

from backend.market_data import GIB, atomic_json, collection_lease, collection_state, database
from backend.warehouse import WAREHOUSE, catalog, instrument_id, register, store_series_batch


def migrate_once(state=None):
    state = state or {}
    current = collection_state()
    if not current.get("snapshot_id"):
        return state
    source_db, directory = database(current["snapshot_id"])
    try:
        rows = [dict(r) for r in source_db.execute("SELECT * FROM instruments WHERE status='ok'")]
    finally:
        source_db.close()
    with catalog() as db:
        existing = {r[0] for r in db.execute("SELECT source_key FROM sources")}
        db.execute(
            "UPDATE instruments SET country='United States' WHERE country='' AND id IN (SELECT instrument_id FROM legacy_map WHERE snapshot=?)",
            (current["snapshot_id"],),
        )
    pending = []
    for row in rows:
        key = "us_npz|" + current["snapshot_id"] + "|" + row["id"]
        if key in existing:
            continue
        asset = "etf" if row["type"] == "ETF" else "equity"
        row.update(
            country="United States",
            asset_class=asset,
            source_key=key,
            canonical_id=instrument_id(asset, row["symbol"], row["exchange"], row["mic"]),
        )
        pending.append(row)
    pending.sort(key=lambda r: (r["asset_class"], r["canonical_id"]))
    state.update(
        total_cached=len(rows), pending=len(pending), stage="Converting immutable cache files to partitioned Parquet"
    )
    for start in range(0, len(pending), 256):
        if shutil.disk_usage(WAREHOUSE).free < 10 * GIB:
            raise ValueError("Conversion paused: fewer than 10 GiB remain on the storage drive")
        if psutil.virtual_memory().available < 24 * GIB or psutil.Process().memory_info().rss > 4 * GIB:
            raise ValueError("Conversion reached its 4 GiB process or 24 GiB OS reserve guard")
        batch = []
        with catalog() as db:
            for row in pending[start : start + 256]:
                path = directory / (row["id"] + ".npz")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                identity = register(db, row, row["asset_class"])
                db.execute(
                    "INSERT OR REPLACE INTO legacy_map VALUES(?,?,?)", (current["snapshot_id"], row["id"], identity)
                )
                with np.load(path, allow_pickle=False) as raw:
                    item = {name: raw[name].copy() for name in raw.files}
                item.update(
                    id=identity,
                    asset_class=row["asset_class"],
                    source_key=row["source_key"],
                    sha256=digest,
                    quality=int(
                        all(name in item and np.isfinite(item[name]).all() for name in ("open", "high", "low"))
                    ),
                )
                batch.append(item)
        count = store_series_batch(batch)
        state["converted_sources"] = state.get("converted_sources", 0) + len(batch)
        state["converted_rows"] = state.get("converted_rows", 0) + count
        state["pending"] = max(0, len(pending) - start - len(batch))
        state["updated"] = time.time()
        atomic_json(WAREHOUSE / "migration.json", state)
    state.update(stage="Cache is synchronized; watching for new completed histories", pending=0)
    atomic_json(WAREHOUSE / "migration.json", state)
    return state


def main(watch=False):
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    with collection_lease(WAREHOUSE / "migration.lock") as acquired:
        if not acquired:
            return
        state = {
            "status": "running",
            "pid": os.getpid(),
            "process_created": psutil.Process().create_time(),
            "converted_sources": 0,
            "converted_rows": 0,
        }
        atomic_json(WAREHOUSE / "migration.json", state)
        while True:
            state = migrate_once(state)
            if not watch:
                break
            time.sleep(30)
        state.update(status="completed")
        atomic_json(WAREHOUSE / "migration.json", state)


if __name__ == "__main__":
    try:
        main("--watch" in sys.argv)
    except Exception as exc:
        path = WAREHOUSE / "migration.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(status="failed", stage=str(exc)[:400])
        atomic_json(path, state)
        raise SystemExit(1) from None
