"""Immutable Zstandard Parquet price lake with SQLite manifests and DuckDB scans."""

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend.discovery_data import DATA

WAREHOUSE = DATA / "warehouse"
SCHEMA = pa.schema(
    [
        ("instrument_id", pa.string()),
        ("date", pa.date32()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("adjustment", pa.string()),
        ("source_hash", pa.string()),
        ("source_key", pa.string()),
        ("quality", pa.int8()),
        ("ingested_at", pa.timestamp("us", tz="UTC")),
        ("version_id", pa.string()),
    ]
)


@contextmanager
def catalog():
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(WAREHOUSE / "manifest.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript("""PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS instruments(id TEXT PRIMARY KEY,asset_class TEXT,symbol TEXT,exchange TEXT,mic TEXT,currency TEXT,country TEXT,instrument_type TEXT,metadata TEXT);
      CREATE INDEX IF NOT EXISTS instrument_search ON instruments(asset_class,symbol,country);
      CREATE TABLE IF NOT EXISTS price_files(id TEXT PRIMARY KEY,path TEXT,asset_class TEXT,bucket TEXT,year INTEGER,rows INTEGER,bytes INTEGER,sha256 TEXT,created TEXT);
      CREATE TABLE IF NOT EXISTS file_instruments(file_id TEXT,instrument_id TEXT,PRIMARY KEY(file_id,instrument_id));
      CREATE INDEX IF NOT EXISTS instrument_files ON file_instruments(instrument_id,file_id);
      CREATE TABLE IF NOT EXISTS sources(source_key TEXT PRIMARY KEY,source_hash TEXT,instrument_id TEXT,rows INTEGER,quality INTEGER,first_date TEXT,last_date TEXT,created TEXT);
      CREATE INDEX IF NOT EXISTS sources_by_instrument ON sources(instrument_id,source_hash);
      CREATE TABLE IF NOT EXISTS legacy_map(snapshot TEXT,legacy_id TEXT,instrument_id TEXT,PRIMARY KEY(snapshot,legacy_id));
      CREATE TABLE IF NOT EXISTS downloads(id TEXT PRIMARY KEY,asset_class TEXT,priority INTEGER DEFAULT 100,status TEXT DEFAULT 'pending',attempts INTEGER DEFAULT 0,error TEXT);
      CREATE TABLE IF NOT EXISTS fundamental_access(endpoint TEXT PRIMARY KEY,cost INTEGER,status TEXT,note TEXT,checked TEXT);
      CREATE TABLE IF NOT EXISTS fundamental_files(id TEXT PRIMARY KEY,instrument_id TEXT,endpoint TEXT,path TEXT,observed_at TEXT,sha256 TEXT);
    """)
    try:
        with db:
            yield db
    finally:
        db.close()


def instrument_id(asset_class, symbol, exchange="", mic=""):
    return hashlib.sha256("|".join(["TD", asset_class, symbol, exchange or "", mic or ""]).encode()).hexdigest()[:24]


def register(db, row, asset_class):
    symbol = row["symbol"]
    exchange = row.get("exchange", "") or ""
    mic = row.get("mic") or row.get("mic_code") or ""
    identity = instrument_id(asset_class, symbol, exchange, mic)
    currency = row.get("currency") or row.get("currency_quote") or row.get("quote_currency") or ""
    db.execute(
        "INSERT INTO instruments VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET metadata=excluded.metadata",
        (
            identity,
            asset_class,
            symbol,
            exchange,
            mic,
            currency,
            row.get("country", ""),
            row.get("type", asset_class),
            json.dumps(row),
        ),
    )
    return identity


def has_source(db, key, digest):
    row = db.execute("SELECT source_hash FROM sources WHERE source_key=?", (key,)).fetchone()
    return row is not None and row[0] == digest


def store_series_batch(items):
    """Commit immutable files before publishing a manifest transaction. No readers scan uncommitted globs."""
    if not items:
        return 0
    groups = {}
    version = uuid.uuid4().hex
    observed = datetime.now(timezone.utc)
    for item in items:
        dates = np.asarray(item["dates"], dtype="datetime64[D]")
        if len(dates) != len(item["close"]):
            raise ValueError("Price column length mismatch")
        if len(np.unique(dates)) != len(dates):
            raise ValueError("Duplicate observations in one source series")
        if not np.isfinite(item["close"]).all() or (np.asarray(item["close"]) <= 0).any():
            raise ValueError("Close prices must be finite and positive")
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        for year in np.unique(years):
            mask = years == year
            n = int(mask.sum())
            columns = {
                "instrument_id": pa.array([item["id"]] * n),
                "date": pa.array(dates[mask].astype("datetime64[ms]"), type=pa.date32()),
                "adjustment": pa.array([item.get("adjustment", "all_provider")] * n),
                "source_hash": pa.array([item["sha256"]] * n),
                "source_key": pa.array([item["source_key"]] * n),
                "quality": pa.array([item.get("quality", 0)] * n, type=pa.int8()),
                "ingested_at": pa.array([observed] * n, type=SCHEMA.field("ingested_at").type),
                "version_id": pa.array([version] * n),
            }
            for name in ["open", "high", "low", "close", "volume"]:
                values = np.asarray(item.get(name, np.full(len(dates), np.nan)), dtype=float)[mask]
                columns[name] = pa.array(values, mask=~np.isfinite(values), type=pa.float64())
            key = (item["asset_class"], item["id"][0], int(year))
            groups.setdefault(key, []).append(pa.Table.from_pydict(columns, schema=SCHEMA))
    files = []
    for (asset, bucket, year), tables in groups.items():
        table = pa.concat_tables(tables).sort_by([("instrument_id", "ascending"), ("date", "ascending")])
        identity = uuid.uuid4().hex
        directory = WAREHOUSE / "prices" / f"asset_class={asset}" / f"bucket={bucket}" / f"year={year}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (identity + ".parquet")
        temp = directory / (identity + ".partial")
        pq.write_table(
            table, temp, compression="zstd", compression_level=3, row_group_size=65536, write_statistics=True
        )
        temp.replace(path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        files.append(
            (
                identity,
                path,
                asset,
                bucket,
                year,
                table.num_rows,
                path.stat().st_size,
                digest,
                set(table.column("instrument_id").to_pylist()),
            )
        )
    with catalog() as db:
        for identity, path, asset, bucket, year, rows, size, digest, ids in files:
            db.execute(
                "INSERT INTO price_files VALUES(?,?,?,?,?,?,?,?,?)",
                (identity, str(path), asset, bucket, year, rows, size, digest, observed.isoformat()),
            )
            db.executemany("INSERT INTO file_instruments VALUES(?,?)", [(identity, i) for i in ids])
        for item in items:
            dates = item["dates"]
            db.execute(
                "INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?,?)",
                (
                    item["source_key"],
                    item["sha256"],
                    item["id"],
                    len(dates),
                    item.get("quality", 0),
                    str(min(dates)),
                    str(max(dates)),
                    observed.isoformat(),
                ),
            )
    return sum(f[5] for f in files)


def _files(identities, start=None, end=None):
    if not identities:
        return []
    with catalog() as db:
        found = {}
        for begin in range(0, len(identities), 500):
            batch = identities[begin : begin + 500]
            sql = (
                "SELECT DISTINCT f.* FROM price_files f JOIN file_instruments m ON m.file_id=f.id WHERE m.instrument_id IN ("
                + ",".join("?" for _ in batch)
                + ")"
            )
            params = list(batch)
            if start:
                sql += " AND f.year>=?"
                params.append(int(str(start)[:4]))
            if end:
                sql += " AND f.year<=?"
                params.append(int(str(end)[:4]))
            for row in db.execute(sql, params):
                found[row["id"]] = dict(row)
    return list(found.values())


def query_prices(identities, start=None, end=None, columns=None, limit=100000, source_hashes=None, pinned_files=None):
    columns = columns or ["instrument_id", "date", "open", "high", "low", "close", "volume", "adjustment", "quality"]
    allowed = set(SCHEMA.names)
    if not set(columns) <= allowed:
        raise ValueError("Unknown price column")
    paths = [row["path"] for row in _files(identities, start, end) if pinned_files is None or row["id"] in pinned_files]
    if not paths:
        return pa.Table.from_batches([], schema=pa.schema([SCHEMA.field(c) for c in columns]))
    sql = (
        "SELECT "
        + ",".join('"' + c + '"' for c in columns)
        + " FROM read_parquet(?,hive_partitioning=false,union_by_name=true) WHERE instrument_id IN (SELECT unnest(?))"
    )
    params = [paths, list(identities)]
    if start:
        sql += " AND date>=CAST(? AS DATE)"
        params.append(str(start))
    if end:
        sql += " AND date<=CAST(? AS DATE)"
        params.append(str(end))
    if source_hashes:
        sql += " AND source_hash IN (SELECT unnest(?))"
        params.append(list(source_hashes))
    sql += " QUALIFY ROW_NUMBER() OVER (PARTITION BY instrument_id,date,adjustment ORDER BY quality DESC,ingested_at DESC,version_id DESC)=1 ORDER BY instrument_id,date LIMIT ?"
    params.append(int(limit))
    connection = duckdb.connect(config={"threads": 8, "memory_limit": "4GB"})
    try:
        return connection.execute(sql, params).fetch_arrow_table()
    finally:
        connection.close()


def search(q="", asset_class="", limit=50, country=""):
    clauses = ["(symbol LIKE ? OR metadata LIKE ?)"]
    params = ["%" + q.upper() + "%", "%" + q + "%"]
    if asset_class:
        clauses.append("i.asset_class=?")
        params.append(asset_class)
    if country:
        clauses.append("i.country=?")
        params.append(country)
    with catalog() as db:
        rows = db.execute(
            "SELECT i.*,d.status AS download_status,d.error FROM instruments i LEFT JOIN downloads d ON d.id=i.id WHERE "
            + " AND ".join(clauses)
            + " ORDER BY symbol LIMIT ?",
            params + [min(500, max(1, limit))],
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            meta = json.loads(item.pop("metadata"))
            item["name"] = meta.get("name") or meta.get("currency_base") or item["symbol"]
            coverage = db.execute(
                "SELECT MAX(rows),MAX(quality),MIN(first_date),MAX(last_date) FROM sources WHERE instrument_id=?",
                (item["id"],),
            ).fetchone()
            item.update(bars=int(coverage[0] or 0), ohlcv=coverage[1] == 1, first=coverage[2], last=coverage[3])
            result.append(item)
        return result


def info():
    with catalog() as db:
        classes = [
            dict(row)
            for row in db.execute("SELECT asset_class,COUNT(*) AS instruments FROM instruments GROUP BY asset_class")
        ]
        rows, size, files = db.execute(
            "SELECT COALESCE(SUM(rows),0),COALESCE(SUM(bytes),0),COUNT(*) FROM price_files"
        ).fetchone()
        states = [
            dict(row)
            for row in db.execute(
                "SELECT asset_class,status,COUNT(*) AS count FROM downloads GROUP BY asset_class,status"
            )
        ]
        sources = db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        fundamental = [dict(row) for row in db.execute("SELECT * FROM fundamental_access")]
        rows_by_class = [
            dict(row)
            for row in db.execute(
                "SELECT asset_class,SUM(rows) AS rows,SUM(bytes) AS bytes FROM price_files GROUP BY asset_class"
            )
        ]
    return {
        "asset_classes": classes,
        "stored_rows_including_versions": rows,
        "compressed_bytes": size,
        "parquet_files": files,
        "source_series": sources,
        "download_states": states,
        "fundamental_access": fundamental,
        "rows_by_class": rows_by_class,
        "format": "Zstandard Parquet / DuckDB / transactional SQLite manifests",
        "notes": [
            "Collection is scoped to US stocks. Previously cached ETF/fund reference data are retained in a separate asset-class partition.",
            "Legacy cached histories have close/volume only. Missing OHLC is NULL, never manufactured. New histories retain provider OHLCV.",
            "The latest-price view resolves versions by OHLCV quality then ingestion time; frozen research selects its original source hashes.",
        ],
    }


def frozen_panel(body):
    """Accelerate unchanged legacy manifests; unconverted sources retain the verified NPZ fallback."""
    if body.get("source") == "parquet_warehouse":
        ready = [(i, r["id"], None) for i, r in enumerate(body["records"])]
        files = body["parquet_files"]
        pinned = {r["id"] for r in files}
        return _panel_arrays(body, ready, files, pinned)
    snapshot = body.get("snapshot_id")
    if not snapshot:
        return None
    with catalog() as db:
        mapping = {
            r["legacy_id"]: r["instrument_id"]
            for r in db.execute("SELECT * FROM legacy_map WHERE snapshot=?", (snapshot,))
        }
        ready = []
        for offset, row in enumerate(body["records"]):
            identity = mapping.get(row["id"])
            if (
                identity
                and db.execute(
                    "SELECT 1 FROM sources WHERE instrument_id=? AND source_hash=?", (identity, row["sha256"])
                ).fetchone()
            ):
                ready.append((offset, identity, row["sha256"]))
    if not ready:
        return None
    files = _files([r[1] for r in ready], body["start"], body["end"])
    return _panel_arrays(body, ready, files)


def _panel_arrays(body, ready, files, pinned=None):
    for row in files:
        with open(row["path"], "rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != row["sha256"]:
                raise ValueError("A Parquet file changed after publication")
    dates = np.array(body["dates"], dtype="datetime64[D]")
    logs = np.full((body["assets"], len(dates)), np.nan)
    volumes = np.full_like(logs, np.nan)
    positions = {identity: offset for offset, identity, _ in ready}
    ready.sort(key=lambda r: r[1])
    for start in range(0, len(ready), 512):
        batch = ready[start : start + 512]
        table = query_prices(
            [r[1] for r in batch],
            body["start"],
            body["end"],
            ["instrument_id", "date", "close", "volume"],
            limit=len(batch) * 10000,
            source_hashes=[r[2] for r in batch] if pinned is None else None,
            pinned_files=pinned,
        )
        if not table.num_rows:
            continue
        encoded = table["instrument_id"].combine_chunks().dictionary_encode()
        row_map = np.array([positions[i] for i in encoded.dictionary.to_pylist()])
        rows = row_map[encoded.indices.to_numpy()]
        observed = table["date"].to_numpy().astype("datetime64[D]")
        columns = np.searchsorted(dates, observed)
        mask = columns < len(dates)
        mask[mask] &= dates[columns[mask]] == observed[mask]
        logs[rows[mask], columns[mask]] = np.log(table["close"].to_numpy()[mask])
        volumes[rows[mask], columns[mask]] = table["volume"].to_numpy()[mask]
    return logs, volumes, {offset for offset, _, _ in ready}


def capture_snapshot(name, security_csv="", membership_csv=""):
    from backend.lab_data import _finish, metadata_csv
    from backend.lab_store import now
    from backend.market_data import BENCHMARK_SYMBOLS, collection_state

    state = collection_state()
    with catalog() as db:
        published = db.execute(
            "SELECT COUNT(*) FROM legacy_map m JOIN sources s ON s.source_key=('us_npz|' || m.snapshot || '|' || m.legacy_id) WHERE m.snapshot=?",
            (state.get("snapshot_id", ""),),
        ).fetchone()[0]
        records = []
        for row in db.execute(
            "SELECT DISTINCT i.* FROM instruments i JOIN legacy_map l ON l.instrument_id=i.id WHERE l.snapshot=? AND i.currency='USD' AND EXISTS (SELECT 1 FROM sources s WHERE s.instrument_id=i.id)",
            (state.get("snapshot_id", ""),),
        ):
            if row["asset_class"] != "equity" and row["symbol"] not in BENCHMARK_SYMBOLS:
                continue
            metadata = json.loads(row["metadata"])
            records.append(
                {
                    **metadata,
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "type": row["instrument_type"],
                    "exchange": row["exchange"],
                    "mic": row["mic"],
                    "currency": row["currency"],
                }
            )
    if len(records) < 10 or not any(r["symbol"] == "SPY" for r in records):
        return None
    records.sort(key=lambda r: r["id"])
    files = _files([r["id"] for r in records], state["start_date"], state["end_date"])
    pinned = {r["id"] for r in files}
    spy = next(r for r in records if r["symbol"] == "SPY")
    table = query_prices([spy["id"]], state["start_date"], state["end_date"], ["date"], 10000, pinned_files=pinned)
    dates = [str(day) for day in table["date"].to_pylist() if day.isoformat() < state["end_date"] and day.weekday() < 5]
    digest = hashlib.sha256(
        (
            json.dumps([(r["id"], r["sha256"]) for r in sorted(files, key=lambda r: r["id"])])
            + json.dumps(records, sort_keys=True)
            + json.dumps(dates)
            + security_csv
            + membership_csv
        ).encode()
    ).hexdigest()
    return _finish(
        {
            "name": name,
            "source": "parquet_warehouse",
            "created": now(),
            "snapshot_id": state["snapshot_id"],
            "assets": len(records),
            "sessions": len(dates),
            "start": dates[0],
            "end": dates[-1],
            "dates": dates,
            "records": records,
            "parquet_files": files,
            "security": metadata_csv(security_csv, "security"),
            "membership": metadata_csv(membership_csv, "membership"),
            "input_sha256": digest,
            "coverage": "configured_stock_scope_complete"
            if state.get("status") == "completed" and published >= state.get("completed", 0)
            else "converted_cache_subset",
            "membership_supplied": bool(membership_csv),
            "security_metadata_supplied": bool(security_csv),
            "notes": [
                "Immutable Parquet file manifest; future downloads and price revisions cannot alter this snapshot.",
                "US stock histories plus required market-reference ETFs. Current provider catalogue is not a certified historical universe.",
                "Only converted, committed histories are included. Unconverted downloads appear in a later snapshot.",
            ],
        },
        digest[:24],
    )
