"""Immutable input manifests and explicit data quality; never fill inference prices."""

import csv
import hashlib
import io
import json
import re
from datetime import date

import numpy as np
import pandas as pd

from backend import lab_store
from backend.market_analysis import calendar_positions
from backend.market_data import atomic_json, collection_state, database

SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,23}$")


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def metadata_csv(text, kind):
    if not text.strip():
        return []
    frame = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
    required = {"symbol", "start", "end"} if kind == "membership" else {"symbol", "security_id", "known_at"}
    if not required <= set(frame.columns):
        raise ValueError(f"{kind} CSV needs columns: {', '.join(sorted(required))}")
    if len(frame) > 100000:
        raise ValueError("Metadata CSV is limited to 100,000 records")
    rows = frame.to_dict("records")
    for row in rows:
        row["symbol"] = row["symbol"].strip().upper()
        if not SYMBOL.fullmatch(row["symbol"]):
            raise ValueError("Metadata contains an invalid symbol")
        for key in ("known_at", "start", "end", "valid_from", "valid_to"):
            if row.get(key):
                date.fromisoformat(row[key])
        if kind == "membership" and row["end"] and row["end"] < row["start"]:
            raise ValueError("Membership end must follow its start")
        for key in ("borrow_rate", "spread_bps"):
            if row.get(key):
                value = float(row[key])
                limit = 5 if key == "borrow_rate" else 1000
                if not np.isfinite(value) or not 0 <= value <= limit:
                    raise ValueError(f"Invalid {key} in security metadata")
        if row.get("borrow_available", "").lower() not in ("", "true", "false"):
            raise ValueError("borrow_available must be true, false or blank")
    return rows


def _finish(manifest, identity):
    directory = lab_store.LAB / "datasets" / identity
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        return lab_store.dataset(identity)
    manifest["id"] = identity
    atomic_json(directory / "manifest.json", manifest)
    body = {
        k: v for k, v in manifest.items() if k not in ("records", "dates", "security", "membership", "parquet_files")
    }
    return lab_store.put_dataset(body)


def capture_cache(name="Twelve Data cached universe", security_csv="", membership_csv=""):
    from backend.warehouse import capture_snapshot

    parquet = capture_snapshot(name, security_csv, membership_csv)
    if parquet is not None:
        return parquet
    state = collection_state()
    if not state.get("snapshot_id"):
        raise ValueError("No provider cache exists yet")
    db, directory = database(state["snapshot_id"])
    try:
        rows = [dict(r) for r in db.execute("SELECT * FROM instruments WHERE status='ok' ORDER BY symbol,id")]
    finally:
        db.close()
    if len(rows) < 10:
        raise ValueError("Capture requires at least ten completed histories")
    digest = hashlib.sha256()
    for row in rows:
        row["path"] = str(directory / (row["id"] + ".npz"))
        row["sha256"] = hashlib.sha256((directory / (row["id"] + ".npz")).read_bytes()).hexdigest()
        digest.update((row["id"] + row["sha256"]).encode())
    digest.update((security_csv + "|" + membership_csv).encode())
    identity = digest.hexdigest()[:24]
    spy = [r for r in rows if r["symbol"] == "SPY"]
    if not spy:
        raise ValueError("SPY must be downloaded for the observed-session calendar")
    anchor = max(spy, key=lambda row: row["bars"])
    with np.load(anchor["path"], allow_pickle=False) as data:
        dates = [str(day) for day in data["dates"] if date.fromisoformat(str(day)).weekday() < 5]
    return _finish(
        {
            "name": name[:100],
            "source": "twelvedata_cache",
            "snapshot_id": state["snapshot_id"],
            "created": lab_store.now(),
            "assets": len(rows),
            "sessions": len(dates),
            "start": dates[0],
            "end": dates[-1],
            "dates": dates,
            "records": rows,
            "security": metadata_csv(security_csv, "security"),
            "membership": metadata_csv(membership_csv, "membership"),
            "input_sha256": digest.hexdigest(),
            "coverage": (
                "configured_stock_scope_complete"
                if state.get("counts", {}).get("deferred_scope")
                else "full_catalogue_attempted"
            )
            if state.get("status") == "completed"
            else "cached_subset",
            "membership_supplied": bool(membership_csv.strip()),
            "security_metadata_supplied": bool(security_csv.strip()),
            "notes": [
                "Current provider catalogue; historical delisted coverage is incomplete. A cached subset is priority ordered, not a random sample.",
                "Observed SPY weekday sessions are the alignment reference. Adjusted close data are provider supplied.",
                "Security aliases/share classes and price revisions require review. A price duplicate is not proof of identical legal securities.",
            ],
        },
        identity,
    )


def import_prices(name, text, adjusted_usd, security_csv="", membership_csv=""):
    if not adjusted_usd:
        raise ValueError("Confirm that imported close prices are split/dividend adjusted USD observations")
    security = metadata_csv(security_csv, "security")
    membership = metadata_csv(membership_csv, "membership")
    headers = next(csv.reader(io.StringIO(text)), [])
    if len(headers) != len(set(headers)):
        raise ValueError("Price CSV contains duplicate column names")
    frame = pd.read_csv(io.StringIO(text))
    if "Date" not in frame or frame.columns.duplicated().any():
        raise ValueError("Price CSV needs a unique Date column and unique symbol columns")
    dates = pd.to_datetime(frame.pop("Date"), errors="raise", format="%Y-%m-%d")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing")
    if len(dates) < 504 or len(dates) > 10000:
        raise ValueError("Import 504–10,000 observed daily sessions")
    if (dates.dt.day_of_week >= 5).any():
        raise ValueError("Daily US price imports must omit weekends")
    if (dates.dt.date >= date.today()).any():
        raise ValueError("Only completed dates before today can be imported")
    symbols = [col for col in frame if not col.endswith("_volume")]
    if not 10 <= len(symbols) <= 5000 or any(not SYMBOL.fullmatch(s) for s in symbols):
        raise ValueError("Import 10–5,000 US symbols; optional volume columns use SYMBOL_volume")
    if "SPY" not in symbols:
        raise ValueError("Include SPY for an explicit benchmark")
    prices = frame[symbols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float).T
    if np.isinf(prices).any() or (prices[np.isfinite(prices)] <= 0).any():
        raise ValueError("Observed close prices must be positive and finite; blanks may represent missing observations")
    volume = np.full_like(prices, np.nan)
    for i, symbol in enumerate(symbols):
        if symbol + "_volume" in frame:
            values = pd.to_numeric(frame[symbol + "_volume"], errors="raise").to_numpy(dtype=float)
            if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
                raise ValueError("Volumes must be nonnegative finite values or blanks")
            volume[i] = values
    digest = hashlib.sha256((text + "|" + security_csv + "|" + membership_csv).encode()).hexdigest()
    identity = digest[:24]
    directory = lab_store.LAB / "datasets" / identity
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        return lab_store.dataset(identity)
    np.save(directory / "logs.npy", np.log(prices))
    np.save(directory / "volumes.npy", volume)
    return _finish(
        {
            "name": name[:100],
            "source": "user_adjusted_csv",
            "array_hashes": {name: file_hash(directory / name) for name in ["logs.npy", "volumes.npy"]},
            "created": lab_store.now(),
            "assets": len(symbols),
            "sessions": len(dates),
            "start": dates.iloc[0].date().isoformat(),
            "end": dates.iloc[-1].date().isoformat(),
            "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
            "records": [
                {
                    "symbol": s,
                    "id": hashlib.sha256(s.encode()).hexdigest()[:24],
                    "type": "Common Stock",
                    "exchange": "User supplied",
                    "name": s,
                }
                for s in symbols
            ],
            "security": security,
            "membership": membership,
            "input_sha256": digest,
            "coverage": "user_supplied_unverified",
            "membership_supplied": bool(membership_csv.strip()),
            "security_metadata_supplied": bool(security_csv.strip()),
            "notes": [
                "User asserts adjusted USD prices; source, complete-universe coverage and corporate actions are not independently verified.",
                "Missing volume requires an explicit unknown-liquidity assumption. Never treat unknown volume as proven tradability.",
            ],
        },
        identity,
    )


def manifest(identity):
    lab_store.dataset(identity)
    return json.loads((lab_store.LAB / "datasets" / identity / "manifest.json").read_text(encoding="utf-8"))


def load_dataset(identity):
    body = manifest(identity)
    dates = np.array(body["dates"], dtype="U10")
    if body["source"] == "parquet_warehouse":
        from backend.warehouse import frozen_panel

        logs, volumes, _ = frozen_panel(body)
        return body, dates, logs, volumes
    if body["source"] == "user_adjusted_csv":
        directory = lab_store.LAB / "datasets" / identity
        for name, digest in body.get("array_hashes", {}).items():
            if file_hash(directory / name) != digest:
                raise ValueError("Imported price arrays changed since capture; reimport the source")
        return (
            body,
            dates,
            np.load(directory / "logs.npy", mmap_mode="r"),
            np.load(directory / "volumes.npy", mmap_mode="r"),
        )
    logs = np.full((body["assets"], len(dates)), np.nan)
    volumes = np.full_like(logs, np.nan)
    from backend.warehouse import frozen_panel

    accelerated = frozen_panel(body)
    converted = set()
    if accelerated is not None:
        logs, volumes, converted = accelerated
    for i, row in enumerate(body["records"]):
        if i in converted:
            continue
        from pathlib import Path

        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Cached input changed since capture: {row['symbol']}. Capture a new dataset.")
        with np.load(row["path"], allow_pickle=False) as data:
            positions, mask = calendar_positions(dates, data["dates"])
            logs[i, positions] = np.log(data["close"][mask])
            volumes[i, positions] = data["volume"][mask]
    return body, dates, logs, volumes


def asof_metadata(body, symbol, at_date):
    known = [
        row
        for row in body.get("security", [])
        if row["symbol"] == symbol
        and row["known_at"] <= at_date
        and (not row.get("valid_from") or row["valid_from"] <= at_date)
        and (not row.get("valid_to") or at_date <= row["valid_to"])
    ]
    return max(known, key=lambda row: row["known_at"]) if known else {}


def member_asof(body, symbol, at_date):
    if not body.get("membership"):
        return True
    return any(
        row["symbol"] == symbol
        and row["start"] <= at_date
        and (not row["end"] or at_date <= row["end"])
        and (row.get("known_at") or row["start"]) <= at_date
        for row in body["membership"]
    )


def quality(identity, limit=100):
    body, dates, logs, volume = load_dataset(identity)
    rows = []
    for i, record in enumerate(body["records"]):
        valid = np.isfinite(logs[i])
        indices = np.flatnonzero(valid)
        changes = np.diff(logs[i])
        observed = np.isfinite(changes)
        rows.append(
            {
                "symbol": record["symbol"],
                "exchange": record.get("exchange", ""),
                "coverage": float(valid.mean()),
                "observations": int(valid.sum()),
                "first": str(dates[indices[0]]) if len(indices) else None,
                "last": str(dates[indices[-1]]) if len(indices) else None,
                "large_daily_moves": int((np.abs(changes[observed]) > np.log(1.5)).sum()),
                "zero_return_fraction": float((np.abs(changes[observed]) < 1e-12).mean()) if observed.any() else None,
                "volume_known": bool(np.isfinite(volume[i]).any()),
            }
        )
    return {
        "dataset": {
            k: v for k, v in body.items() if k not in ("records", "dates", "security", "membership", "parquet_files")
        },
        "assets": rows[:limit],
        "assets_total": len(rows),
        "missing_fraction": float((~np.isfinite(logs)).mean()),
        "large_move_assets": sum(row["large_daily_moves"] > 0 for row in rows),
        "unknown_volume_assets": sum(not row["volume_known"] for row in rows),
    }
