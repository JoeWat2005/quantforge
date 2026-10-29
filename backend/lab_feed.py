"""A single low-rate, cancellable provider refresh for forward paper prices."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import date, timedelta

import numpy as np
import psutil

from backend import lab_store
from backend.engine import ROOT
from backend.lab_data import _finish
from backend.lab_paper import update_paper
from backend.market_data import GIB, TwelveClient, atomic_json, collection_state

LOCK = threading.RLock()


def feed_status():
    path = lab_store.LAB / "paper_refresh.json"
    launch_path = lab_store.LAB / "paper_launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
    body = json.loads(path.read_text()) if path.exists() else {"status": "idle"}
    if launch and body.get("run_id") != launch["run_id"]:
        body = launch
    if body.get("status") in ("starting", "running", "waiting"):
        try:
            process = psutil.Process(body["pid"])
            if abs(process.create_time() - body.get("process_created", 0)) > 0.1:
                body["status"] = "interrupted"
        except psutil.NoSuchProcess:
            body["status"] = "interrupted"
    return body


def refresh_rate(collection):
    total = int(collection.get("credits_per_minute") or 0)
    used = (
        int(collection.get("paced_requests_per_minute") or 0)
        if collection.get("status") in ("running", "waiting")
        else 0
    )
    available = total - used
    if available < 2:
        raise ValueError(
            "Insufficient spare provider quota. Pause the main collection or wait for it to finish before refreshing paper prices."
        )
    return available, max(1, int(available * 0.9))


def start_refresh(identity):
    with LOCK:
        state = feed_status()
        if state["status"] in ("starting", "running", "waiting"):
            raise ValueError("A paper refresh is already active; wait or cancel it first")
        lab_store.paper(identity)
        path = lab_store.LAB / "paper_feed.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["QF_LAB_DIR"] = str(lab_store.LAB)
        run_id = uuid.uuid4().hex[:16]
        with path.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "backend.lab_feed", identity, run_id],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        state = {
            "status": "starting",
            "run_id": run_id,
            "paper_id": identity,
            "pid": process.pid,
            "process_created": psutil.Process(process.pid).create_time(),
            "stage": "Starting forward-price refresh",
        }
        # Launch ownership is separate from child progress, avoiding a fast-child overwrite race.
        atomic_json(lab_store.LAB / "paper_launch.json", state)
        return state


def stop_refresh():
    with LOCK:
        state = feed_status()
        if state.get("pid") and state["status"] in ("starting", "running", "waiting"):
            try:
                process = psutil.Process(state["pid"])
                if (
                    abs(process.create_time() - state.get("process_created", 0)) < 0.1
                    and "backend.lab_feed" in process.cmdline()
                ):
                    for child in process.children(recursive=True):
                        child.kill()
                    process.kill()
                    psutil.wait_procs([process], timeout=2)
            except psutil.NoSuchProcess:
                pass
            state.update(status="cancelled", stage="Paper refresh cancelled; existing paper history is unchanged")
            atomic_json(lab_store.LAB / "paper_refresh.json", state)
        return state


def refresh(identity, run_id=None):
    paper = lab_store.paper(identity)
    state = {
        "status": "running",
        "run_id": run_id or uuid.uuid4().hex[:16],
        "paper_id": identity,
        "pid": os.getpid(),
        "process_created": psutil.Process().create_time(),
        "completed": 0,
        "stage": "Checking the forward interval",
    }

    def publish():
        atomic_json(lab_store.LAB / "paper_refresh.json", {**state, "updated": lab_store.now()})

    publish()
    if date.today() <= date.fromisoformat(paper["start_after"]) + timedelta(days=1):
        state.update(
            status="completed",
            stage="No completed post-creation trading dates can exist yet; no provider requests were made",
        )
        publish()
        return
    limit, pace = refresh_rate(collection_state())
    client = TwelveClient()
    client.limit = limit
    state["paced_requests_per_minute"] = pace
    holdings = {row["symbol"]: row for row in paper["holdings"]}
    holdings.setdefault("SPY", {"symbol": "SPY", "mic": "ARCX", "exchange": "NYSE"})
    state["total"] = len(holdings)
    directory = lab_store.LAB / "paper_feed" / uuid.uuid4().hex[:16]
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    calendar = None
    start = (date.fromisoformat(paper["start_after"]) - timedelta(days=7)).isoformat()
    end = date.today().isoformat()
    last_sent = 0
    for symbol, holding in holdings.items():
        client.limit, pace = refresh_rate(collection_state())
        client.next_request = max(client.next_request, last_sent + 60 / pace)
        state["paced_requests_per_minute"] = pace
        if psutil.virtual_memory().available < 24 * GIB or psutil.Process().memory_info().rss > GIB:
            raise ValueError("Paper refresh resource reserve reached; existing ledger remains unchanged")
        state.update(stage=f"Downloading forward adjusted closes · {symbol}", status="running")
        publish()
        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": 5000,
            "start_date": start,
            "end_date": end,
            "adjust": "all",
            "order": "asc",
            "country": "United States",
        }
        if holding.get("mic"):
            params["mic_code"] = holding["mic"]
        response = None
        for attempt in range(3):
            try:
                last_sent = max(time.monotonic(), client.next_request)
                response = client.request("/time_series", params)
            except RuntimeError:
                if attempt == 2:
                    raise
                state.update(status="waiting", stage="Provider connection failed; paper refresh retries in 60 seconds")
                publish()
                time.sleep(60)
                continue
            if response.get("error_code") == 429:
                state.update(status="waiting", stage="Provider quota reached; retrying after 65 seconds")
                publish()
                time.sleep(65)
                continue
            break
        if not response or response.get("error_code"):
            raise ValueError((response or {}).get("error", "Provider did not supply a usable response"))
        meta = response.get("meta", {})
        if meta.get("symbol") != symbol or meta.get("currency") != "USD":
            raise ValueError("Forward quote identity or currency mismatch")
        values = []
        for bar in response.get("values", []):
            day = bar["datetime"][:10]
            close = float(bar["close"])
            volume = float(bar.get("volume") or 0)
            if start <= day < end and np.isfinite(close) and close > 0 and date.fromisoformat(day).weekday() < 5:
                values.append((day, close, volume if np.isfinite(volume) else 0))
        values = sorted({v[0]: v for v in values}.values())
        if not values:
            raise ValueError(f"No complete forward prices returned for {symbol}")
        path = directory / (hashlib.sha256(symbol.encode()).hexdigest()[:24] + ".npz")
        np.savez_compressed(
            path,
            dates=np.array([v[0] for v in values], dtype="U10"),
            close=np.array([v[1] for v in values]),
            volume=np.array([v[2] for v in values]),
        )
        records.append(
            {
                **holding,
                "id": holding.get("id") or path.stem,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "type": "Common Stock",
                "name": symbol,
            }
        )
        if symbol == "SPY":
            calendar = [v[0] for v in values]
        state["completed"] += 1
        publish()
    digest = hashlib.sha256("".join(r["symbol"] + r["sha256"] for r in records).encode()).hexdigest()
    data = _finish(
        {
            "name": paper["name"] + " · forward prices",
            "source": "twelvedata_paper",
            "purpose": "paper_only",
            "created": lab_store.now(),
            "assets": len(records),
            "sessions": len(calendar),
            "start": calendar[0],
            "end": calendar[-1],
            "dates": calendar,
            "records": records,
            "security": [],
            "membership": [],
            "input_sha256": digest,
            "coverage": "forward_paper_prices",
            "membership_supplied": False,
            "security_metadata_supplied": False,
            "notes": ["Forward adjusted-close total-return research feed; no real orders or historical fills."],
        },
        digest[:24],
    )
    result = update_paper(identity, data["id"])
    state.update(
        status="completed",
        dataset_id=data["id"],
        stage=result.get("update_note") or "Forward observations verified and appended",
    )
    publish()


if __name__ == "__main__":
    try:
        refresh(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    except Exception as exc:
        state = feed_status()
        state.update(status="failed", stage=str(exc)[:400])
        atomic_json(lab_store.LAB / "paper_refresh.json", state)
        raise SystemExit(1) from None
