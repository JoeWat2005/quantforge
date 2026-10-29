"""Local process controls for the resumable data job and automatic full analysis."""

import json
import os
import subprocess
import sys

import psutil

from backend import lab_store
from backend.engine import ROOT
from backend.market_data import MARKET, STATUS, MarketConfig, atomic_json, collection_state

CONTROL = MARKET / "control.json"


def controls():
    if CONTROL.exists():
        return json.loads(CONTROL.read_text(encoding="utf-8"))
    return {"auto_analyze": True, "auto_resume": True, "analysis": MarketConfig().model_dump()}


def save_controls(config, auto_analyze=True):
    body = controls()
    body.update(auto_analyze=auto_analyze, analysis=config.model_dump())
    atomic_json(CONTROL, body)
    return body


def start_collection(years=10):
    state = collection_state()
    if state.get("status") in ("running", "waiting"):
        return state
    body = controls()
    body["auto_resume"] = True
    body.setdefault("analysis", MarketConfig(years=years).model_dump())
    atomic_json(CONTROL, body)
    log = MARKET / "collector.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        subprocess.Popen(
            [sys.executable, "-m", "backend.market_data", "--years", str(years)],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    return {"status": "starting", "years": years}


def stop_collection(paused_by_user=True):
    state = collection_state()
    pid = state.get("pid")
    if pid:
        try:
            process = psutil.Process(pid)
            command = process.cmdline()
            if "backend.market_data" in command and abs(process.create_time() - state.get("process_created", 0)) < 0.1:
                descendants = process.children(recursive=True)
                for child in reversed(descendants):
                    child.kill()
                process.kill()
                psutil.wait_procs([process, *descendants], timeout=3)
        except psutil.NoSuchProcess:
            pass
    if state.get("status") not in ("completed", "not_started"):
        state.update(status="paused", stage="Downloads paused; completed histories are preserved")
        atomic_json(STATUS, state)
    body = controls()
    body["auto_resume"] = not paused_by_user
    atomic_json(CONTROL, body)
    return state


def latest_market():
    path = MARKET / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def watch_market(controller):
    checked_resume = False
    while not controller.stop.wait(2):
        try:
            state = collection_state()
            policy = controls()
            if not checked_resume:
                checked_resume = True
                if state.get("status") == "paused" and policy.get("auto_resume"):
                    start_collection(state.get("years", 10))
            if state.get("status") != "completed" or not policy.get("auto_analyze", True):
                continue
            if lab_store.next_queued() or lab_store.waiting_data():
                continue
            snapshot = state["snapshot_id"]
            if policy.get("analysis_attempted_for") == snapshot:
                continue
            with controller.lock:
                if controller.process and controller.process.poll() is None:
                    continue
                temperature = controller.telemetry.get("cpu_temperature")
                if temperature is None or temperature > 80:
                    continue
                config = MarketConfig.model_validate({**policy.get("analysis", {}), "snapshot_id": snapshot})
                controller.start(config, module="backend.market_worker")
                policy["analysis_attempted_for"] = snapshot
                atomic_json(CONTROL, policy)
        except Exception as exc:
            # No secrets are included in configuration or process controls.
            error = str(exc)[:300]
            with controller.lock:
                controller.telemetry["market_auto_start_note"] = error
