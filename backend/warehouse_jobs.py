"""Lifecycle controls for the low-memory Parquet synchronizer."""

import json
import os
import subprocess
import sys
import threading

import psutil

from backend.engine import ROOT
from backend.market_data import atomic_json
from backend.warehouse import WAREHOUSE

LOCK = threading.RLock()


def migration_state():
    path = WAREHOUSE / "migration.json"
    if not path.exists():
        return {"status": "not_started"}
    body = json.loads(path.read_text())
    if body.get("status") in ("running", "starting"):
        try:
            if abs(psutil.Process(body["pid"]).create_time() - body["process_created"]) > 0.1:
                body["status"] = "interrupted"
        except psutil.NoSuchProcess:
            body["status"] = "interrupted"
    return body


def start_migration():
    with LOCK:
        current = migration_state()
        if current["status"] in ("running", "starting"):
            return current
        WAREHOUSE.mkdir(parents=True, exist_ok=True)
        atomic_json(WAREHOUSE / "control.json", {"watch": True})
        with (WAREHOUSE / "migration.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "backend.warehouse_migrate", "--watch"],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        return {"status": "starting", "pid": process.pid}


def stop_migration(user=True):
    with LOCK:
        current = migration_state()
        if user:
            atomic_json(WAREHOUSE / "control.json", {"watch": False})
        if current.get("status") == "running":
            try:
                process = psutil.Process(current["pid"])
                if (
                    abs(process.create_time() - current["process_created"]) < 0.1
                    and "backend.warehouse_migrate" in process.cmdline()
                ):
                    for child in process.children(recursive=True):
                        child.kill()
                    process.kill()
                    psutil.wait_procs([process], timeout=2)
            except psutil.NoSuchProcess:
                pass
            current.update(status="paused", stage="Parquet synchronization paused; committed files remain readable")
            atomic_json(WAREHOUSE / "migration.json", current)
        return current


def watch_warehouse(controller):
    while not controller.stop.wait(10):
        path = WAREHOUSE / "control.json"
        if not path.exists():
            continue
        try:
            if json.loads(path.read_text()).get("watch") and migration_state()["status"] in (
                "paused",
                "interrupted",
                "not_started",
            ):
                start_migration()
        except (OSError, ValueError, psutil.Error):
            continue
