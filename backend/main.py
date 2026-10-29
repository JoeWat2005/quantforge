"""Local control plane. One compute job; sensors and watchdog stay outside it."""

import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import lab_store
from backend.config import PRESETS, RunConfig
from backend.discovery_data import DATA, DiscoveryConfig, parse_prices_csv
from backend.engine import ROOT, library_path
from backend.lab_api import create_router
from backend.lab_feed import stop_refresh
from backend.lab_service import finish_job, watch_research
from backend.market_data import MARKET, MarketConfig, collection_state
from backend.market_service import (
    controls,
    latest_market,
    save_controls,
    start_collection,
    stop_collection,
    watch_market,
)
from backend.telemetry import GIB, Telemetry
from backend.warehouse_api import router as warehouse_router
from backend.warehouse_jobs import stop_migration, watch_warehouse


class Controller:
    def __init__(self):
        self.lock = threading.RLock()
        self.job = {"status": "idle", "completed": 0, "history": []}
        self.process = None
        self.telemetry = {}
        self.stop = threading.Event()
        self.started_at = 0.0

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({"job": self.job, "telemetry": self.telemetry})

    def start(self, config: RunConfig, module="backend.worker"):
        with self.lock:
            policy = lab_store.resource_policy()
            config = config.model_copy(
                update={
                    "ram_gb": min(config.ram_gb, policy["ram_ceiling_gb"]),
                    "threads": min(config.threads, policy["thread_ceiling"], os.cpu_count() or 1),
                    "reserve_gb": max(config.reserve_gb, policy["os_reserve_gb"]),
                    "cpu_temp_limit": min(config.cpu_temp_limit, policy["cpu_stop"]),
                    "max_seconds": min(config.max_seconds, policy["duration_ceiling"]),
                }
            )
            if self.process and self.process.poll() is None:
                raise HTTPException(409, "A simulation is already running")
            if not self.telemetry:
                raise HTTPException(503, "Sensors are initializing; retry in a few seconds")
            for kind in ["cpu", "gpu"] if config.engine == "compare" else [config.engine]:
                if not library_path(kind).exists():
                    raise HTTPException(400, f"{kind.upper()} engine is not built. See setup instructions.")
            if config.require_cpu_temperature and self.telemetry.get("cpu_temperature") is None:
                raise HTTPException(
                    400,
                    "CPU temperature required but unavailable. Run the prepared LibreHardwareMonitor as administrator; its local sensor feed uses port 8085.",
                )
            if config.engine != "cpu" and not self.telemetry.get("gpu"):
                raise HTTPException(400, "GPU telemetry unavailable; CUDA runs require an observable GPU")
            if config.ram_gb + config.reserve_gb > psutil.virtual_memory().available / GIB:
                raise HTTPException(400, "RAM ceiling plus OS reserve exceeds available memory. Lower the ceiling.")
            for sensor, limit in [
                (self.telemetry.get("cpu_temperature"), config.cpu_temp_limit),
                ((self.telemetry.get("gpu") or {}).get("temperature"), config.gpu_temp_limit),
            ]:
                if sensor is not None and sensor >= limit:
                    raise HTTPException(400, "Temperature is already at the configured stop threshold")
            self.started_at = time.monotonic()
            self.job = {
                "id": uuid.uuid4().hex[:12],
                "status": "running",
                "completed": 0,
                "config": config.model_dump(mode="json"),
                "history": [],
                "engines": {},
                "started": time.time(),
                "task": {"backend.discovery_worker": "discovery", "backend.market_worker": "market"}.get(
                    module, "research" if module == "backend.lab_worker" else "simulation"
                ),
            }
            env = os.environ.copy()
            env.update(
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                BLIS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                OMP_WAIT_POLICY="PASSIVE",
            )
            self.process = subprocess.Popen(
                [sys.executable, "-m", module],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.process.stdin.write(config.model_dump_json() + "\n")
            self.process.stdin.close()
            threading.Thread(target=self._read, args=(self.process, self.job["id"]), daemon=True).start()
            return {"id": self.job["id"]}

    def _read(self, process, job_id):
        complete_received = False
        try:
            for line in process.stdout:
                event = json.loads(line)
                with self.lock:
                    if self.job.get("id") != job_id or self.job["status"] != "running":
                        continue
                    kind = event.pop("type")
                    if kind == "ready":
                        self.job["worker_pid"] = event["pid"]
                    elif kind == "progress":
                        self.job.update(event)
                        self.job["history"] = (
                            self.job["history"]
                            + [
                                {
                                    "elapsed": event["elapsed"],
                                    "throughput": event["throughput"],
                                    "engine": event["engine"],
                                }
                            ]
                        )[-120:]
                    elif kind == "engine_complete":
                        self.job["engines"][event["engine"]] = event
                    elif kind == "research_complete":
                        self.job["discovery"] = event["discovery"]
                    elif kind == "market_complete":
                        self.job["market"] = event["market"]
                    elif kind == "lab_complete":
                        self.job["lab_summary"] = event["summary"]
                    elif kind == "error":
                        self.job.update(status="failed", error=event["error"])
                    elif kind == "complete":
                        complete_received = True
            code = process.wait()
            stderr = process.stderr.read()[-4000:]
            with self.lock:
                if self.job.get("id") == job_id and self.job["status"] == "running":
                    if code == 0 and complete_received:
                        self.job["status"] = "completed"
                    else:
                        self.job.update(status="failed", error=f"Worker exited unexpectedly ({code}). {stderr}")
        except (ValueError, OSError) as exc:
            with self.lock:
                if self.job.get("id") == job_id:
                    self.cancel("failed", f"Worker protocol failed: {exc}")
        finally:
            with self.lock:
                if self.job.get("id") == job_id:
                    self._save()

    def _save(self):
        finish_job(self.job)
        directory = ROOT / "runs"
        directory.mkdir(exist_ok=True)
        (directory / (self.job["id"] + ".json")).write_text(json.dumps(self.job, indent=2), encoding="utf-8")
        # Bound NVMe usage. Each file is summaries and a tiny sample only.
        files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for file in files[:-20]:
            file.unlink()

    def cancel(self, status="cancelled", reason="Cancelled by user"):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.job.update(status=status, error=reason)
                # Windows venv python can be a launcher with a real interpreter
                # child. Account for and terminate the complete compute tree.
                try:
                    root = psutil.Process(self.process.pid)
                    descendants = root.children(recursive=True)
                    for child in reversed(descendants):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    descendants = []
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
                psutil.wait_procs(descendants, timeout=2)
                finish_job(self.job)
            return {"status": self.job["status"]}

    def worker_rss_gb(self):
        root = psutil.Process(self.process.pid)
        total = 0
        for process in [root, *root.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total / GIB

    def monitor(self):
        sensors = Telemetry()
        while not self.stop.is_set():
            try:
                data = sensors.read()
                with self.lock:
                    self.telemetry = data
                    if self.process and self.process.poll() is None:
                        config = RunConfig.model_validate(self.job["config"])
                        try:
                            rss = self.worker_rss_gb()
                        except psutil.NoSuchProcess:
                            rss = 0
                        data["worker_ram_gb"] = rss
                        reason = None
                        if data["ram_available_gb"] < config.reserve_gb:
                            reason = "OS memory reserve reached"
                        if rss > config.ram_gb:
                            reason = "Worker RAM ceiling reached"
                        if time.monotonic() - self.started_at > config.max_seconds:
                            reason = "Run duration limit reached"
                        if data["cpu_temperature"] is not None and data["cpu_temperature"] >= config.cpu_temp_limit:
                            reason = "CPU temperature limit reached"
                        if config.require_cpu_temperature and data["cpu_temperature"] is None:
                            reason = "Required CPU temperature sensor lost"
                        gpu_temp = (data.get("gpu") or {}).get("temperature")
                        if gpu_temp is not None and gpu_temp >= config.gpu_temp_limit:
                            reason = "GPU temperature limit reached"
                        if config.engine != "cpu" and gpu_temp is None:
                            reason = "GPU temperature sensor lost"
                        if reason:
                            self.cancel("stopped", reason)
            except (OSError, psutil.Error, ValueError) as exc:
                with self.lock:
                    self.telemetry["sensor_error"] = str(exc)
                    self.cancel("stopped", "Resource monitor failed")
            self.stop.wait(0.5)

    def watchdog(self):
        """Memory/time enforcement never waits for a slow hardware sensor query."""
        while not self.stop.wait(0.25):
            with self.lock:
                if not self.process or self.process.poll() is not None:
                    continue
                config = RunConfig.model_validate(self.job["config"])
                try:
                    rss = self.worker_rss_gb()
                    available = psutil.virtual_memory().available / GIB
                    self.telemetry["worker_ram_gb"] = rss
                    reason = None
                    if rss > config.ram_gb:
                        reason = "Worker RAM ceiling reached"
                    if available < config.reserve_gb:
                        reason = "OS memory reserve reached"
                    if time.monotonic() - self.started_at > config.max_seconds:
                        reason = "Run duration limit reached"
                    if time.time() - self.telemetry.get("timestamp", 0) > 12:
                        reason = "Resource telemetry is stale"
                    if reason:
                        self.cancel("stopped", reason)
                except psutil.NoSuchProcess:
                    pass
                except (OSError, psutil.Error):
                    self.cancel("stopped", "Memory watchdog failed")


controller = Controller()


@asynccontextmanager
async def lifespan(app):
    controller.stop.clear()
    thread = threading.Thread(target=controller.monitor, daemon=True)
    guard = threading.Thread(target=controller.watchdog, daemon=True)
    thread.start()
    guard.start()
    market_watch = threading.Thread(target=watch_market, args=(controller,), daemon=True)
    market_enabled = os.environ.get("QF_DISABLE_MARKET_SERVICE") != "1"
    if market_enabled:
        market_watch.start()
    research_watch = threading.Thread(target=watch_research, args=(controller,), daemon=True)
    if market_enabled:
        research_watch.start()
    warehouse_watch = threading.Thread(target=watch_warehouse, args=(controller,), daemon=True)
    if market_enabled:
        warehouse_watch.start()
    yield
    controller.stop.set()
    controller.cancel(reason="Server shutdown")
    if market_enabled:
        stop_collection(paused_by_user=False)
        stop_refresh()
        stop_migration(user=False)
    thread.join(timeout=6)
    guard.join(timeout=2)
    if market_enabled:
        market_watch.join(timeout=2)
        research_watch.join(timeout=2)
        warehouse_watch.join(timeout=2)


app = FastAPI(title="QUANTFORGE", lifespan=lifespan)
app.include_router(create_router(controller))
app.include_router(warehouse_router)


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = request.url.hostname
    origin = request.headers.get("origin")
    if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
        return JSONResponse({"detail": "Local requests only"}, status_code=403)
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        return JSONResponse({"detail": "Cross-origin control is disabled"}, status_code=403)
    return await call_next(request)


@app.get("/api/config")
def configuration():
    return {
        "defaults": RunConfig().model_dump(),
        "presets": PRESETS,
        "engines": {kind: library_path(kind).exists() for kind in ("cpu", "gpu")},
    }


@app.get("/api/state")
def state():
    return controller.snapshot()


@app.get("/api/market/state")
def market_state():
    return {
        "collection": collection_state(),
        "control": controls(),
        "latest": latest_market(),
        "compute": controller.snapshot(),
    }


@app.get("/api/market/config")
def market_config():
    return {"defaults": MarketConfig().model_dump(), "collection": collection_state(), "control": controls()}


class CollectionRequest(BaseModel):
    years: int = Field(default=10, ge=3, le=15)


@app.post("/api/market/collect")
def market_collect(payload: CollectionRequest):
    return start_collection(payload.years)


@app.post("/api/market/pause")
def market_pause():
    return stop_collection(paused_by_user=True)


@app.post("/api/market/analyze")
def market_analyze(config: MarketConfig):
    return controller.start(config, module="backend.market_worker")


class MarketPolicy(BaseModel):
    analysis: MarketConfig
    auto_analyze: bool = True


@app.post("/api/market/policy")
def market_policy(payload: MarketPolicy):
    return save_controls(payload.analysis, payload.auto_analyze)


@app.get("/api/market/export/{kind}")
def market_export(kind: str):
    report = latest_market()
    if not report:
        raise HTTPException(404, "Complete a market analysis first")
    directory = MARKET / "analysis" / report["metadata"]["analysis_id"]
    files = {"relationships": "relationships.csv", "baskets": "baskets.csv", "report": "report.json"}
    if kind not in files:
        raise HTTPException(404, "Unknown report")
    return FileResponse(directory / files[kind], filename="quantforge-market-" + files[kind])


@app.get("/api/research/config")
def research_configuration():
    return {
        "defaults": DiscoveryConfig().model_dump(),
        "native_ready": library_path("research").exists(),
        "csv_format": "Date,AAPL,MSFT,SPY followed by positive USD adjusted daily closing prices",
    }


class PriceImport(BaseModel):
    csv: str = Field(min_length=30, max_length=16_000_000)


@app.post("/api/research/import")
def research_import(payload: PriceImport):
    if controller.snapshot()["job"]["status"] == "running":
        raise HTTPException(409, "Finish the active job before importing data")
    try:
        frame = parse_prices_csv(payload.csv)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    directory = DATA / "imports"
    directory.mkdir(parents=True, exist_ok=True)
    raw = payload.csv.encode("utf-8")
    key = hashlib.sha256(raw).hexdigest()
    path = directory / (key + ".csv")
    if not path.exists() and sum(p.stat().st_size for p in directory.glob("*.csv")) + len(raw) > 256 * 1024**2:
        raise HTTPException(400, "Import storage limit reached (256 MiB); manage data/imports before adding more")
    path.write_bytes(raw)
    return {"dataset_id": key, "assets": len(frame.columns), "rows": len(frame), "tickers": frame.columns.tolist()}


@app.post("/api/research/start")
def research_start(config: DiscoveryConfig):
    if config.engine != "cpu":
        raise HTTPException(400, "Discovery currently uses the CPU research engine")
    if not library_path("research").exists():
        raise HTTPException(400, "Research engine is not built; run python scripts/build.py")
    return controller.start(config, module="backend.discovery_worker")


@app.get("/api/research/latest")
def research_latest():
    path = DATA / "research" / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@app.get("/api/research/export")
def research_export():
    report = research_latest()
    if not report:
        raise HTTPException(404, "Complete an asset discovery run first")
    filename = report["metadata"]["csv_file"]
    return FileResponse(DATA / "research" / filename, media_type="text/csv", filename="quantforge-discovery-pairs.csv")


@app.post("/api/start")
def start(config: RunConfig):
    return controller.start(config)


@app.post("/api/cancel")
def cancel():
    return controller.cancel()


@app.get("/api/report")
def report():
    return JSONResponse(
        controller.snapshot(), headers={"Content-Disposition": 'attachment; filename="quantforge-report.json"'}
    )


@app.get("/api/events")
async def events(request: Request):
    async def stream():
        while not await request.is_disconnected():
            yield "data: " + json.dumps(controller.snapshot(), allow_nan=False) + "\n\n"
            await asyncio.sleep(0.75)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def dashboard():
        return FileResponse(DIST / "index.html")
