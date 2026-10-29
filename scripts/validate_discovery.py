"""Run the real default US universe through the guarded research API."""

import argparse
import json
import time
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--preset", choices=["default", "deep"], default="default")
args = parser.parse_args()
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=20) as client:
    state = client.get("/api/state").json()
    if state["job"]["status"] == "running":
        raise SystemExit("A user job is active; wait before validating discovery")
    config = client.get("/api/research/config").json()["defaults"]
    if args.preset == "deep":
        config.update(shortlist=1000, bootstrap_repetitions=2000, threads=128, cpu_ceiling=95)
    response = client.post("/api/research/start", json=config)
    response.raise_for_status()
    peak_ram = 0.0
    peak_cpu = 0.0
    peak_temp = 0.0
    last_stage = None
    while True:
        state = client.get("/api/state").json()
        job = state["job"]
        telemetry = state["telemetry"]
        peak_ram = max(peak_ram, telemetry.get("worker_ram_gb", 0))
        peak_cpu = max(peak_cpu, telemetry.get("cpu_percent", 0))
        peak_temp = max(peak_temp, telemetry.get("cpu_temperature") or 0)
        if job.get("stage") != last_stage:
            last_stage = job.get("stage")
            print(last_stage, flush=True)
        if job["status"] != "running":
            break
        time.sleep(0.3)
    if job["status"] != "completed":
        raise SystemExit(f"Discovery stopped: {job.get('error')}")
    result = job["discovery"]
    assert result["metadata"]["source"] == "yahoo"
    assert len(result["pairs"]) > 0
    assert result["metadata"]["train_end"] < result["metadata"]["holdout_start"]
    csv = client.get("/api/research/export")
    csv.raise_for_status()
    measurements = {
        "source": "Real Yahoo adjusted daily closes",
        "metadata": result["metadata"],
        "elapsed_seconds": result["elapsed"],
        "bootstrap_resamples": result["native_bootstrap_resamples"],
        "bootstrap_return_observations": result["bootstrap_return_observations"],
        "peak_sampled_worker_tree_gb": peak_ram,
        "peak_sampled_system_cpu_percent": peak_cpu,
        "peak_sampled_cpu_temperature": peak_temp,
        "status": "passed",
        "config": config,
        "export_bytes": len(csv.content),
    }
    filename = "DISCOVERY_DEEP_VALIDATION.json" if args.preset == "deep" else "DISCOVERY_VALIDATION.json"
    (root / filename).write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in measurements.items() if k not in ("config", "metadata")}), flush=True)
