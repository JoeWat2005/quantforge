"""A small, bounded RAM retention/eviction experiment through the guarded API."""

import json
import time
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[1]
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=15) as client:
    if client.get("/api/state").json()["job"]["status"] == "running":
        raise SystemExit("An existing run is active; wait before validating cache eviction")
    config = client.get("/api/config").json()["defaults"]
    config.update(paths=20_000_000, steps=1, ram_gb=1, memory_stress=True, threads=64, max_seconds=30)
    response = client.post("/api/start", json=config)
    response.raise_for_status()
    peak = 0.0
    while True:
        state = client.get("/api/state").json()
        peak = max(peak, state["telemetry"].get("worker_ram_gb", 0))
        if state["job"]["status"] != "running":
            break
        time.sleep(0.05)
    job = state["job"]
    assert job["status"] == "completed", job.get("error")
    result = job["engines"]["cpu"]
    assert 0 < result["cache_paths"] < config["paths"], "Cache must retain and evict real scenarios"
    assert result["cache_gb"] < config["ram_gb"]
    assert peak <= config["ram_gb"], f"Observed tree RSS {peak} exceeded ceiling"
    result = {
        "config": config,
        "result": result,
        "peak_sampled_worker_tree_gb": peak,
        "status": "passed",
        "note": "Small 1 GiB ceiling experiment, not a large-RAM stress or thermal soak",
    }
    (root / "CACHE_VALIDATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cache_paths": result["result"]["cache_paths"],
                "cache_gb": result["result"]["cache_gb"],
                "peak_tree_gb": peak,
            }
        )
    )
