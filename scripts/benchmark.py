"""Measured, guarded API benchmarks. Server must be running on loopback."""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--paths", type=int, default=1_000_000)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--threads", default="1,8,32,64,128")
parser.add_argument("--output-prefix", default="BENCHMARKS")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
client = httpx.Client(base_url=args.url, timeout=15)
settings = client.get("/api/config").json()
initial = client.get("/api/state").json()
if initial["job"]["status"] == "running":
    raise SystemExit("A user run is active; finish or cancel it before benchmarking.")
results = []
for kind in ["cpu", "gpu"] if settings["engines"]["gpu"] else ["cpu"]:
    counts = [int(t) for t in args.threads.split(",")] if kind == "cpu" else [1]
    for threads in counts:
        for repeat in range(args.repeats):
            config = {
                **settings["defaults"],
                "paths": args.paths,
                "steps": args.steps,
                "threads": threads,
                "cpu_ceiling": 100,
                "gpu_duty": 100,
                "engine": kind,
                "max_seconds": 60,
            }
            response = client.post("/api/start", json=config)
            response.raise_for_status()
            peak_cpu = 0.0
            peak_ram = 0.0
            peak_gpu_temp = None
            while True:
                snapshot = client.get("/api/state").json()
                job = snapshot["job"]
                telemetry = snapshot["telemetry"]
                peak_cpu = max(peak_cpu, telemetry.get("cpu_percent", 0))
                peak_ram = max(peak_ram, telemetry.get("worker_ram_gb", 0))
                temp = (telemetry.get("gpu") or {}).get("temperature")
                if temp is not None:
                    peak_gpu_temp = max(peak_gpu_temp or temp, temp)
                if job["status"] != "running":
                    break
                time.sleep(0.1)
            if job["status"] != "completed":
                raise SystemExit(f"Benchmark stopped: {job.get('error')}")
            result = job["engines"][kind]
            entry = {
                "engine": kind,
                "threads": threads,
                "repeat": repeat + 1,
                "paths": args.paths,
                "steps": args.steps,
                "wall_seconds": result["elapsed"],
                "native_seconds": result["native_seconds"],
                "paths_per_second": args.paths / result["elapsed"],
                "var": result["results"]["var"],
                "cvar": result["results"]["cvar"],
                "peak_sampled_system_cpu_percent": peak_cpu,
                "peak_sampled_worker_gb": peak_ram,
                "peak_sampled_gpu_temp": peak_gpu_temp,
            }
            results.append(entry)
            print(json.dumps(entry), flush=True)
report = {
    "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    "platform": platform.platform(),
    "hardware": initial["telemetry"],
    "config": config,
    "results": results,
    "cuda_status": "Measured" if settings["engines"]["gpu"] else "Not compiled: nvcc unavailable in this environment",
}
(root / (args.output_prefix + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
rows = []
for kind, threads in dict.fromkeys((r["engine"], r["threads"]) for r in results):
    group = [r for r in results if r["engine"] == kind and r["threads"] == threads]
    rows.append(
        f"| {kind.upper()} | {threads} | {statistics.median(r['wall_seconds'] for r in group):.4f} | {statistics.median(r['paths_per_second'] for r in group):,.0f} | {max(r['peak_sampled_system_cpu_percent'] for r in group):.1f}% |"
    )
text = f"""# QUANTFORGE benchmark report

Measured {report["measured_at_utc"]} on {report["platform"]}.
CPU: AMD Threadripper PRO 5995WX, {initial["telemetry"].get("physical_cores")} physical / {initial["telemetry"].get("logical_cores")} logical cores.
RAM: {initial["telemetry"].get("ram_total_gb", 0):.2f} GiB installed.
GPU: {(initial["telemetry"].get("gpu") or {}).get("name", "unavailable")}.

Workload: {args.paths:,} paths × {args.steps} time steps × 4 assets, one-year horizon, seed 42.
Median of {args.repeats} runs. Batch 131,072; no retained scenario cache; CPU thread ceiling 100%; GPU duty 100%.
Wall time includes native calls, host allocations, risk aggregation, progress serialization and any duty sleep. It excludes worker startup and native library loading. Native time includes host output allocation, compute and finite-value checks; CUDA also includes device allocation and host transfer. No warmup runs are excluded.

| Engine | Requested threads | Median wall seconds | Median paths/s | Peak sampled system CPU |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

CUDA: {report["cuda_status"]}. No GPU speedup is claimed without a measured run.
CPU temperature: {"available" if initial["telemetry"].get("cpu_temperature") is not None else "unavailable; no long heavy/extreme thermal soak was attempted"}.

System CPU samples include other applications, and short runs can finish between samples. Requested OpenMP threads do not prove utilization of both Windows processor groups. Compare observed CPU activity before choosing 128 threads; oversubscription and OpenMP team overhead can reduce performance. These are local measurements, not peak hardware claims.

100M and 1B presets are implemented but were not benchmarked here. Their multi-step workloads can take a long time. No extrapolated results are presented as measurements.

Full individual measurements and hardware telemetry are in {args.output_prefix}.json. Reproduce with `python scripts/benchmark.py --paths {args.paths} --steps {args.steps} --threads {args.threads} --repeats {args.repeats} --output-prefix {args.output_prefix}` against a running local server. Close competing workloads for controlled comparisons.
"""
(root / (args.output_prefix + ".md")).write_text(text, encoding="utf-8")
