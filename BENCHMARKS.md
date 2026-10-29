# QUANTFORGE benchmark report

Measured 2026-10-28T16:27:13.291284+00:00 on Windows-11-10.0.26300-SP0.
CPU: AMD Threadripper PRO 5995WX, 64 physical / 128 logical cores.
RAM: 255.84 GiB installed.
GPU: NVIDIA GeForce RTX 4080 SUPER.

Workload: 1,000,000 paths × 12 time steps × 4 assets, one-year horizon, seed 42.
Median of 3 runs. Batch 131,072; no retained scenario cache; CPU thread ceiling 100%; GPU duty 100%.
Wall time includes native calls, host allocations, risk aggregation, progress serialization and any duty sleep. It excludes worker startup and native library loading. Native time includes host output allocation, compute and finite-value checks; CUDA also includes device allocation and host transfer. No warmup runs are excluded.

| Engine | Requested threads | Median wall seconds | Median paths/s | Peak sampled system CPU |
|---|---:|---:|---:|---:|
| CPU | 1 | 1.1932 | 838,098 | 41.5% |
| CPU | 8 | 0.2559 | 3,908,013 | 41.5% |
| CPU | 32 | 0.1589 | 6,294,946 | 40.5% |
| CPU | 64 | 0.1506 | 6,638,330 | 41.2% |
| CPU | 128 | 0.1520 | 6,580,580 | 41.4% |

CUDA: Not compiled: nvcc unavailable in this environment. No GPU speedup is claimed without a measured run.
CPU temperature: unavailable; no long heavy/extreme thermal soak was attempted.

System CPU samples include other applications, and short runs can finish between samples. Requested OpenMP threads do not prove utilization of both Windows processor groups. Compare observed CPU activity before choosing 128 threads; oversubscription and OpenMP team overhead can reduce performance. These are local measurements, not peak hardware claims.

100M and 1B presets are implemented but were not benchmarked here. Their multi-step workloads can take a long time. No extrapolated results are presented as measurements.

Full individual measurements and hardware telemetry are in BENCHMARKS.json. Reproduce with `python scripts/benchmark.py --paths 1000000 --steps 12 --threads 1,8,32,64,128 --repeats 3 --output-prefix BENCHMARKS` against a running local server. Close competing workloads for controlled comparisons.
