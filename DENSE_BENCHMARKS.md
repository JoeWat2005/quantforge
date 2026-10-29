# QUANTFORGE benchmark report

Measured 2026-10-28T16:27:48.314523+00:00 on Windows-11-10.0.26300-SP0.
CPU: AMD Threadripper PRO 5995WX, 64 physical / 128 logical cores.
RAM: 255.84 GiB installed.
GPU: NVIDIA GeForce RTX 4080 SUPER.

Workload: 2,000,000 paths × 252 time steps × 4 assets, one-year horizon, seed 42.
Median of 3 runs. Batch 131,072; no retained scenario cache; CPU thread ceiling 100%; GPU duty 100%.
Wall time includes native calls, host allocations, risk aggregation, progress serialization and any duty sleep. It excludes worker startup and native library loading. Native time includes host output allocation, compute and finite-value checks; CUDA also includes device allocation and host transfer. No warmup runs are excluded.

| Engine | Requested threads | Median wall seconds | Median paths/s | Peak sampled system CPU |
|---|---:|---:|---:|---:|
| CPU | 64 | 1.3085 | 1,528,492 | 56.7% |
| CPU | 128 | 1.1452 | 1,746,401 | 67.9% |

CUDA: Not compiled: nvcc unavailable in this environment. No GPU speedup is claimed without a measured run.
CPU temperature: unavailable; no long heavy/extreme thermal soak was attempted.

System CPU samples include other applications, and short runs can finish between samples. Requested OpenMP threads do not prove utilization of both Windows processor groups. Compare observed CPU activity before choosing 128 threads; oversubscription and OpenMP team overhead can reduce performance. These are local measurements, not peak hardware claims.

100M and 1B presets are implemented but were not benchmarked here. Their multi-step workloads can take a long time. No extrapolated results are presented as measurements.

Full individual measurements and hardware telemetry are in DENSE_BENCHMARKS.json. Reproduce with `python scripts/benchmark.py --paths 2000000 --steps 252 --threads 64,128 --repeats 3 --output-prefix DENSE_BENCHMARKS` against a running local server. Close competing workloads for controlled comparisons.
