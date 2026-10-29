# Verification on the available machine

Windows 11, AMD Threadripper PRO 5995WX (64 cores / 128 logical processors), 255.84 GiB reported RAM, NVIDIA RTX 4080 SUPER (16,376 MiB reported VRAM).

- `setup.ps1`: passed, creating a project-local environment, installing dependencies, compiling C++20/OpenMP, installing the locked frontend packages, and building the dashboard.
- `start.ps1`: passed; loopback FastAPI server serves the production React dashboard.
- `python -m pytest -q`: **38 passed, 1 skipped**. The skipped test requires the optional CUDA library. A dependency deprecation warning notes that this Starlette version prefers httpx2 for its test client; current httpx tests passed.
- Python formatting and lint checks: passed using Ruff 0.17.3.
- `npm run build`: passed. `npm ci` audit reported zero vulnerabilities at install time.
- Browser checks: dashboard loaded at both the narrow in-app viewport and 1,440 px desktop width; launch control transitioned to running and then completed, with actual progress, distribution, VaR/CVaR and comparison row displayed. Production assets were rebuilt and reloaded after UI changes.
- Cancellation test waits for the actual interpreter's ready event, verifies tree RAM accounting, cancels a billion-path job, checks cancellation responds within three seconds, and confirms that the compute PID no longer exists. This includes Windows virtual environment launcher/child handling.
- Duration and simulated low-available-memory tests stopped actual workers. These checks simulate a resource threshold safely; no real memory exhaustion or overheating was induced.
- Required but missing CPU temperature readings reject launch. Missing sensors are displayed explicitly.
- A 20M-path × 1-step × 4-asset RAM test under a 1 GiB ceiling retained 12,266,752 scenarios in a 0.45697 GiB rolling cache, evicted older scenarios, and completed with 0.52158 GiB peak sampled worker-tree RSS. See `CACHE_VALIDATION.json`; this is a small bounded-memory verification, not a 192 GiB stress test.
- CPU benchmark measurements and raw repetitions are recorded in `BENCHMARKS.md` / `.json` and `DENSE_BENCHMARKS.md` / `.json`. CPU temperature was unavailable, so no sustained Heavy/Extreme soak was attempted.
- CUDA source and build instructions are supplied, but `nvcc` is unavailable. CUDA compilation, kernel correctness, GPU cancellation and CPU/GPU speedup remain unverified on this machine. No fabricated GPU numbers are included.
- Linux/macOS and CMake/MSVC builds were not exercised. The tested compiler is MSYS2 UCRT64 GCC 15.2.0.

Risk results use explicit histogram error bounds, with a unit test against exact empirical tail statistics. These are numerical discretization bounds, not sampling-error or financial forecasting guarantees. All model assumptions are illustrative.

Sensor integration repair: current LibreHardwareMonitor HTTP metrics are supported, including hottest-CPU selection, Celsius units independent of display units, exclusion of GPU/motherboard sensors, and explicit missing/invalid readings. Three additional tests passed. The official v0.9.6 portable release was downloaded and SHA256-verified; a loopback-only feed was configured. Real CPU temperature still requires the user to start the monitor with Windows administrator approval.

Asset-discovery extension: 22 tests passed, one optional CUDA test skipped. Native block bootstrap, training-only screening, held-out pair evaluation, CSV import/export and relationship grouping are verified. Two guarded real-data experiments scanned 102 US assets; detailed measurements and caveats are in DISCOVERY.md and the discovery validation JSON files. The Deep preset was also launched through the React dashboard.
Research cancellation was also verified while its process pool was active: the worker and all observed descendant processes exited.

US-market extension: exact pairwise-complete correlation estimates match independent per-pair calculations with missing values. Direct and Woodbury precision calculations agree. A synthetic three-asset stationary relationship is detected, and changing the later data cannot change training weights. Reproducible unique basket generation, reference-calendar alignment, next-interval execution, fixed-price strategy/benchmark costs, conservative fund-share-class filtering, resumable database records, OS process leases, private-header authentication, key-redacted errors, Windows status-file sharing retries, compulsory sensor/CPU-engine validation, and exactly-once automatic analysis after queue completion are tested. Real cached-data preview, process-pool cancellation and measured resources are recorded in MARKET_BENCHMARKS.md. The full catalogue job remains incomplete; no full-market completion or GPU speedup is claimed.

## Research Studio and Parquet verification · 2026-10-29

The complete suite passed **63 tests**, with **one optional CUDA test skipped** (26.51 seconds). Added checks cover chronological and manual windows, training isolation, factor removal, portfolio costs/caps/missing data, saved API execution, exports, cancellation/resume, forward paper revisions/idempotence, shared provider credits, nullable legacy OHLC, Parquet version selection, frozen file sets, bounded dated API exports and invalid input rejection. Ruff lint and production Vite build passed. DuckDB/Arrow and the FastAPI test client emitted deprecation warnings; these did not fail the checks.

Real stock pilot: STUDIO_VALIDATION.json / STUDIO_REPORT.json. Parquet storage snapshot: WAREHOUSE_STATUS.json. Measured peak CPU temperature 75.6°C; 6,000 basket fits across six folds completed. No full-market research result, GPU research benchmark, maximum-RAM stress or live brokerage execution is claimed.

Expanded CPU pool: the Research Studio unit/API checks passed22 tests after removing the32-worker cap. CPU/RAM/work-count bounds and existing execution/cancellation/checkpoint tests passed. A real61-process,10,000-attempt check completed; see CPU_RESEARCH_VALIDATION.json. This is not a claim of100% sustained CPU usage or a safe long-term temperature.
