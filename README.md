# QUANTFORGE Research Studio

A local quantitative research app for testing investment ideas against historical data and explicit assumptions. React dashboard, FastAPI backend, Parquet prices, multicore statistical research, and a separate C++20/OpenMP portfolio Monte Carlo simulator with optional CUDA source.

**This is a research tool. It does not connect to Trading 212, send orders, guarantee returns, or establish that a historical strategy was executable.**

## What it does

| Workspace | Purpose |
|---|---|
| Research Studio (`/#research`) | Freeze datasets, define dated experiments, queue/cancel work, inspect reports and keep forward paper records |
| Price warehouse (inside Studio) | Browse stored stock histories and export Parquet |
| Market scan (`/#market`) | Return correlations, PCA, conditional associations and sampled 3/5/8-stock baskets |
| Pair discovery (`/#discovery`) | Parallel bootstrap and two-asset relationship research |
| Monte Carlo (`/#simulator`) | Generate hypothetical correlated portfolio outcomes and estimate tail risk |
| Monthly trend CLI | Test a fixed ETF trend rule against buy-and-hold and declared sensitivity checks |

Historical research uses downloaded/imported observations. Monte Carlo generates hypothetical futures from supplied assumptions. These are different tasks.

Each historical walk-forward fold separates earlier training observations, validation for a small declared rule grid, and a later non-overlapping test. Current dashboard strategies are diversified long-only, basket mean reversion, or validation choosing between them. Reports include returns, volatility, drawdown, Sharpe, benchmark curves, cost stress, per-window audits, relationships, basket diagnostics, simulated orders and input/code fingerprints.

## Start on Windows

Requirements: Python 3.11+, Node.js 20.19+ or 22.12+, and a C++ compiler with OpenMP. The tested native compiler is MSYS2 UCRT64 GCC; put its `bin` directory on PATH. CUDA is optional and requires a compatible toolkit. Windows is the tested host.

```powershell
./setup.ps1
./start.ps1
```

Open **http://127.0.0.1:8000/#research**. The server binds to loopback. A new checkout builds native libraries and frontend assets locally. Dependencies, binaries, API keys, market data and local run records are excluded from Git.

`requirements-lock.txt` records the tested Python versions; `requirements.txt` contains compatible ranges used by setup. The frontend commits its npm lockfile.

### Hardware monitoring

Research requires a readable CPU temperature. On the tested Windows setup, official LibreHardwareMonitor runs as administrator with its loopback metrics feed on `127.0.0.1:8085`. See [STUDIO.md](STUDIO.md). Missing/stale sensor readings block or stop compute. RAM ceilings, OS reserve, temperature, duration and process-tree cancellation guards remain active.

Designed for a 64-core/128-thread workstation with 256 GiB RAM, but settings can be reduced. Matrix work respects the thread ceiling. Basket fits can use up to 61 single-threaded child processes on Windows, reduced by CPU/RAM/work limits. CPU usage varies by stage. Historical research does not use the GPU. Optional Monte Carlo CUDA source is provided, but CUDA compilation/correctness/performance remain unverified on the tested host.

## Data setup

**CSV:** use Data library to import `Date,SPY,AAPL,...` with positive split/dividend-adjusted USD daily closing prices. Optional `SYMBOL_volume` columns supply volume. Stock research needs at least ten assets including SPY and sufficient history. Optional dated security-identity and membership CSVs improve provenance.

**Twelve Data:** configure your own private key locally before starting:

```powershell
$env:TWELVEDATA_API_KEY = 'YOUR_OWN_KEY'
./start.ps1
```

Alternatively use `.secrets/twelvedata.key`, which is ignored by Git. No key or data entitlement is bundled. The provider uses header authentication and a shared conservative credit budget; coverage/rates depend on your subscription.

Start collection in Data library, then enable **Price warehouse → Resume sync**. Collection is scoped to the current provider US equity catalogue, including OTC and other equity types. Pending non-reference funds are deferred; required market-reference ETFs and already cached funds can remain available. A current catalogue is not a historical point-in-time universe.

Prices are converted to immutable Zstandard Parquet, indexed through transactional SQLite manifests and queried with DuckDB. Legacy close-only inputs retain NULL open/high/low fields. Frozen datasets pin published files/hashes so later downloads cannot alter an existing experiment. See [WAREHOUSE.md](WAREHOUSE.md).

## Run the monthly ETF trend experiment

The included CLI tests a fixed rule: at completed month-end, hold the selected instrument when its adjusted close exceeds its trailing 10-month average; otherwise hold cash. Execute at a later daily close. Compare with buy-and-hold, doubled costs, 8/12-month sensitivity and an extra execution delay on common dates.

```powershell
# CSV columns: Date,SPY. Supply positive observed adjusted daily closes.
.venv/Scripts/python.exe -m backend.trend_research --csv prices.csv --symbol SPY --output runs/trend-study

# Or supply a frozen dataset ID from your local Studio:
.venv/Scripts/python.exe -m backend.trend_research --dataset-id YOUR_LOCAL_DATASET_ID --symbol SPY --output runs/trend-study
```

Optional flags: `--start YYYY-MM-DD`, `--end YYYY-MM-DD`, `--cost-bps 11`. Supply preceding history for monthly warm-up. Results are written to `report.json` (provenance, orders, curves) and `summary.csv`. Costs are assumed basis points per side; cash interest is zero. Input currency is not inferred or converted. The runner never downloads prices or places trades.

This is a **CLI module**, not a strategy in the dashboard menu. [TRADING212_RESEARCH.md](TRADING212_RESEARCH.md) explains the earlier exploratory SPY USD check, its weak primary result, and why it is not a GBP brokerage backtest.

## Not implemented

- Monthly stock-momentum ranking: [STRATEGY_RESEARCH_PLAN.md](STRATEGY_RESEARCH_PLAN.md) is a proposed protocol, not a working dashboard strategy.
- Trading 212 account access, real-money execution or live buy recommendations.
- CFD backtests with historical financing, broker spreads, margin calls and close-outs.
- Options-chain backtests with premiums, bid/ask spreads, assignment/exercise and contract details.
- Certified historical constituents, complete delisting returns or automatic resolution of every ticker change/reuse.

## Research limitations

Correlation is not profit or causation. Current catalogues introduce survivorship bias. Old/new ticker aliases and reused symbols can create spurious relationships; unresolved identity findings need review. Successful fits and training screens are not proof of a trading edge.

Daily-close execution, costs, cash/funding/borrow inputs and adjusted-price units are approximations. Missing held prices invalidate evaluation rather than creating a fictitious sale. Intraday fills, taxes and corporate-action cashflows are not fully reconstructed. Conditional bootstrap intervals do not correct all selection, data or model error. Already-inspected dates are development evidence; new forward observations are needed for fresh evidence.

UK execution needs the actual eligible instrument and GBP cash/currency treatment. A USD SPY study cannot be relabelled a UK UCITS, CFD or options result. Printed allocation weights are research specifications, not a buy list.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest -q
ruff check backend tests
cd frontend
npm.cmd run build
```

Tests cover numerical engines, resource guards, cancellation, chronology, train/test isolation, accounting, paper records, Parquet versions/frozen reads, exports and monthly trend timing/costs. Optional CUDA checks skip without the toolchain. See [TESTING.md](TESTING.md) for dated observations and limitations. Detailed local price-derived reports are intentionally not committed.

## Source map

- `backend/lab_*`: data/experiment persistence, chronology, models, accounting, guarded workers, queue, paper feed and API.
- `backend/warehouse*`, `provider_budget.py`: Parquet/SQLite/DuckDB access, conversion and provider quota.
- `backend/trend_research.py`: monthly ETF trend research CLI.
- `frontend/src/Studio*`, `PriceWarehouse.jsx`: React app and reports.
- `native/`: C++20/OpenMP engines and optional CUDA source.
- `scripts/`, `tests/`: builds, validation helpers and tests.

Further detail: [STUDIO.md](STUDIO.md), [WAREHOUSE.md](WAREHOUSE.md), [DISCOVERY.md](DISCOVERY.md), [MARKET.md](MARKET.md), [COMPREHENSIVE_STUDY.md](COMPREHENSIVE_STUDY.md).
