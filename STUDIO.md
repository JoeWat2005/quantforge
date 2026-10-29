# Research Studio

Start with `./start.ps1`, then open http://127.0.0.1:8000/#research. First setup: `./setup.ps1`; the dependency lock records the tested Windows versions. Keep LibreHardwareMonitor running as administrator with its loopback sensor feed on port 8085. Missing or stale CPU temperature blocks compute. This app places no real orders.

## Use the app

1. **Price warehouse:** inspect the downloaded stocks and export Parquet. Collection and conversion continue in the background.
2. **Data library:** freeze committed prices, or import positive USD adjusted daily prices with optional dated identity, membership and cost files. Review coverage and large-move flags. Dataset manifests and file hashes preserve reproducibility.
3. **New experiment:** choose a frozen dataset and dates. Preview the exact training, validation and test windows. Select daily or weekly relationship observations, market/sector factor removal, liquidity filters, basket sizes, costs and resource ceilings. Save a draft or queue it. The full-data option waits for the current stock collection and its conversion before attaching a snapshot once.
4. **Experiments:** follow the persistent queue, open reports, cancel immediately, resume compatible completed checkpoints after interruption, or copy settings for another trial. Completed configurations are immutable.
5. **Results:** compare after-cost curves, SPY, equal-weight and diversified baselines; inspect each window, candidate baskets, factors, raw/conditional relationships, costs, trade ledger, input hashes and audit records. Download CSV/JSON or the report bundle.
6. **Paper portfolios:** freeze the latest fitted long-only allocation from a completed experiment. Its record begins at creation and accepts only complete later observations. Refresh retrieves eligible completed daily prices through the existing private provider connection. Same-day refresh correctly makes no requests when no eligible dates can exist. Missing holdings or incoherent revisions do not become fabricated fills.

## What the research does

Correlations use returns, not upward-trending price levels. Weekly mode reduces observation frequency using one common reference session per week. A training-only ridge regression can remove observed market/sector ETF influences, followed by PCA and conditional association modelling. These are statistical controls, not proof of causation or a full inflation/fundamental model.

Each fold filters using earlier liquidity, coverage, activity and dated metadata. It samples 3/5/8-asset Johansen baskets, fits only on training observations, and selects among a small declared strategy/rule grid on validation observations. The next non-overlapping test measures that frozen choice. Default chronology: three training years, twelve validation months, twelve test months, annual advances, and twenty-session boundary gaps. Manual or expanding windows are also supported. Maximum thirty folds, two thousand training-shortlisted assets and one hundred thousand basket attempts per fold; exceeding limits is rejected or visibly capped by configuration, never an exhaustive search of all baskets.

Execution uses daily adjusted closes: an observed signal executes at the next close. Synthetic units, turnover, commission, half-spread, slippage, a training-volume impact approximation, cash, funding and short borrow are recorded. Portfolio gross/net/name/known-sector limits and a drawdown stop apply. Missing held prices invalidate an evaluation instead of dropping the failed asset. Fold boundaries close positions; gaps hold modeled cash. Baselines and a double-cost stress are included. Bootstrap intervals are conditional on the frozen research procedure, not corrected evidence that the selected strategy will profit.

## Boundaries

Provider data use today's catalogue, including OTC and unusual share classes. Dated historical membership, delisting returns, true security identity, sectors and borrow availability are not automatically purchased or reconstructed. Import reliable dated metadata when available. Unknown liquidity/borrowing assumptions are explicit. Adjusted-close synthetic returns are not executable historical orders; corporate-action cashflows, actual borrow recalls, intraday order books, taxes and exchange-specific fills are not fully modelled.

Already inspected historical periods become development evidence. Multiple experiments create selection bias even when every individual fold is chronological. A fresh forward paper interval is useful; no historical result guarantees future returns. The research workflow currently uses CPU/BLAS/multiprocessing; GPU acceleration is not implemented for it. The separate Monte Carlo workspace retains optional unverified CUDA source.

## Real validation

See STUDIO_VALIDATION.json and STUDIO_REPORT.json. A stock-only, 40-symbol requested-universe pilot ran six windows and 6,000 fits from a frozen Parquet dataset: 34.51 seconds worker time, 38.48 seconds observed including queue dispatch, sampled maximum 1.96 GiB worker-tree RSS and 75.6°C CPU temperature. These are measured checks, not a maximum-load benchmark. Selected after-cost total return was 50.36%, versus SPY 129.79%, with different risk/exposure. It did not outperform SPY on total return. This present-day selected sample does not establish an investable edge.

A larger waiting experiment uses the completed stock scope, per-fold training-liquidity shortlist of up to 1,000 names, 20,000 basket attempts per fold, 64 threads, 64 GiB ceiling, six-hour duration ceiling and CPU stop 85°C. It waits for complete download/conversion and readable temperature below 80°C before dispatch. Actual heavy-run results are not available yet.

## Source map

`backend/lab_*`: persistence, data validation, chronology, models, accounting, guarded worker, queue, forward paper feed and API. `backend/warehouse*`: Parquet/manifest/DuckDB access and conversion. `backend/provider_budget.py`: shared credits. `frontend/src/Studio*` and `PriceWarehouse.jsx`: app views. Original native engines and specialist tools remain available. Data is local under `data/`, or QF_DATA_DIR when configured before startup. Reports are exported separately from the source-only ZIP; private credentials and downloaded datasets are excluded from that ZIP.
