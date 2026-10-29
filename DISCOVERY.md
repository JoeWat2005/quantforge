# US asset-discovery lab

Open http://127.0.0.1:8000/#discovery. The original portfolio simulator remains at /#simulator.

The lab looks for historical relationships and possible diversification candidates using actual adjusted closing prices. It does not estimate a stock's intrinsic value or predict which stock will rise.

1. Choose the US ticker universe. The built-in example contains 102 stocks and ETFs; it is editable and is not a recommended portfolio.
2. Choose history length and the fraction reserved for training.
3. Use Explore for an initial scan or Deep for more statistical work.
4. Click Discover relationships. Downloading prices can take longer than subsequent cached runs.
5. Inspect Related pairs, Diversifiers and Data & methods. Click a pair to see its held-out strategy curve and diagnostics.
6. Export the full pair table as CSV. Larger scans display the first 250 training-ranked rows while CSV includes every tested pair.

Explore tests up to 50 pairs with 250 bootstrap repetitions each. Deep tests up to 1,000 with 2,000 repetitions. Extreme tests up to 5,000 with 10,000 repetitions. The pair count is capped by the actual universe; 102 assets have 5,151 possible pairs. Computational expense grows with training observations × tested pairs × repetitions. Adding observations and distinct assets broadens the research question; endlessly increasing repetitions on one dataset has diminishing value.

Related pairs ranks by training daily-return correlation. Rolling correlations and block-bootstrap intervals show historical stability. Engle–Granger tests examine log-price cointegration, assuming I(1) series; the screen does not establish that assumption. Bonferroni p-value adjustment includes every possible universe pair. Pointwise bootstrap intervals are descriptive and are not corrected for shortlist selection or simultaneous coverage.

The later evaluation interval is excluded from pair selection and coefficient fitting. A fixed exploratory pair strategy is evaluated there with transaction and short-borrow costs. It uses training spread statistics, signals applied to the following interval, and fixed adjusted-price units between entry and exit. Very highly correlated funds can still lose money in that experiment after costs.

Diversifiers ranks assets by their training correlation to the chosen anchor, with later-period correlation and stress-day correlation shown separately. Low historical correlation alone does not imply positive future returns, suitability or a complete asset allocation.

Present-day universe selection, missing delisted assets, provider adjustments, execution assumptions and repeated tuning can bias results. The data panel records provenance, failed downloads, sparse-data exclusions, time boundaries and an input hash. No current-day incomplete provider session is included; missing prices are never forward filled.

The same software guards as the simulator enforce configured RAM, duration and temperature thresholds and cancel the entire compute process tree. A CPU sensor is required by default. The bootstrap uses native C++/OpenMP; cointegration uses up to 16 subprocess workers for bounded memory. GPU research is not implemented.

Yahoo chart data uses an unofficial endpoint, subject to provider availability and terms. Cached responses are reused for 24 hours, with no repeated retry after a rate-limit response. CSV import supports independently obtained adjusted-price data: Date followed by three or more USD asset columns and at least 504 overlapping dates. Imports stay on this PC.

## Measured verification

See DISCOVERY_VALIDATION.json and DISCOVERY_DEEP_VALIDATION.json for full parameters, provenance and measurements.

The real-data validation used 102 assets and 1,254 complete dates from 2021-10-28 through 2026-10-27. Training ended 2025-04-28; later evaluation began 2025-04-29. Every one of the 102 requested assets loaded in that run.

A cached Deep run tested 1,000 pairs with 2,000 bootstrap repetitions, processing 1,752,000,000 resampled return observations. Its measured research wall time was 7.6303 seconds, peak sampled worker-tree RSS 2.51584 GiB, peak sampled system CPU 43.11%, and peak sampled CPU temperature 72.91°C. These are one local run's observations, not isolated peak hardware performance. Timing starts after master-worker Python imports. It includes native library loading, cached data loading, analysis, cointegration subprocess startup and CSV preparation; final JSON serialization is excluded. Sampling can miss transient peaks. Requested worker count does not guarantee full Windows processor-group utilization.

The default 100-pair/500-repetition cached validation took 4.2814 seconds, with 2.49304 GiB peak sampled worker-tree RSS. The earlier uncached run took about 20.4 seconds. Neither a 1,000-asset universe nor the Extreme preset was benchmarked.

The expanded suite passed 23 tests, with one optional CUDA test skipped. New checks cover native bootstrap reproducibility across thread counts, a known correlated input, training/holdout separation, detection of a synthetic cointegrated spread, delayed signal execution, CSV validation, grouping redundant exposures and a full CSV-to-report API run. Synthetic data is restricted to tests; the delivered research results use real provider data.

Primary references: [statsmodels Engle–Granger test](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html), [Gatev et al. pairs-trading research](https://www.nber.org/papers/w7032), [Bailey et al. on backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
