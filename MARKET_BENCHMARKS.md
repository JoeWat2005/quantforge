# Measured US-market preview — not a full-market benchmark

Measured on the user's Windows Threadripper PRO 5995WX, 64 physical / 128 logical processors, approximately 256 GiB RAM. Real Twelve Data cached adjusted prices; no synthetic performance claims. The catalogue collector continued during this run. Another development build also ran during part of the observed interval, so these are end-to-end development measurements, not an isolated scaling study.

| Measurement | Observed |
|---|---:|
| Provider catalogue | 31,957 instruments |
| Histories available when preview started | 1,927 |
| Eligible for raw correlation analysis | 1,441 |
| Exchange-traded, liquid, 95%-coverage joint core | 538 |
| All possible eligible pairs | 1,037,520 |
| Pairs meeting observed-overlap requirements | 1,025,768 |
| Baskets attempted / successful fits | 10,000 / 10,000 |
| Material mixed-sign baskets with non-full nominal rank | 964 |
| Requested numerical-library thread ceiling | 64 |
| Independent basket processes | 61 |
| Worker computation including loading and reporting | 15.590 seconds |
| Client-observed start-to-terminal time | 18.272 seconds |
| Maximum sampled whole-system CPU | 62.90% |
| Maximum sampled hottest CPU sensor | 78.06°C |
| Maximum sampled conservative process-tree RSS | 9.401 GiB |
| Real process-pool cancellation response | 0.0409 seconds |

Training: 2016-10-28 through 2024-10-23. Later evaluation: 2024-10-24 through 2026-10-27. Off-calendar bars excluded: 2,389. Raw correlations use pairwise-complete observations; basket tests use complete consecutive segments, with their own actual dates. The initial cache is priority ordered and incomplete, so its results are not representative of the whole US market.

For cancellation, a separate 200,000-basket job reached its process-pool stage. The stop request returned in the measured interval, and the worker and all 10 observed descendants were confirmed stopped. No overheating or actual memory exhaustion was induced. The independent watchdog remains active.

These short CPU workloads did not demonstrate sustained 100% CPU, 64 GiB allocation, full catalogue throughput, overnight stability, or GPU acceleration. GPU acceleration is absent from this workspace. The complete catalogue collection and 100,000-basket automatic full run remain pending. Measurements are sampled, may miss brief peaks, and should not be extrapolated linearly to the complete market.

Raw measurements: MARKET_VALIDATION.json. Bounded report: MARKET_REPORT.json. Successful basket fits: MARKET_BASKETS.csv. Strong retained raw neighbors: MARKET_RELATIONSHIPS.csv. Full float32 matrix and row order remain under data/market/analysis/2e012f05a04f73ff6ef42090/ locally. Matrix dimensions: 1441 × 1441, row major; instrument order is instruments.json. Failed or unsupported histories are excluded with counts in collection status.

Algorithm source SHA256: `90eb274eaf037454570507f6dfb62c4e4b76962a2d5938a9f179a408cd643918`. Snapshot: `277bd51bd6164325995c83d2`. The report records the seed and complete analysis configuration. Reproduction requires the same cached input snapshot and compatible dependency versions; provider revisions or a later cohort can change results.
