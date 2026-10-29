# Repository verification — 29 October 2026

- Full local Python suite: **68 passed, 1 optional CUDA test skipped**, in 37.30 seconds.
- `ruff check backend tests`: passed.
- React production build: passed (Vite 7.3.7).
- The reusable `backend.trend_research` CLI completed against the original frozen local SPY dataset. Its primary 10-month result reproduced the earlier exploratory calculation: approximately 7.63% annualised versus 14.82% for buy-and-hold over 2017-10-02 through 2026-10-27. These are USD, assumption-based historical results, not Trading 212/GBP execution results or future forecasts.
- New trend tests verify independent buy-and-hold cost accounting, signal/execution separation, immunity to observations after the evaluation end, aligned comparisons, cost stress, and rejection of missing/duplicate/insufficient input.
- The source upload was checked against the actual local Twelve Data credential and common token patterns. Credentials, original market data, local run records, generated dependencies/binaries and screenshots are excluded from the tracked source.

The suite emitted FastAPI/httpx and DuckDB/Arrow deprecation warnings. CUDA compilation and GPU correctness/performance were not verified. The workstation's existing native CPU libraries were present during tests; a fresh checkout builds them with `setup.ps1`. Other operating systems were not tested.

The historical identity, survivorship, cost and model limitations in README.md still apply. Tests validate implementation behaviour; they do not certify data quality or investment performance.
