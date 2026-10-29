# US stock price warehouse

Open http://127.0.0.1:8000/#research and choose **Price warehouse**. Search a stock, choose dates, apply, and export a genuine `.parquet` file. Use **Data library → Freeze current cache** to create an immutable research snapshot. A snapshot includes only committed converted histories; subsequent downloads cannot change it.

## What is stored

Collection is restricted to the existing Twelve Data US equity catalogue: common and preferred stocks, ADRs, REITs and other equity security types, including OTC. This is the provider's present-day catalogue, not certified historical US market membership. Unavailable symbols, new listings, stale instruments and incomplete histories remain explicit. No forex or crypto collection is running. Existing ETF/fund histories remain intact in a separate partition; required ETFs supply market/sector factors. Pending non-reference ETF/fund downloads are deferred.

The requested daily history is 2016-10-28 through 2026-10-27, with end date exclusive 2026-10-28. The current download retains provider-adjusted OHLCV. Older cached files supplied close and volume only: their open/high/low fields are NULL. Prices are not fabricated or forward-filled. Prices and volume are provider data; adjustment semantics and anomalies require review before an investment decision.

## Storage and access

- `data/warehouse/prices/asset_class=equity/bucket=…/year=…/*.parquet`: immutable Zstandard-compressed columnar price files, with statistics and instrument/date ordering.
- `data/warehouse/manifest.sqlite`: transactional file manifests, identities, source hashes and coverage. IDs distinguish symbol, exchange and MIC rather than assuming a ticker uniquely identifies a security.
- `data/warehouse/migration.json`: conversion status. Original `data/market` inputs are preserved for recovery.
- `data/lab/datasets/…`: frozen input manifests pin file hashes and metadata. Completed experiments retain their original inputs even as new files are published.

DuckDB reads only registered committed files and requested columns/date partitions. Raw files may contain multiple versions: **do not simply concatenate the Parquet glob for research**. The application resolves instrument/date/adjustment versions by OHLCV quality, ingestion time and version ID. Frozen manifests select their original file set. The warehouse is a collection of Parquet files with a SQLite catalogue, rather than one enormous constantly rewritten file.

From the project environment, a bounded direct query is:

```python
from backend.warehouse import search, query_prices
stock = next(x for x in search("AAPL", "equity") if x["symbol"] == "AAPL")
table = query_prices([stock["id"]], "2020-01-01", "2026-10-27")
```

The API offers `/api/warehouse`, `/api/warehouse/instruments`, `/api/warehouse/prices/{id}` and `/api/warehouse/prices/{id}/parquet`. Dates are inclusive in price queries. Browser queries/exports are capped at 10,000 daily rows per instrument; exports retain the latest 50 small files. Arbitrary SQL is not exposed over HTTP.

## Operations and limits

Keep the local server running. Collection and conversion resume independently; pause controls preserve committed data. Conversion checks a 24 GiB OS reserve, 4 GiB converter RSS and 10 GiB free disk reserve before each batch. DuckDB queries use up to eight threads and a 4 GB memory limit. Analyses have separate resource guards and immediate process-tree cancellation.

Every provider client shares a process-safe credit budget. The verified 55-credit/minute account is conservatively paced at 49 credits/minute. Downloading is mainly a network/quota task, so low CPU usage during collection is expected. The app does not upgrade the account or purchase extra access.

## Measured check

During verification, 20,004,312 rows from 13,281 source histories occupied 219,947,282 compressed bytes. Of these, 14,027,827 rows / 8,392 histories were equities; the remainder were previously cached funds. These are a timestamped **incomplete** snapshot, not final download totals; see WAREHOUSE_STATUS.json.

A single AAPL range query returned 1,714 bars in 0.0326 seconds in a direct local check. A frozen 7,880-instrument × 2,512-session price/volume panel loaded in 7.65 seconds. Dashboard AAPL retrieval separately measured 48 ms for 2,512 bars. These are individual warm/local observations, not repeatable throughput guarantees or CPU/GPU comparisons. Future storage growth and cache state affect timings.
