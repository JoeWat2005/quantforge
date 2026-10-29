from datetime import date

import numpy as np
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import provider_budget, warehouse, warehouse_api


@pytest.fixture
def lake(monkeypatch, tmp_path):
    monkeypatch.setattr(warehouse, "WAREHOUSE", tmp_path / "warehouse")
    monkeypatch.setattr(warehouse_api, "WAREHOUSE", warehouse.WAREHOUSE)
    with warehouse.catalog() as db:
        identity = warehouse.register(db, {"symbol": "TEST", "currency": "USD", "exchange": "NYSE"}, "equity")
    return identity


def source(identity, key="old", close=(100.0, 101.0), quality=0):
    item = dict(
        id=identity,
        asset_class="equity",
        dates=np.array(["2020-01-02", "2020-01-03"]),
        close=np.array(close),
        volume=np.array([1000.0, 2000.0]),
        source_key=key,
        sha256=key,
        quality=quality,
    )
    if quality:
        item.update(open=np.array(close) - 1, high=np.array(close) + 1, low=np.array(close) - 2)
    return item


def test_versions_pinned_snapshots_and_nullable_legacy_prices(lake):
    assert warehouse.store_series_batch([source(lake)]) == 2
    with warehouse.catalog() as db:
        pinned = {row[0] for row in db.execute("SELECT id FROM price_files")}
    legacy = warehouse.query_prices([lake]).to_pylist()
    assert legacy[0]["open"] is None
    warehouse.store_series_batch([source(lake, "new", (110.0, 111.0), 1)])
    latest = warehouse.query_prices([lake]).to_pylist()
    assert len(latest) == 2
    assert latest[0]["close"] == 110
    assert latest[0]["open"] == 109
    frozen = warehouse.query_prices([lake], pinned_files=pinned).to_pylist()
    assert frozen == legacy
    original = warehouse.query_prices([lake], source_hashes=["old"]).to_pylist()
    assert original == legacy
    assert warehouse.query_prices([lake], date(2020, 1, 3), date(2020, 1, 3)).num_rows == 1
    assert warehouse.query_prices(["f" * 24]).num_rows == 0
    with pytest.raises(ValueError, match="Unknown price column"):
        warehouse.query_prices([lake], columns=["close; DROP TABLE prices"])


def test_invalid_series_never_published(lake):
    invalid = source(lake)
    invalid["dates"] = np.array(["2020-01-02", "2020-01-02"])
    with pytest.raises(ValueError, match="Duplicate"):
        warehouse.store_series_batch([invalid])
    assert warehouse.info()["source_series"] == 0
    invalid = source(lake, close=(-1.0, 100.0))
    with pytest.raises(ValueError, match="positive"):
        warehouse.store_series_batch([invalid])
    assert warehouse.info()["parquet_files"] == 0


def test_search_api_dates_and_real_parquet_export(lake):
    warehouse.store_series_batch([source(lake)])
    app = FastAPI()
    app.include_router(warehouse_api.router)
    with TestClient(app) as client:
        assert client.get("/api/warehouse/instruments?q=TEST").json()[0]["id"] == lake
        data = client.get("/api/warehouse/prices/" + lake).json()
        assert len(data["bars"]) == 2
        assert data["bars"][0]["date"] == "2020-01-02"
        assert client.get("/api/warehouse/prices/" + lake + "?start=2021-01-01&end=2020-01-01").status_code == 400
        assert client.get("/api/warehouse/prices/not-an-id").status_code == 422
        response = client.get("/api/warehouse/prices/" + lake + "/parquet?start=2020-01-03")
        assert response.status_code == 200
        assert response.content[:4] == b"PAR1"
        import io

        table = pq.read_table(io.BytesIO(response.content))
        assert table.num_rows == 1
        assert table.column("close")[0].as_py() == 101


def test_shared_provider_budget_records_spend_and_rejects_overquota(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_budget, "DATA", tmp_path)
    provider_budget.configure(55)
    provider_budget.acquire(1)
    db = provider_budget.connection()
    try:
        assert db.execute("SELECT SUM(cost) FROM spent").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0
    finally:
        db.close()
    with pytest.raises(ValueError, match="exceeds"):
        provider_budget.acquire(100)
