import time

import psutil
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.config import RunConfig
from backend.main import app, controller


@pytest.fixture
def client(monkeypatch, tmp_path):
    from backend import lab_store

    monkeypatch.setattr(lab_store, "LAB", tmp_path / "lab")
    monkeypatch.setenv("QF_LAB_DIR", str(tmp_path / "lab"))
    monkeypatch.setenv("QF_DISABLE_MARKET_SERVICE", "1")
    with TestClient(app) as client:
        deadline = time.monotonic() + 15
        while not controller.telemetry and time.monotonic() < deadline:
            time.sleep(0.1)
        yield client


def wait_for_terminal(client, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        job = client.get("/api/state").json()["job"]
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    pytest.fail("Worker did not finish")


def test_start_complete_and_cancel(client):
    config = RunConfig(paths=10000, steps=2, threads=2).model_dump()
    assert client.post("/api/start", json=config).status_code == 200
    job = wait_for_terminal(client)
    assert job["status"] == "completed", job
    assert job["results"]["n"] == 10000
    config.update(paths=1_000_000_000, steps=252)
    assert client.post("/api/start", json=config).status_code == 200
    assert client.post("/api/start", json=config).status_code == 409
    deadline = time.monotonic() + 5
    while not controller.job.get("worker_pid") and time.monotonic() < deadline:
        time.sleep(0.02)
    worker_pid = controller.job.get("worker_pid")
    assert worker_pid is not None
    assert controller.worker_rss_gb() > 0.02
    started = time.perf_counter()
    assert client.post("/api/cancel").json()["status"] == "cancelled"
    assert time.perf_counter() - started < 3
    assert controller.process.poll() is not None
    assert not psutil.pid_exists(worker_pid)


def test_resource_and_origin_rejections(client):
    invalid = RunConfig().model_dump()
    invalid.update(ram_gb=224, reserve_gb=128)
    assert client.post("/api/start", json=invalid).status_code == 400
    assert client.post("/api/start", json={"paths": 0}).status_code == 422
    assert client.post("/api/cancel", headers={"Origin": "https://malicious.example"}).status_code == 403
    assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 403


def test_duration_guard(client):
    config = RunConfig(paths=1_000_000_000, steps=252, threads=2, max_seconds=1)
    assert client.post("/api/start", json=config.model_dump()).status_code == 200
    job = wait_for_terminal(client, 12)
    assert job["status"] == "stopped"
    assert "duration" in job["error"]


def test_small_ram_cache_has_real_asset_scenarios(client):
    config = RunConfig(paths=5000, steps=1, threads=2, memory_stress=True)
    assert client.post("/api/start", json=config.model_dump()).status_code == 200
    job = wait_for_terminal(client)
    assert job["status"] == "completed"
    result = job["engines"]["cpu"]
    assert result["cache_gb"] > 0
    assert len(result["retained_sample"][0]) == 5
    assert len(result["cached_asset_mean_ratios"]) == 4
    assert result["cache_paths"] == 5000


def test_required_temperature_missing_rejected(client):
    with controller.lock:
        previous = controller.telemetry.get("cpu_temperature")
        controller.telemetry["cpu_temperature"] = None
        with pytest.raises(HTTPException) as error:
            controller.start(RunConfig(require_cpu_temperature=True))
        controller.telemetry["cpu_temperature"] = previous
    assert error.value.status_code == 400
    assert "temperature required" in error.value.detail


def test_runtime_memory_reserve_stops_worker(client, monkeypatch):
    import psutil

    config = RunConfig(paths=1_000_000_000, steps=252, threads=2)
    assert client.post("/api/start", json=config.model_dump()).status_code == 200
    memory = psutil.virtual_memory()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory._replace(available=0))
    job = wait_for_terminal(client, 5)
    assert job["status"] == "stopped"
    assert "memory reserve" in job["error"]


def test_discovery_csv_end_to_end(client, monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd

    import backend.main as main_module
    from backend.discovery_data import DiscoveryConfig

    monkeypatch.setattr(main_module, "DATA", tmp_path)
    monkeypatch.setenv("QF_DATA_DIR", str(tmp_path))
    rng = np.random.default_rng(78)
    logs = np.cumsum(rng.normal(0, 0.01, (650, 4)), axis=0)
    frame = pd.DataFrame(
        np.exp(logs) * 100,
        columns=["AAA", "BBB", "CCC", "DDD"],
        index=pd.date_range("2020-01-01", periods=650, freq="B"),
    )
    frame.index.name = "Date"
    imported = client.post("/api/research/import", json={"csv": frame.to_csv()})
    assert imported.status_code == 200
    config = DiscoveryConfig(
        source="csv",
        dataset_id=imported.json()["dataset_id"],
        tickers=["AAA", "BBB", "CCC", "DDD"],
        shortlist=3,
        bootstrap_repetitions=50,
        threads=2,
        require_cpu_temperature=False,
        ram_gb=4,
    )
    response = client.post("/api/research/start", json=config.model_dump())
    assert response.status_code == 200
    job = wait_for_terminal(client, 30)
    assert job["status"] == "completed", job.get("error")
    assert job["completed"] == 100
    assert len(job["discovery"]["pairs"]) == 3
    assert job["discovery"]["metadata"]["all_pairs"] == 6
    assert client.get("/api/research/export").status_code == 200


def test_cancel_discovery_terminates_its_process_pool(client, monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd

    import backend.main as main_module
    from backend.discovery_data import DiscoveryConfig

    monkeypatch.setattr(main_module, "DATA", tmp_path)
    monkeypatch.setenv("QF_DATA_DIR", str(tmp_path))
    rng = np.random.default_rng(991)
    frame = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0, 0.01, (650, 20)), axis=0)) * 100,
        columns=["S" + str(i) for i in range(20)],
        index=pd.date_range("2020-01-01", periods=650, freq="B"),
    )
    frame.index.name = "Date"
    imported = client.post("/api/research/import", json={"csv": frame.to_csv()}).json()
    config = DiscoveryConfig(
        source="csv",
        dataset_id=imported["dataset_id"],
        threads=2,
        ram_gb=4,
        shortlist=100,
        bootstrap_repetitions=50,
        require_cpu_temperature=False,
    )
    assert client.post("/api/research/start", json=config.model_dump()).status_code == 200
    deadline = time.monotonic() + 20
    descendants = []
    while time.monotonic() < deadline:
        pid = controller.job.get("worker_pid")
        if pid:
            descendants = psutil.Process(pid).children(recursive=True)
            if descendants:
                break
        time.sleep(0.02)
    assert descendants, "Research process pool never started"
    worker_pid = controller.job["worker_pid"]
    response = client.post("/api/cancel")
    assert response.json()["status"] == "cancelled"
    _, alive = psutil.wait_procs(descendants, timeout=2)
    assert not alive
    assert not psutil.pid_exists(worker_pid)
