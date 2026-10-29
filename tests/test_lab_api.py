import io
import time
import zipfile

import psutil
import pytest
from fastapi.testclient import TestClient
from test_lab import synthetic_input

from backend import lab_store
from backend.lab_models import ResearchConfig
from backend.main import app, controller
from backend.telemetry import Telemetry


@pytest.fixture
def studio_client(monkeypatch, tmp_path):
    target = tmp_path / "lab"
    monkeypatch.setattr(lab_store, "LAB", target)
    monkeypatch.setenv("QF_LAB_DIR", str(target))
    monkeypatch.setenv("QF_DISABLE_MARKET_SERVICE", "1")
    monkeypatch.setattr(
        Telemetry,
        "read",
        lambda self: {
            "cpu_temperature": 60.0,
            "gpu": None,
            "ram_available_gb": 64.0,
            "cpu_percent": 10.0,
            "timestamp": time.time(),
        },
    )
    with TestClient(app) as client:
        deadline = time.monotonic() + 5
        while controller.telemetry.get("cpu_temperature") != 60 and time.monotonic() < deadline:
            time.sleep(0.02)
        yield client


def make_experiment(client):
    response = client.post(
        "/api/lab/datasets/import",
        json={"name": "Synthetic API verification", "csv": synthetic_input(), "adjusted_usd": True},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    config = ResearchConfig(
        dataset_id=data["id"],
        name="Chronological API integration",
        split_mode="manual",
        manual_dates={
            "train_start": "2016-01-04",
            "train_end": "2018-12-31",
            "validation_start": "2019-02-01",
            "validation_end": "2019-12-31",
            "test_start": "2020-02-03",
            "test_end": "2020-12-31",
        },
        max_assets=12,
        basket_trials=30,
        threads=2,
        ram_gb=4,
        max_seconds=120,
        bootstrap_samples=100,
        factor_mode="market",
    )
    response = client.post("/api/lab/experiments", json=config.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    return data, response.json()


def wait_terminal(client, seconds=40):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = client.get("/api/state").json()["job"]
        if value["status"] != "running":
            return value
        time.sleep(0.05)
    pytest.fail("Research worker timed out")


def test_saved_experiment_end_to_end_exports_and_checkpoint(studio_client):
    client = studio_client
    dataset, experiment = make_experiment(client)
    identity = experiment["id"]
    plan = client.post("/api/lab/plan", json=experiment["config"])
    assert plan.status_code == 200
    assert plan.json()["folds"][0]["test_start"] == "2020-02-03"
    assert client.post(f"/api/lab/experiments/{identity}/queue", json={}).status_code == 200
    lab_store.transition(identity, "running")
    controller.start(ResearchConfig.model_validate(experiment["config"]), module="backend.lab_worker")
    job = wait_terminal(client)
    assert job["status"] == "completed", job.get("error")
    result = client.get(f"/api/lab/experiments/{identity}").json()
    assert result["status"] == "completed"
    assert len(result["report"]["folds"]) == 1
    assert result["report"]["input_sha256"] == dataset["input_sha256"]
    assert result["report"]["aggregate"]["selected"]["metrics"]["status"] == "evaluated"
    assert client.post(f"/api/lab/experiments/{identity}/queue", json={}).status_code == 400
    exports = {
        kind: client.get(f"/api/lab/experiments/{identity}/export/{kind}")
        for kind in ["report", "trades", "folds", "returns", "bundle"]
    }
    assert all(response.status_code == 200 for response in exports.values())
    with zipfile.ZipFile(io.BytesIO(exports["bundle"].content)) as archive:
        assert archive.testzip() is None
        assert "daily_returns.csv" in archive.namelist()
        assert all(".secrets" not in name and ".key" not in name for name in archive.namelist())
    # Same frozen source/config can reuse the completed fold checkpoint exactly.
    lab_store.transition(identity, "running")
    controller.start(ResearchConfig.model_validate(experiment["config"]), module="backend.lab_worker")
    assert wait_terminal(client)["status"] == "completed"
    again = client.get(f"/api/lab/experiments/{identity}").json()["report"]
    assert again["resumed_folds"] == 1
    assert again["aggregate"]["selected"]["metrics"] == result["report"]["aggregate"]["selected"]["metrics"]


def test_research_cancellation_persists_terminal_status(studio_client):
    client = studio_client
    _, experiment = make_experiment(client)
    config = ResearchConfig.model_validate({**experiment["config"], "basket_trials": 100000})
    lab_store.transition(experiment["id"], "running")
    controller.start(config, module="backend.lab_worker")
    deadline = time.monotonic() + 10
    while not controller.job.get("worker_pid") and time.monotonic() < deadline:
        time.sleep(0.02)
    pid = controller.job.get("worker_pid")
    assert pid
    response = client.post(f"/api/lab/experiments/{experiment['id']}/cancel", json={})
    assert response.status_code == 200
    assert not psutil.pid_exists(pid)
    assert lab_store.experiment(experiment["id"])["status"] == "cancelled"


def test_resource_limits_and_identifier_validation(studio_client):
    client = studio_client
    assert client.post("/api/lab/resources", json={"cpu_stop": 95}).status_code == 422
    assert client.get("/api/lab/experiments/not-an-id").status_code == 422
    assert (
        client.post(
            "/api/lab/datasets/import", json={"name": "x", "csv": synthetic_input(), "adjusted_usd": False}
        ).status_code
        == 400
    )
    assert client.post("/api/lab/plan", json={"dataset_id": "abc"}).status_code == 400
