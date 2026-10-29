"""Local research application API. No credential is returned to the browser."""

import json
import zipfile
from typing import Annotated

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import lab_store
from backend.lab_analysis import schedule
from backend.lab_data import capture_cache, import_prices, manifest, quality
from backend.lab_feed import feed_status, start_refresh, stop_refresh
from backend.lab_models import ResearchConfig, ResourcePolicy
from backend.lab_paper import create_paper, update_paper
from backend.market_data import atomic_json, collection_state
from backend.warehouse import info as warehouse_info
from backend.warehouse_jobs import migration_state

Identity = Annotated[str, Field(pattern=r"^[a-f0-9]{1,32}$")]


class Capture(BaseModel):
    name: str = Field(default="Twelve Data frozen cache", min_length=1, max_length=100)
    security_csv: str = Field(default="", max_length=4_000_000)
    membership_csv: str = Field(default="", max_length=8_000_000)


class Import(Capture):
    csv: str = Field(max_length=32_000_000)
    adjusted_usd: bool = False


class PaperCreate(BaseModel):
    experiment_id: Identity
    name: str = Field(default="Forward paper portfolio", min_length=1, max_length=100)


class PaperUpdate(BaseModel):
    dataset_id: Identity


def guarded(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


def create_router(controller):
    router = APIRouter(prefix="/api/lab", tags=["Research Studio"])

    @router.get("")
    def overview():
        compute = controller.snapshot()
        for key in ("market", "discovery"):
            compute["job"].pop(key, None)
        papers = [
            {key: value for key, value in p.items() if key not in ("ledger", "trades", "config")}
            for p in lab_store.papers()
        ]
        return {
            "defaults": ResearchConfig().model_dump(mode="json"),
            "datasets": lab_store.datasets(),
            "experiments": lab_store.experiments(),
            "paper": papers,
            "collection": collection_state(),
            "compute": compute,
            "resources": lab_store.resource_policy(),
            "paper_refresh": feed_status(),
            "warehouse": {**warehouse_info(), "migration": migration_state()},
        }

    @router.post("/resources")
    def resource_settings(policy: ResourcePolicy):
        return lab_store.save_resources(policy)

    @router.post("/datasets/capture")
    def capture(payload: Capture):
        return guarded(capture_cache, payload.name, payload.security_csv, payload.membership_csv)

    @router.post("/datasets/import")
    def upload(payload: Import):
        return guarded(
            import_prices, payload.name, payload.csv, payload.adjusted_usd, payload.security_csv, payload.membership_csv
        )

    @router.get("/datasets/{identity}/quality")
    def dataset_quality(identity: Identity, q: str = "", limit: int = 100):
        directory = lab_store.LAB / "datasets" / identity
        path = directory / "quality.json"
        if not path.exists():
            result = guarded(quality, identity, 100000)
            atomic_json(path, result)
        result = json.loads(path.read_text(encoding="utf-8"))
        result["assets"] = [r for r in result["assets"] if q.upper() in r["symbol"]][: max(1, min(500, limit))]
        return result

    @router.post("/plan")
    def preview(config: ResearchConfig):
        body = guarded(manifest, config.dataset_id)
        return {
            "folds": guarded(schedule, body["dates"], config),
            "dataset": lab_store.dataset(config.dataset_id),
            "max_validation_rules_per_fold": len(config.entry_grid) * len(config.holding_grid) + 1,
            "note": "Each fold refits on training data only; validation chooses settings, then tests the frozen choice.",
        }

    @router.post("/experiments")
    def create_experiment(config: ResearchConfig):
        if not config.await_snapshot_id:
            preview(config)
        elif collection_state().get("snapshot_id") != config.await_snapshot_id:
            raise HTTPException(400, "The requested collection snapshot is not the current provider queue")
        return lab_store.create(config)

    @router.get("/experiments/{identity}")
    def get_experiment(identity: Identity):
        value = guarded(lab_store.experiment, identity)
        value["report"] = lab_store.report(identity) if value["status"] == "completed" else None
        checkpoint = lab_store.report_path(identity).with_name("checkpoint.json")
        value["completed_checkpoint_folds"] = (
            len(json.loads(checkpoint.read_text())["completed_folds"]) if checkpoint.exists() else 0
        )
        value["audit"] = lab_store.audit_rows(identity)
        return value

    @router.post("/experiments/{identity}/queue")
    def queue_experiment(identity: Identity):
        value = guarded(lab_store.experiment, identity)
        config = ResearchConfig.model_validate(value["config"])
        if not config.await_snapshot_id:
            preview(config)
        guarded(lab_store.queue, identity)
        return lab_store.experiment(identity)

    @router.post("/experiments/{identity}/duplicate")
    def duplicate(identity: Identity):
        value = guarded(lab_store.experiment, identity)
        config = ResearchConfig.model_validate(
            {**value["config"], "experiment_id": "", "name": (value["config"]["name"] + " · copy")[:100]}
        )
        return lab_store.create(config)

    @router.post("/experiments/{identity}/cancel")
    def cancel_experiment(identity: Identity):
        value = guarded(lab_store.experiment, identity)
        with controller.lock:
            if (
                controller.job.get("config", {}).get("experiment_id") == identity
                and controller.job.get("status") == "running"
            ):
                return controller.cancel()
        if value["status"] in ("queued", "waiting_data"):
            lab_store.transition(identity, "cancelled", "Cancelled while queued")
            return {"status": "cancelled"}
        raise HTTPException(400, "This experiment is not queued or running")

    @router.get("/experiments/{identity}/export/{kind}")
    def export(identity: Identity, kind: str):
        value = guarded(lab_store.experiment, identity)
        if value["status"] != "completed":
            raise HTTPException(400, "Complete the experiment before exporting its final report")
        directory = lab_store.report_path(identity).parent
        if kind == "bundle":
            path = directory / "research_bundle.zip"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for candidate in [
                    directory / "report.json",
                    directory / "trades.csv",
                    directory / "folds.csv",
                    directory / "daily_returns.csv",
                    *sorted(directory.glob("candidates_fold_*.csv")),
                ]:
                    if candidate.exists():
                        archive.write(candidate, candidate.name)
            return FileResponse(path, filename=f"quantforge-{identity}.zip")
        names = {"report": "report.json", "trades": "trades.csv", "folds": "folds.csv", "returns": "daily_returns.csv"}
        if kind not in names:
            raise HTTPException(404, "Unknown export")
        return FileResponse(directory / names[kind], filename=f"quantforge-{identity}-{names[kind]}")

    @router.post("/paper")
    def paper_create(payload: PaperCreate):
        return guarded(create_paper, payload.experiment_id, payload.name)

    @router.get("/paper/{identity}")
    def paper_get(identity: Identity):
        return guarded(lab_store.paper, identity)

    @router.post("/paper/{identity}/update")
    def paper_update(identity: Identity, payload: PaperUpdate):
        return guarded(update_paper, identity, payload.dataset_id)

    @router.post("/paper/{identity}/refresh")
    def paper_refresh(identity: Identity):
        return guarded(start_refresh, identity)

    @router.post("/paper-refresh/cancel")
    def paper_refresh_cancel():
        return guarded(stop_refresh)

    return router
