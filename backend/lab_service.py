"""One durable research queue, sharing the existing compute and thermal controller."""

import csv
import io

from backend import lab_store
from backend.lab_analysis import schedule
from backend.lab_data import capture_cache, manifest
from backend.lab_models import ResearchConfig
from backend.market_data import collection_state


def metadata_text(rows):
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def watch_research(controller):
    lab_store.recover()
    while not controller.stop.wait(1):
        try:
            state = collection_state()
            if state.get("status") == "completed":
                # Freeze the full scope only after every successful download is published in Parquet.
                from backend.warehouse import catalog

                with catalog() as db:
                    published = db.execute(
                        "SELECT COUNT(*) FROM legacy_map m JOIN sources s ON s.source_key=('us_npz|' || m.snapshot || '|' || m.legacy_id) WHERE m.snapshot=?",
                        (state.get("snapshot_id"),),
                    ).fetchone()[0]
                if published < state.get("completed", 0):
                    controller.telemetry["research_queue_note"] = (
                        "Waiting for the final downloaded histories to finish Parquet conversion"
                    )
                    continue
                for waiting in lab_store.waiting_data():
                    if waiting["config"].get("await_snapshot_id") == state.get("snapshot_id"):
                        previous = (
                            manifest(waiting["config"]["dataset_id"]) if waiting["config"].get("dataset_id") else {}
                        )
                        if previous.get("security") or previous.get("membership"):
                            data = capture_cache(
                                "Completed US provider catalogue",
                                metadata_text(previous.get("security", [])),
                                metadata_text(previous.get("membership", [])),
                            )
                        else:
                            data = capture_cache("Completed US provider catalogue")
                        config = ResearchConfig.model_validate(
                            {**waiting["config"], "dataset_id": data["id"], "await_snapshot_id": ""}
                        )
                        schedule(manifest(data["id"])["dates"], config)
                        lab_store.attach_dataset(waiting["id"], config)
            identity = lab_store.next_queued()
            if not identity:
                continue
            with controller.lock:
                if controller.process and controller.process.poll() is None:
                    continue
                temperature = controller.telemetry.get("cpu_temperature")
                if temperature is None or temperature >= 80:
                    controller.telemetry["research_queue_note"] = (
                        "Waiting for a readable CPU sensor below 80°C before dispatch"
                    )
                    continue
                config = ResearchConfig.model_validate(lab_store.experiment(identity)["config"])
                try:
                    controller.start(config, module="backend.lab_worker")
                except Exception as exc:
                    # A RAM reservation or initializing sensor may clear. Keep the queue visible.
                    controller.telemetry["research_queue_note"] = str(getattr(exc, "detail", exc))[:300]
                    continue
                lab_store.transition(identity, "running")
                controller.telemetry.pop("research_queue_note", None)
        except Exception as exc:
            with controller.lock:
                controller.telemetry["research_queue_note"] = str(exc)[:300]


def finish_job(job):
    if job.get("task") != "research" or job.get("status") == "running":
        return
    identity = job.get("config", {}).get("experiment_id")
    if not identity:
        return
    try:
        previous = lab_store.experiment(identity)
        if previous["status"] == job["status"]:
            return
        summary = job.get("lab_summary")
        lab_store.transition(identity, job["status"], job.get("error"), summary)
    except ValueError:
        pass
