"""Durable local experiment, dataset and forward-paper records."""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from backend.discovery_data import DATA
from backend.market_data import atomic_json

LAB = Path(os.environ.get("QF_LAB_DIR", str(DATA / "lab")))


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    LAB.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(LAB / "studio.sqlite", timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,body TEXT NOT NULL,created TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY,config TEXT NOT NULL,status TEXT NOT NULL,
        created TEXT NOT NULL,updated TEXT NOT NULL,error TEXT,summary TEXT);
      CREATE TABLE IF NOT EXISTS paper(id TEXT PRIMARY KEY,body TEXT NOT NULL,created TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY,created TEXT,kind TEXT,record_id TEXT,body TEXT);
    """)
    try:
        with db:
            yield db
    finally:
        db.close()


def audit(db, kind, record_id, body):
    db.execute(
        "INSERT INTO audit(created,kind,record_id,body) VALUES (?,?,?,?)", (now(), kind, record_id, json.dumps(body))
    )


def put_dataset(body):
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO datasets VALUES (?,?,?)", (body["id"], json.dumps(body), now()))
        audit(db, "dataset_captured", body["id"], {"source": body["source"], "assets": body["assets"]})
    return body


def datasets():
    with connect() as db:
        return [json.loads(row[0]) for row in db.execute("SELECT body FROM datasets ORDER BY created DESC LIMIT 100")]


def dataset(identity):
    with connect() as db:
        row = db.execute("SELECT body FROM datasets WHERE id=?", (identity,)).fetchone()
    if not row:
        raise ValueError("Dataset not found; capture the cache or import prices first")
    return json.loads(row[0])


def create(config):
    identity = uuid.uuid4().hex[:16]
    body = config.model_copy(update={"experiment_id": identity}).model_dump(mode="json")
    with connect() as db:
        db.execute(
            "INSERT INTO experiments VALUES (?,?,?,?,?,?,?)",
            (identity, json.dumps(body), "draft", now(), now(), None, None),
        )
        audit(db, "experiment_created", identity, body)
    return experiment(identity)


def experiment(identity):
    with connect() as db:
        row = db.execute("SELECT * FROM experiments WHERE id=?", (identity,)).fetchone()
    if not row:
        raise ValueError("Experiment not found")
    body = dict(row)
    body["config"] = json.loads(body["config"])
    body["summary"] = json.loads(body["summary"]) if body["summary"] else None
    return body


def experiments():
    with connect() as db:
        identities = [row[0] for row in db.execute("SELECT id FROM experiments ORDER BY created DESC LIMIT 100")]
    return [experiment(identity) for identity in identities]


def transition(identity, status, error=None, summary=None):
    with connect() as db:
        db.execute(
            "UPDATE experiments SET status=?,updated=?,error=?,summary=COALESCE(?,summary) WHERE id=?",
            (status, now(), error, json.dumps(summary) if summary is not None else None, identity),
        )
        audit(db, "experiment_" + status, identity, {"error": error})


def queue(identity):
    value = experiment(identity)
    target = "waiting_data" if value["config"].get("await_snapshot_id") else "queued"
    with connect() as db:
        changed = db.execute(
            "UPDATE experiments SET status=?,updated=?,error=NULL WHERE id=? AND status IN ('draft','interrupted','cancelled','stopped','failed')",
            (target, now(), identity),
        ).rowcount
        if not changed:
            raise ValueError(
                "This experiment is already queued/running or completed; duplicate a completed experiment to rerun"
            )
        audit(db, "experiment_queued", identity, {})


def waiting_data():
    with connect() as db:
        return [
            experiment(r[0])
            for r in db.execute("SELECT id FROM experiments WHERE status='waiting_data' ORDER BY updated LIMIT 100")
        ]


def attach_dataset(identity, config):
    with connect() as db:
        db.execute(
            "UPDATE experiments SET config=?,status='queued',updated=? WHERE id=? AND status='waiting_data'",
            (config.model_dump_json(), now(), identity),
        )
        audit(db, "full_dataset_attached", identity, {"dataset_id": config.dataset_id})


def resource_policy():
    from backend.lab_models import ResourcePolicy

    path = LAB / "resources.json"
    return ResourcePolicy.model_validate(json.loads(path.read_text()) if path.exists() else {}).model_dump()


def save_resources(policy):
    atomic_json(LAB / "resources.json", policy.model_dump())
    return policy.model_dump()


def next_queued():
    with connect() as db:
        row = db.execute("SELECT id FROM experiments WHERE status='queued' ORDER BY updated,id LIMIT 1").fetchone()
    return row[0] if row else None


def recover():
    with connect() as db:
        db.execute(
            "UPDATE experiments SET status='interrupted',updated=?,error='Server stopped during execution; resume from the last completed fold' WHERE status='running'",
            (now(),),
        )


def report_path(identity):
    # All callers validate IDs, and the route also uses this explicit guard.
    if not identity or any(c not in "0123456789abcdef" for c in identity) or len(identity) > 32:
        raise ValueError("Invalid experiment identifier")
    return LAB / "experiments" / identity / "report.json"


def save_report(identity, report):
    atomic_json(report_path(identity), report)


def report(identity):
    path = report_path(identity)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def papers():
    with connect() as db:
        return [json.loads(r[0]) for r in db.execute("SELECT body FROM paper ORDER BY created DESC")]


def paper(identity):
    with connect() as db:
        row = db.execute("SELECT body FROM paper WHERE id=?", (identity,)).fetchone()
    if not row:
        raise ValueError("Paper portfolio not found")
    return json.loads(row[0])


def save_paper(body, event):
    with connect() as db:
        db.execute(
            "INSERT INTO paper VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body",
            (body["id"], json.dumps(body), body["created"]),
        )
        audit(db, event, body["id"], {"through": body.get("last_date"), "observations": len(body.get("ledger", []))})


def audit_rows(identity):
    with connect() as db:
        return [
            dict(row) for row in db.execute("SELECT * FROM audit WHERE record_id=? ORDER BY id LIMIT 1000", (identity,))
        ]
