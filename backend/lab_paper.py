"""Forward-only synthetic adjusted-close paper ledger, never a broker order service."""

import threading
import uuid
from datetime import datetime

import numpy as np

from backend import lab_store
from backend.lab_data import asof_metadata, load_dataset
from backend.lab_models import ResearchConfig
from backend.lab_portfolio import run_portfolio

LOCK = threading.RLock()


def create_paper(experiment_id, name):
    experiment = lab_store.experiment(experiment_id)
    report = lab_store.report(experiment_id)
    if experiment["status"] != "completed" or not report:
        raise ValueError("Complete a walk-forward experiment before freezing a paper portfolio")
    deployment = report["deployment"]
    if not deployment["holdings"]:
        raise ValueError("The fitted forward allocation has no holdings")
    body = {
        "id": uuid.uuid4().hex[:16],
        "name": name[:100],
        "experiment_id": experiment_id,
        "created": lab_store.now(),
        "start_after": datetime.now().astimezone().date().isoformat(),
        "model_asof": deployment["asof"],
        "holdings": deployment["holdings"],
        "config": report["config"],
        "last_date": None,
        "ledger": [],
        "trades": [],
        "status": "awaiting_forward_prices",
        "note": "Frozen long-only adjusted-total-return paper experiment. Synthetic units, no real orders. Only completed observations strictly after creation are eligible.",
    }
    lab_store.save_paper(body, "paper_created")
    return body


def update_paper(identity, dataset_id):
    with LOCK:
        paper = lab_store.paper(identity)
        body, dates, logs, _ = load_dataset(dataset_id)
        future = np.flatnonzero(dates > paper["start_after"])
        if not len(future) or (
            paper.get("last_dataset_id") == dataset_id
            and paper["last_date"]
            and dates[future[-1]] <= paper["last_date"]
        ):
            return {
                **paper,
                "update_note": "No new completed post-creation observations. Historical backtests cannot populate a forward ledger.",
            }
        if paper["ledger"] and dates[future[0]] != paper["ledger"][0]["date"]:
            raise ValueError("Supply the complete forward interval, including existing ledger dates, to verify history")
        if paper["last_date"] and dates[future[-1]] < paper["last_date"]:
            raise ValueError("The dataset ends before the existing paper ledger; supply the complete interval")
        selected = []
        for holding in paper["holdings"]:
            choices = [i for i, row in enumerate(body["records"]) if row["symbol"] == holding["symbol"]]
            if len(choices) != 1:
                raise ValueError(f"Paper update needs one unambiguous history for {holding['symbol']}")
            selected.append(choices[0])
        config = ResearchConfig.model_validate(paper["config"])
        report = lab_store.report(paper["experiment_id"])
        deployment = report["deployment"]
        source = deployment.get("cost_inputs", {})
        model = {
            "weights": np.array([h["weight"] for h in paper["holdings"]]),
            "turnover": np.array(
                source.get("turnover", [max(config.minimum_turnover, 1e6)] * len(selected)), dtype=float
            ),
            "volatility": np.array(source.get("volatility", [0.3] * len(selected)), dtype=float),
            "attributes": source.get(
                "attributes", [asof_metadata(body, h["symbol"], paper["start_after"]) for h in paper["holdings"]]
            ),
        }
        rows = [body["records"][i] for i in selected]
        result = run_portfolio(
            np.asarray(logs[selected]),
            dates,
            rows,
            model,
            int(future[0]),
            int(future[-1]),
            config,
            close_at_end=False,
            dated_metadata=body,
        )
        if result["status"] != "evaluated":
            raise ValueError(result["reason"] + "; paper history was not changed")
        # A consistent adjusted-price rebase can change synthetic units, but cannot
        # rewrite already recorded NAV, costs or dates. Noncoherent revisions fail.
        for old, new in zip(paper["ledger"], result["trace"]):
            if old["date"] != new["date"] or abs(old["equity"] - new["equity"]) > 1e-8 * max(1, abs(old["equity"])):
                raise ValueError(
                    "Revised data or costs change existing paper history; ledger remains unchanged. Review input adjustments."
                )
        cutoff = paper["last_date"] or ""
        additions = [row for row in result["trace"] if row["date"] > cutoff]
        paper["ledger"].extend(additions)
        paper["trades"].extend(row for row in result["trades"] if row["date"] > cutoff)
        paper.update(
            last_date=paper["ledger"][-1]["date"],
            status="active",
            metrics=result["metrics"],
            last_dataset_id=dataset_id,
            fees_fraction=result["fees_fraction"],
        )
        lab_store.save_paper(paper, "paper_observations_appended" if additions else "paper_data_verified")
        return paper
