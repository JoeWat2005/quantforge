"""A process-safe FIFO credit budget shared by every Twelve Data downloader."""

import sqlite3
import time
import uuid

import psutil

from backend.discovery_data import DATA


def connection():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA / "provider_budget.sqlite", timeout=15, isolation_level=None)
    db.executescript("""
      PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS quota(id INTEGER PRIMARY KEY CHECK(id=1),credits INTEGER,next_slot REAL);
      INSERT OR IGNORE INTO quota VALUES(1,8,0);
      CREATE TABLE IF NOT EXISTS tickets(id INTEGER PRIMARY KEY AUTOINCREMENT,token TEXT UNIQUE,pid INTEGER,created REAL,cost INTEGER);
      CREATE TABLE IF NOT EXISTS spent(at REAL,cost INTEGER);
    """)
    return db


def configure(credits):
    db = connection()
    try:
        db.execute("UPDATE quota SET credits=? WHERE id=1", (max(1, int(credits)),))
    finally:
        db.close()


def acquire(cost=1):
    if cost < 1:
        raise ValueError("Provider credit cost must be positive")
    db = connection()
    token = uuid.uuid4().hex
    process = psutil.Process()
    try:
        budget = max(1, int(db.execute("SELECT credits FROM quota").fetchone()[0] * 0.9))
        if cost > budget:
            raise ValueError(
                f"Endpoint costs {cost} credits; it exceeds the configured {budget}-credit safe minute budget"
            )
        db.execute(
            "INSERT INTO tickets(token,pid,created,cost) VALUES(?,?,?,?)",
            (token, process.pid, process.create_time(), cost),
        )
        while True:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            db.execute("DELETE FROM spent WHERE at<=?", (now - 60.1,))
            for ticket, pid, created in db.execute(
                "SELECT id,pid,created FROM tickets ORDER BY id LIMIT 16"
            ).fetchall():
                try:
                    alive = abs(psutil.Process(pid).create_time() - created) < 0.1
                except psutil.NoSuchProcess:
                    alive = False
                if not alive:
                    db.execute("DELETE FROM tickets WHERE id=?", (ticket,))
            head = db.execute("SELECT token FROM tickets ORDER BY id LIMIT 1").fetchone()
            credits, next_slot = db.execute("SELECT credits,next_slot FROM quota").fetchone()
            budget = max(1, int(credits * 0.9))
            if cost > budget:
                db.execute("ROLLBACK")
                raise ValueError("The updated account quota cannot accommodate this endpoint")
            spent = db.execute("SELECT COALESCE(SUM(cost),0) FROM spent").fetchone()[0]
            if head and head[0] == token and spent + cost <= budget and now >= next_slot:
                db.execute("INSERT INTO spent VALUES(?,?)", (now, cost))
                db.execute("DELETE FROM tickets WHERE token=?", (token,))
                db.execute("UPDATE quota SET next_slot=?", (now + 60 * cost / budget,))
                db.execute("COMMIT")
                return
            db.execute("COMMIT")
            time.sleep(0.2)
    finally:
        if db.in_transaction:
            db.execute("ROLLBACK")
        db.execute("DELETE FROM tickets WHERE token=?", (token,))
        db.close()
