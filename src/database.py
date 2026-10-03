"""SQLite logging of predictions. Deliberately privacy-conscious: only the model output is
stored (no inputs, no identifiers)."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.config import DEFAULT_DB_PATH, RISK_THRESHOLD

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    prediction    INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    probability   REAL    NOT NULL CHECK (probability BETWEEN 0 AND 1),
    model_version TEXT    NOT NULL
)
"""


def get_db_path() -> Path:
    """Read at call time so tests (and deployments) can redirect via PULSEPREDICT_DB."""
    return Path(os.environ.get("PULSEPREDICT_DB", DEFAULT_DB_PATH))


def _connect() -> sqlite3.Connection:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn


def init_db() -> None:
    with closing(_connect()):
        pass


def log_prediction(prediction: int, probability: float, model_version: str) -> int:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as conn:
        cur = conn.execute(
            "INSERT INTO predictions (timestamp, prediction, probability, model_version) VALUES (?, ?, ?, ?)",
            (ts, int(prediction), float(probability), model_version),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_stats() -> dict:
    with closing(_connect()) as conn:
        total, avg_prob, higher = conn.execute(
            "SELECT COUNT(*), AVG(probability), "
            "COALESCE(SUM(CASE WHEN probability >= ? THEN 1 ELSE 0 END), 0) FROM predictions",
            (RISK_THRESHOLD,),
        ).fetchone()
    return {
        "predictions_made": int(total),
        "average_probability": round(float(avg_prob), 4) if avg_prob is not None else 0.0,
        "higher_risk_classifications": int(higher),
    }


def get_recent(limit: int = 10) -> list[dict]:
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, timestamp, prediction, probability, model_version "
            "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
