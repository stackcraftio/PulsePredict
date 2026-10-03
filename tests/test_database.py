import sqlite3

from src.database import get_recent, get_stats, init_db, log_prediction


def test_log_and_stats(temp_db):
    init_db()
    assert get_stats()["predictions_made"] == 0
    log_prediction(1, 0.8, "1.0")
    log_prediction(0, 0.2, "1.0")
    stats = get_stats()
    assert stats["predictions_made"] == 2
    assert stats["higher_risk_classifications"] == 1
    assert abs(stats["average_probability"] - 0.5) < 1e-9


def test_recent_returns_newest_first(temp_db):
    log_prediction(0, 0.1, "1.0")
    log_prediction(1, 0.9, "1.0")
    rows = get_recent(5)
    assert rows[0]["probability"] == 0.9


def test_table_rejects_out_of_range_probability(temp_db):
    init_db()
    conn = sqlite3.connect(temp_db)
    try:
        conn.execute("INSERT INTO predictions (timestamp, prediction, probability, model_version) "
                     "VALUES ('t', 1, 1.5, '1.0')")
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    finally:
        conn.close()
    assert raised
