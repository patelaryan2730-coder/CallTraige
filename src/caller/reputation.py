"""
Caller Reputation Scorer
--------------------------
Deliberately simple - no ML needed. Checks whether a caller number is known,
trusted, or entirely new, and returns a risk score. This is a cheap, reliable
signal that complements the more complex audio and transcript scorers.

Uses a local SQLite database so state persists across calls, which also
makes it easy to plug straight into the Streamlit dashboard.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).parent / "caller_reputation.db"


def init_db(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS known_callers (
            phone_number TEXT PRIMARY KEY,
            trust_level TEXT NOT NULL,  -- 'trusted', 'seen_before', 'unknown'
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            call_count INTEGER DEFAULT 1
        )
        """
    )
    conn.commit()
    conn.close()


@dataclass
class CallerReputationResult:
    phone_number: str
    trust_level: str
    risk_score: float  # 0 = fully trusted, 1 = maximum risk
    call_count: int


def score_caller(phone_number: str, db_path: Path = DB_PATH) -> CallerReputationResult:
    """
    Looks up a phone number and returns a risk score.
      - Trusted contact (manually added)        -> risk 0.05
      - Seen before, not explicitly trusted      -> risk 0.35
      - Brand new / never seen                   -> risk 0.85
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT trust_level, call_count FROM known_callers WHERE phone_number = ?",
        (phone_number,),
    )
    row = cursor.fetchone()
    now = datetime.now(timezone.utc).isoformat()

    if row is None:
        # Never seen this number before - record it as unknown, high risk.
        conn.execute(
            "INSERT INTO known_callers (phone_number, trust_level, first_seen, last_seen, call_count) "
            "VALUES (?, 'unknown', ?, ?, 1)",
            (phone_number, now, now),
        )
        conn.commit()
        conn.close()
        return CallerReputationResult(phone_number, "unknown", 0.85, 1)

    trust_level, call_count = row
    conn.execute(
        "UPDATE known_callers SET last_seen = ?, call_count = call_count + 1 WHERE phone_number = ?",
        (now, phone_number),
    )
    conn.commit()
    conn.close()

    risk_map = {"trusted": 0.05, "seen_before": 0.35, "unknown": 0.85}
    return CallerReputationResult(
        phone_number, trust_level, risk_map.get(trust_level, 0.85), call_count + 1
    )


def mark_as_trusted(phone_number: str, db_path: Path = DB_PATH):
    """Use this to manually add known/trusted contacts, e.g. known suppliers or staff."""
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO known_callers (phone_number, trust_level, first_seen, last_seen, call_count) "
        "VALUES (?, 'trusted', ?, ?, 0) "
        "ON CONFLICT(phone_number) DO UPDATE SET trust_level = 'trusted'",
        (phone_number, now, now),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    mark_as_trusted("+61400000000")

    print("Trusted number:", score_caller("+61400000000"))
    print("New/unknown number:", score_caller("+61499999999"))
    print("Same unknown number again (now 'seen_before' next call):", score_caller("+61499999999"))
