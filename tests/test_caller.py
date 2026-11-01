"""Unit tests for caller reputation (src/caller/reputation.py) using a disposable database.
The real table is known_callers(phone_number, trust_level, first_seen, last_seen, call_count), so the test
database is created with reputation.init_db() instead of a hand-written schema."""
import pytest
import reputation
from reputation import score_caller, mark_as_trusted, init_db

KNOWN_NUMBER = "+61400000000"
UNKNOWN_NUMBER = "+61499999999"


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "test_caller_reputation.db"
    init_db(path)
    mark_as_trusted(KNOWN_NUMBER, path)
    return path


def test_known_caller_scores_lower_risk(db):
    known = score_caller(KNOWN_NUMBER, db_path=db).risk_score      # returns a result object: use .risk_score
    unknown = score_caller(UNKNOWN_NUMBER, db_path=db).risk_score
    assert known < unknown


def test_score_is_in_valid_range(db):
    score = score_caller(UNKNOWN_NUMBER, db_path=db).risk_score
    assert 0.0 <= score <= 1.0
