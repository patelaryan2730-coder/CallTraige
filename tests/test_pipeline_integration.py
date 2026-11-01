"""Integration test: the real chain through run_pipeline.py (validation, wav2vec2, Whisper, caller, fusion, logging).
The pipeline has no db_path argument, so the two databases are redirected to a temporary folder instead.
The pipeline is run once and the three tests inspect the result."""
import sqlite3
import pytest
from pathlib import Path
from conftest import FIXTURES

TEST_CLIP = str(FIXTURES / "clean_clip.flac")
TEST_CALLER = "+61400000000"


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    import app, reputation
    from model import AudioSpoofClassifier
    from run_pipeline import run_pipeline

    folder = tmp_path_factory.mktemp("dbs")
    call_log_db = str(folder / "call_log.db")
    caller_db = Path(folder / "caller_reputation.db")

    mp = pytest.MonkeyPatch()
    mp.setattr(app, "DB_PATH", call_log_db)                           # dashboard call_log database
    mp.setattr(reputation.score_caller, "__defaults__", (caller_db,))  # db_path default argument
    mp.setattr(reputation.init_db, "__defaults__", (caller_db,))
    reputation.init_db(caller_db)
    reputation.mark_as_trusted(TEST_CALLER, caller_db)
    try:
        result = run_pipeline(TEST_CLIP, TEST_CALLER, audio_model=AudioSpoofClassifier())
        yield result, call_log_db
    finally:
        mp.undo()


def test_pipeline_returns_a_fusion_result(pipeline_run):
    result, _ = pipeline_run
    assert 0.0 <= result.final_risk_score <= 1.0
    assert result.tier.value in ("log_only", "on_screen_warning", "require_callback_verification")


def test_pipeline_logs_exactly_one_row(pipeline_run):
    _, db = pipeline_run
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM call_log").fetchone()[0]
    conn.close()
    assert count == 1


def test_transcript_is_encrypted_at_rest(pipeline_run):
    from encryption import encrypt_text, decrypt_text
    _, db = pipeline_run
    conn = sqlite3.connect(db)
    stored = conn.execute("SELECT transcript FROM call_log LIMIT 1").fetchone()[0]
    conn.close()
    if stored:                                   # a clip with no speech stores an empty string
        assert stored.startswith("gAAAA")        # Fernet tokens begin with this prefix
        assert isinstance(decrypt_text(stored), str)
    secret = "wire transfer to account 12345"    # independent round trip, so the test never depends on the clip
    token = encrypt_text(secret)
    assert secret not in token and decrypt_text(token) == secret
