-- ============================================================
-- CallTraige — call_log.db schema
-- Matches docs/DATA_DESIGN.md Table 2
-- ============================================================

CREATE TABLE IF NOT EXISTS call_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,              -- ISO-8601
    caller_number       TEXT NOT NULL,               -- logical FK -> known_callers.phone_number
                                                       -- (not DB-enforced across separate SQLite
                                                       -- files - see DATA_DESIGN.md relationship note)
    transcript          TEXT,
    audio_score         REAL NOT NULL CHECK (audio_score BETWEEN 0 AND 1),
    transcript_score    REAL NOT NULL CHECK (transcript_score BETWEEN 0 AND 1),
    caller_score        REAL NOT NULL CHECK (caller_score BETWEEN 0 AND 1),
    final_risk_score    REAL NOT NULL CHECK (final_risk_score BETWEEN 0 AND 1),
    tier                TEXT NOT NULL CHECK (tier IN (
                             'log_only', 'on_screen_warning', 'require_callback_verification'
                         )),
    matched_categories  TEXT   -- comma-separated; see DATA_DESIGN.md 1NF note
);

-- Speeds up the dashboard's "recent calls" query (ORDER BY timestamp DESC)
CREATE INDEX IF NOT EXISTS idx_call_log_timestamp
    ON call_log (timestamp DESC);

-- Speeds up "show me all calls from this number" lookups
CREATE INDEX IF NOT EXISTS idx_call_log_caller_number
    ON call_log (caller_number);

-- Speeds up dashboard summary metrics (COUNT(*) WHERE tier = ...)
CREATE INDEX IF NOT EXISTS idx_call_log_tier
    ON call_log (tier);



