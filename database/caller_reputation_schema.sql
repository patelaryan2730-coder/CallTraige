-- ============================================================
-- CallTraige — caller_reputation.db schema
-- Matches docs/DATA_DESIGN.md Table 1
-- ============================================================

CREATE TABLE IF NOT EXISTS known_callers (
    phone_number  TEXT PRIMARY KEY,
    trust_level   TEXT NOT NULL CHECK (trust_level IN ('trusted', 'seen_before', 'unknown')),
    first_seen    TEXT NOT NULL,   -- ISO-8601 timestamp
    last_seen     TEXT NOT NULL,   -- ISO-8601 timestamp
    call_count    INTEGER NOT NULL DEFAULT 1 CHECK (call_count >= 0)
);

-- Speeds up dashboard queries that filter/sort by trust level
-- (e.g. "show all unknown callers this week")
CREATE INDEX IF NOT EXISTS idx_known_callers_trust_level
    ON known_callers (trust_level);

-- Speeds up "most recently active callers" queries
CREATE INDEX IF NOT EXISTS idx_known_callers_last_seen
    ON known_callers (last_seen DESC);
