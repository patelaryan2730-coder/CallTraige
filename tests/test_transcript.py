"""Unit tests for the rule-based transcript scorer (src/transcript/risk_scorer.py)."""
from risk_scorer import rule_based_score, matched_categories

VISHING_SAMPLE = ("Hi this is the fraud department, we need you to wire transfer the "
                  "funds immediately and please don't tell your manager about this call.")
BENIGN_SAMPLE = ("Hi, I'm just calling to confirm our meeting time for Thursday "
                 "afternoon, let me know if that still works for you.")


def test_vishing_script_scores_higher_than_benign():
    assert rule_based_score(VISHING_SAMPLE) > rule_based_score(BENIGN_SAMPLE)


def test_vishing_keywords_are_flagged():
    assert len(matched_categories(VISHING_SAMPLE)) > 0


def test_benign_text_flags_nothing():
    assert len(matched_categories(BENIGN_SAMPLE)) == 0


def test_empty_transcript_scores_zero():
    assert rule_based_score("") == 0.0
