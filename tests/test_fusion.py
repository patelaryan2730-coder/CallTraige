"""Unit tests for the fusion engine (src/fusion/fusion_engine.py). Weights: audio 0.4, transcript 0.4, caller 0.2."""
import pytest
from fusion_engine import fuse_scores, DEFAULT_WEIGHTS


def test_weights_sum_to_one():
    assert pytest.approx(sum(DEFAULT_WEIGHTS.values()), abs=1e-6) == 1.0


def test_fused_score_matches_hand_calculation():
    # 0.8*0.4 + 0.6*0.4 + 0.2*0.2 = 0.60
    result = fuse_scores(audio_score=0.8, transcript_score=0.6, caller_score=0.2)
    assert pytest.approx(result.final_risk_score, abs=1e-2) == 0.60


def test_low_scores_map_to_log_only():
    result = fuse_scores(audio_score=0.1, transcript_score=0.1, caller_score=0.1)
    assert result.tier.value == "log_only"            # tier is an Enum: compare its .value


def test_high_scores_map_to_require_callback_verification():
    result = fuse_scores(audio_score=0.9, transcript_score=0.9, caller_score=0.9)
    assert result.tier.value == "require_callback_verification"


def test_result_includes_an_explanation():
    result = fuse_scores(audio_score=0.5, transcript_score=0.5, caller_score=0.5)
    assert result.explanation and isinstance(result.explanation, str)
