"""Unit tests for the audio scorer (src/audio/model.py). Needs head.pt next to model.py for the trained-head test."""
import os
import math
import pytest
from conftest import FIXTURES
from model import AudioSpoofClassifier, score_audio_file


@pytest.fixture(scope="module")
def audio_model():
    return AudioSpoofClassifier()


def test_score_is_a_probability(audio_model):
    """score_audio_file must return a float in [0, 1]."""
    score = score_audio_file(audio_model, str(FIXTURES / "clean_clip.flac"))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_trained_head_checkpoint_is_loaded(audio_model):
    """Replaces the old 'untrained head near 0.5' test, which is only true without head.pt.
    The trained head (head.pt) must be loaded and must give a valid (non-NaN) score."""
    assert audio_model.head_loaded is True
    score = score_audio_file(audio_model, str(FIXTURES / "clean_clip.flac"))
    assert not math.isnan(score) and 0.0 <= score <= 1.0


def test_broken_flac_raises_cleanly(audio_model):
    """A known-bad FLAC must raise a catchable error and not return a wrong score."""
    path = FIXTURES / "broken_clip.flac"
    if not os.path.exists(path):
        pytest.skip("tests/fixtures/broken_clip.flac is missing - add a known-bad FLAC (e.g. DF_E_2000011.flac)")
    with pytest.raises(Exception):
        score_audio_file(audio_model, str(path))


def test_silent_clip_does_not_crash(audio_model):
    score = score_audio_file(audio_model, str(FIXTURES / "silent_clip.wav"))
    assert isinstance(score, float)
