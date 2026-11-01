"""
Transcript Risk Scorer
------------------------
Two-stage pipeline:
  1. Speech-to-text using faster-whisper (open-source, runs locally, no API cost)
  2. Risk scoring of the resulting text using a rule-based layer (fast, explainable)
     combined with an optional fine-tuned text classifier (catches rephrased scams
     that don't match exact keywords).

Install: pip install faster-whisper scikit-learn
"""

import re
import shutil
import subprocess
import numpy as np
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Stage 1: Speech-to-text
# ---------------------------------------------------------------------------

def _load_audio_16k(filepath):
    """Decode any audio file to 16 kHz mono float32 with ffmpeg (avoids the PyAV decoder)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg  # pip install imageio-ffmpeg (bundles an ffmpeg binary)
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-nostdin", "-loglevel", "error", "-i", str(filepath),
           "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


_WHISPER_MODELS = {}


def transcribe_audio(filepath: str, model_size: str = "base") -> str:
    """
    Converts an audio file to text using faster-whisper.
    model_size options: tiny, base, small, medium, large-v3
    """
    from faster_whisper import WhisperModel

    if model_size not in _WHISPER_MODELS:
        _WHISPER_MODELS[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    model = _WHISPER_MODELS[model_size]
    segments, _ = model.transcribe(_load_audio_16k(filepath))
    return " ".join(segment.text for segment in segments)


# ---------------------------------------------------------------------------
# Stage 2a: Rule-based scam pattern scoring (fast, explainable, no training needed)
# ---------------------------------------------------------------------------

# Each pattern group has a weight - tune these against your labelled test set
# rather than guessing. Start with these as a reasonable baseline.
SCAM_PATTERNS = {
    "urgency": {
        "weight": 0.25,
        "patterns": [
            r"\burgent\b", r"\bright away\b", r"\bimmediately\b",
            r"\bact now\b", r"\bexpires? (today|soon)\b", r"\bdeadline\b",
        ],
    },
    "financial_request": {
        "weight": 0.35,
        "patterns": [
            r"\bwire transfer\b", r"\bgift cards?\b", r"\bbank details\b",
            r"\baccount number\b", r"\bverification code\b", r"\bpayment\b",
            r"\btransfer (the )?funds\b", r"\brouting number\b",
        ],
    },
    "secrecy": {
        "weight": 0.25,
        "patterns": [
            r"\bdon'?t tell\b", r"\bkeep this (confidential|private|secret)\b",
            r"\bbetween us\b", r"\bdo not (inform|discuss)\b",
        ],
    },
    "authority_impersonation": {
        "weight": 0.15,
        "patterns": [
            r"\bthis is (your|the) (ceo|manager|bank|irs|police)\b",
            r"\bofficial (request|notice)\b", r"\bon behalf of\b",
        ],
    },
}


def rule_based_score(text: str) -> float:
    """
    Returns a score in [0, 1] based on weighted keyword-pattern matches.
    This is intentionally simple and explainable - useful as a baseline
    and as a sanity check against your ML classifier.
    """
    text_lower = text.lower()
    score = 0.0
    for group in SCAM_PATTERNS.values():
        if any(re.search(pattern, text_lower) for pattern in group["patterns"]):
            score += group["weight"]
    return min(score, 1.0)


def matched_categories(text: str) -> List[str]:
    """Returns which pattern categories matched, for explainability in the dashboard."""
    text_lower = text.lower()
    matched = []
    for name, group in SCAM_PATTERNS.items():
        if any(re.search(pattern, text_lower) for pattern in group["patterns"]):
            matched.append(name)
    return matched


# ---------------------------------------------------------------------------
#  ML text classifier
# ---------------------------------------------------------------------------
# A fine-tuned DistilBERT classifier can catch rephrased scam language that
# doesn't match exact keywords. This is a stretch goal - get the rule-based
# scorer working and evaluated first, then add this if time allows.
#
# See train_text_classifier.py for a minimal fine-tuning script using
# HuggingFace's Trainer API on a small labelled dataset of transcripts.


@dataclass
class TranscriptRiskResult:
    text: str
    rule_score: float
    matched_categories: List[str]


def analyse_transcript(filepath: str) -> TranscriptRiskResult:
    text = transcribe_audio(filepath)
    score = rule_based_score(text)
    categories = matched_categories(text)
    return TranscriptRiskResult(text=text, rule_score=score, matched_categories=categories)


if __name__ == "__main__":
    # Quick test of the rule-based scorer with no audio needed.
    test_benign = "Hi, just calling to confirm your appointment for next Tuesday."
    test_scam = "This is urgent, you need to wire transfer the funds immediately and don't tell your manager."

    print("Benign text score:", rule_based_score(test_benign))
    print("Scam-like text score:", rule_based_score(test_scam))
    print("Matched categories (scam):", matched_categories(test_scam))