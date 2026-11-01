"""
End-to-End Pipeline
----------------------
Ties the four components together: given a test call clip (.wav) and a
caller phone number, runs it through all three scorers, fuses the result,
and logs it to the dashboard database.


Usage:
    python run_pipeline.py --audio path/to/call.wav --caller "+61499999999"
"""

import argparse
import sys
from pathlib import Path

# Allow importing sibling modules (audio/, transcript/, caller/, fusion/, dashboard/)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "audio"))
sys.path.insert(0, str(ROOT / "transcript"))
sys.path.insert(0, str(ROOT / "caller"))
sys.path.insert(0, str(ROOT / "fusion"))
sys.path.insert(0, str(ROOT / "dashboard"))
sys.path.insert(0, str(ROOT / "common"))

from model import AudioSpoofClassifier, score_audio_file          # noqa: E402
from risk_scorer import transcribe_audio, rule_based_score, matched_categories  # noqa: E402
from reputation import score_caller, init_db as init_caller_db     # noqa: E402
from fusion_engine import fuse_scores                              # noqa: E402
from app import init_db as init_dashboard_db, log_call             # noqa: E402
from validation import validate_audio_file, AudioValidationError   # noqa: E402


def run_pipeline(audio_path: str, caller_number: str, audio_model=None):
    """
    Runs the full pipeline on one call clip. Pass in a pre-loaded
    AudioSpoofClassifier if you're processing many clips (avoids reloading
    the model each time); otherwise one is loaded fresh.
    """
    init_caller_db()
    init_dashboard_db()

    print(f"Processing {audio_path} from caller {caller_number}...")

    # 0. Validate input BEFORE any expensive processing (fail closed -
    #    see docs/ETHICS_SECURITY.md Denial-of-Service mitigation)
    try:
        validation = validate_audio_file(audio_path)
        print(f"  Input validated: {validation.duration_seconds:.1f}s, {validation.size_mb:.2f}MB")
    except AudioValidationError as e:
        print(f"  REJECTED: {e}")
        raise

    # 1. Audio spoofing score
    if audio_model is None:
        print("Loading audio model (first run downloads pretrained weights)...")
        audio_model = AudioSpoofClassifier()
    audio_score = score_audio_file(audio_model, audio_path)
    print(f"  Audio spoofing score: {audio_score:.2f}")

    # 2. Transcript risk score
    print("  Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    transcript_score = rule_based_score(transcript)
    categories = matched_categories(transcript)
    print(f"  Transcript risk score: {transcript_score:.2f} (flags: {categories})")

    # 3. Caller reputation score
    caller_result = score_caller(caller_number)
    print(f"  Caller reputation score: {caller_result.risk_score:.2f} ({caller_result.trust_level})")

    # 4. Fusion
    fusion_result = fuse_scores(audio_score, transcript_score, caller_result.risk_score)
    print(f"  >>> {fusion_result.explanation}")

    # 5. Log to dashboard
    log_call(
        caller_number=caller_number,
        transcript=transcript,
        audio_score=audio_score,
        transcript_score=transcript_score,
        caller_score=caller_result.risk_score,
        final_risk_score=fusion_result.final_risk_score,
        tier=fusion_result.tier.value,
        matched_categories=categories,
    )
    print("Logged to dashboard. Run `streamlit run ../dashboard/app.py` to view it.")

    return fusion_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a .wav test call clip")
    parser.add_argument("--caller", required=True, help="Caller phone number, e.g. +61499999999")
    args = parser.parse_args()

    run_pipeline(args.audio, args.caller)
