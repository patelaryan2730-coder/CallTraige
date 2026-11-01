"""
Fusion Engine
--------------
Combines the three independent signal scores (audio, transcript, caller
reputation) into a single risk score using a weighted sum, then maps that
score onto the graduated response tiers.



The default weights below are a reasonable starting point. Your evaluation
step (evaluate.py) should test different weight combinations against your
labelled test set and report whichever performs best - don't just guess and
leave it at the defaults.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class ResponseTier(Enum):
    LOG_ONLY = "log_only"
    WARN = "on_screen_warning"
    VERIFY = "require_callback_verification"


# Default weights - MUST add up to 1.0. Tune these empirically (see
# evaluation/ablation.py) rather than treating these as final.
DEFAULT_WEIGHTS = {
    "audio": 0.40,
    "transcript": 0.40,
    "caller": 0.20,
}

# Thresholds for graduated response - also worth tuning against your
# false-positive rate during evaluation.
TIER_THRESHOLDS = {
    "warn": 0.4,     # risk_score >= 0.4 -> at least a warning
    "verify": 0.7,   # risk_score >= 0.7 -> require callback verification
}


@dataclass
class FusionResult:
    audio_score: float
    transcript_score: float
    caller_score: float
    weights: Dict[str, float]
    final_risk_score: float = field(init=False)
    tier: ResponseTier = field(init=False)
    explanation: str = field(init=False)

    def __post_init__(self):
        self.final_risk_score = (
            self.audio_score * self.weights["audio"]
            + self.transcript_score * self.weights["transcript"]
            + self.caller_score * self.weights["caller"]
        )
        self.tier = self._determine_tier()
        self.explanation = self._build_explanation()

    def _determine_tier(self) -> ResponseTier:
        if self.final_risk_score >= TIER_THRESHOLDS["verify"]:
            return ResponseTier.VERIFY
        elif self.final_risk_score >= TIER_THRESHOLDS["warn"]:
            return ResponseTier.WARN
        return ResponseTier.LOG_ONLY

    def _build_explanation(self) -> str:
        """Human-readable explanation of which signal(s) drove the decision."""
        contributions = {
            "audio": self.audio_score * self.weights["audio"],
            "transcript": self.transcript_score * self.weights["transcript"],
            "caller": self.caller_score * self.weights["caller"],
        }
        dominant = max(contributions, key=contributions.get)
        return (
            f"Risk score {self.final_risk_score:.2f} ({self.tier.value}). "
            f"Largest contributor: {dominant} "
            f"(audio={self.audio_score:.2f}, transcript={self.transcript_score:.2f}, "
            f"caller={self.caller_score:.2f})"
        )


def fuse_scores(
    audio_score: float,
    transcript_score: float,
    caller_score: float,
    weights: Dict[str, float] = None,
) -> FusionResult:
    """
    Main entry point. Pass in the three independent scores (each 0-1) and
    get back a FusionResult with the combined score, response tier, and
    a human-readable explanation for the dashboard/audit log.
    """
    weights = weights or DEFAULT_WEIGHTS
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    return FusionResult(
        audio_score=audio_score,
        transcript_score=transcript_score,
        caller_score=caller_score,
        weights=weights,
    )


if __name__ == "__main__":
    print("--- Benign call example ---")
    result = fuse_scores(audio_score=0.05, transcript_score=0.0, caller_score=0.05)
    print(result.explanation)

    print("\n--- Suspicious call example (cloned voice + scam script + unknown caller) ---")
    result = fuse_scores(audio_score=0.88, transcript_score=0.85, caller_score=0.85)
    print(result.explanation)

    print("\n--- Ambiguous case: good audio detection, but scam-like transcript ---")
    result = fuse_scores(audio_score=0.15, transcript_score=0.80, caller_score=0.85)
    print(result.explanation)
