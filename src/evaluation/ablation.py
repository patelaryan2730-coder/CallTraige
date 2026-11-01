"""
Ablation Evaluation
----------------------


Expects a labelled test manifest CSV with columns:
    filepath, caller_number, true_label
where true_label = 1 for a genuine scam/vishing call, 0 for benign.

Runs four configurations on the same test set:
    1. Audio-only
    2. Transcript-only
    3. Caller-only
    4. Fused (all three)


Usage:
    python ablation.py --manifest data/test_manifest.csv --out results/ablation_results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "audio"))
sys.path.insert(0, str(ROOT / "transcript"))
sys.path.insert(0, str(ROOT / "caller"))
sys.path.insert(0, str(ROOT / "fusion"))

from model import AudioSpoofClassifier, score_audio_file           # noqa: E402
from risk_scorer import transcribe_audio, rule_based_score          # noqa: E402
from reputation import score_caller, init_db as init_caller_db      # noqa: E402
from fusion_engine import fuse_scores                                # noqa: E402

DECISION_THRESHOLD = 0.4  # score >= this counts as "flagged" for metric purposes


def evaluate_manifest(manifest_path: str) -> pd.DataFrame:
    init_caller_db()
    df = pd.read_csv(manifest_path)

    print("Loading audio model once for the whole evaluation run...")
    audio_model = AudioSpoofClassifier()

    results = []
    for _, row in df.iterrows():
        audio_score = score_audio_file(audio_model, row["filepath"])
        transcript = transcribe_audio(row["filepath"])
        transcript_score = rule_based_score(transcript)
        caller_result = score_caller(row["caller_number"])

        fusion = fuse_scores(audio_score, transcript_score, caller_result.risk_score)

        results.append(
            {
                "filepath": row["filepath"],
                "true_label": row["true_label"],
                "audio_score": audio_score,
                "transcript_score": transcript_score,
                "caller_score": caller_result.risk_score,
                "fused_score": fusion.final_risk_score,
            }
        )
        print(f"Processed {row['filepath']}: fused={fusion.final_risk_score:.2f}")

    return pd.DataFrame(results)


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Precision/Recall/F1 for each of the four configurations."""
    y_true = results_df["true_label"]

    configs = {
        "Audio-only": results_df["audio_score"] >= DECISION_THRESHOLD,
        "Transcript-only": results_df["transcript_score"] >= DECISION_THRESHOLD,
        "Caller-only": results_df["caller_score"] >= DECISION_THRESHOLD,
        "Fused (ours)": results_df["fused_score"] >= DECISION_THRESHOLD,
    }

    rows = []
    for name, y_pred in configs.items():
        rows.append(
            {
                "Configuration": name,
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="ablation_results.csv")
    args = parser.parse_args()

    raw_results = evaluate_manifest(args.manifest)
    metrics = compute_metrics(raw_results)

    print("\n=== ABLATION RESULTS (this is your key report figure) ===")
    print(metrics.to_string(index=False))

    raw_results.to_csv(args.out, index=False)
    metrics.to_csv(args.out.replace(".csv", "_metrics.csv"), index=False)
    print(f"\nSaved raw scores to {args.out}")
    print(f"Saved metrics table to {args.out.replace('.csv', '_metrics.csv')}")
