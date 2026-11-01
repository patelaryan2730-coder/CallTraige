"""Makes the project modules importable in the same way run_pipeline.py does (flat imports from src/<folder>)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
for folder in ("audio", "transcript", "caller", "fusion", "dashboard", "common", "evaluation"):
    p = str(SRC / folder)
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
