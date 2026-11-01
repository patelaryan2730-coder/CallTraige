@"
# CallTraige: Call Triage Dashboard

COMP6002 Computer Science Project, Curtin University, Semester 2 2026, Group 5.

CallTraige fuses an audio spoof score (frozen wav2vec2-base + small head), a transcript risk score
(faster-whisper + rule categories) and a caller reputation score into one tiered vishing risk decision,
with a Streamlit dashboard.

## Structure
- src/audio: audio scorer (model.py, train.py, head.pt = trained head)
- src/transcript: transcript scorer
- src/caller: caller reputation (SQLite)
- src/fusion: fusion engine and tiers
- src/common: validation and encryption
- src/dashboard: Streamlit app
- src/evaluation: run_pipeline.py and ablation.py
- tests: pytest suite (18 tests)
- notebooks: AASIST baseline and wav2vec2 head training (Kaggle)

## Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Run
python src\evaluation\run_pipeline.py --audio <clip.wav> --caller "+61499999999"
streamlit run src\dashboard\app.py

## Tests
python -m pytest tests -v

## Results (single runs)
- wav2vec2 head: test EER 17.15% on a balanced ASVspoof 2021 DF set (33,954 clips); AASIST baseline 19.49%.
- Not evaluated: ablation of fused versus single signals; real-voice behaviour.

## Authors
Rupesh Deuja, Aryan Bipinkumar Patel, Joyane Wanuyu Wamae, Prachi Mahendrabhai Joshi, Dheiryan Hitesh Bhesaniya
"@ | Set-Content README.md -Encoding utf8