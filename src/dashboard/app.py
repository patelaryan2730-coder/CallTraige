"""
CallTraige Security Operations Dashboard
------------------------------------------
SOC-style threat intelligence UI for the CallTraige vishing-detection pipeline.

Run with:
    streamlit run app.py
"""

import re
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "audio"))
sys.path.insert(0, str(ROOT / "transcript"))
sys.path.insert(0, str(ROOT / "caller"))
sys.path.insert(0, str(ROOT / "fusion"))

from encryption import encrypt_text, decrypt_text  # noqa: E402
from model import AudioSpoofClassifier, score_audio_file  # noqa: E402
from risk_scorer import transcribe_audio, rule_based_score, matched_categories  # noqa: E402
from reputation import score_caller, init_db as init_caller_db  # noqa: E402
from fusion_engine import fuse_scores  # noqa: E402
from validation import validate_audio_file, AudioValidationError  # noqa: E402

DB_PATH = Path(__file__).parent / "call_log.db"

# ============================================================================
# COLOR SYSTEM
# ============================================================================
COLORS = {
    "bg": "#0A0E17",
    "bg_secondary": "#0F1420",
    "card": "rgba(255, 255, 255, 0.035)",
    "card_border": "rgba(255, 255, 255, 0.08)",
    "text": "#E8EAF0",
    "text_muted": "#8B93A7",
    "success": "#10B981",
    "warning": "#F59E0B",
    "critical": "#EF4444",
    "info": "#06B6D4",
    "accent": "#3B82F6",
}

TIER_META = {
    "log_only": {"label": "LOW", "color": COLORS["success"], "icon": "🟢"},
    "on_screen_warning": {"label": "ELEVATED", "color": COLORS["warning"], "icon": "🟡"},
    "require_callback_verification": {"label": "CRITICAL", "color": COLORS["critical"], "icon": "🔴"},
}

# Display-only keyword highlighting - separate from the real rule_based_score()
# scoring logic in transcript/risk_scorer.py. This is purely cosmetic so an
# analyst can visually scan a transcript; it does not affect any score.
SUSPICIOUS_PHRASES = [
    r"bank account", r"transfer (the )?money", r"one[- ]time password", r"\botp\b",
    r"urgent(ly)?", r"verify your identity", r"suspend(ed)?", r"don'?t tell anyone",
    r"gift card", r"wire transfer", r"social security", r"password", r"pin number",
    r"act now", r"immediately", r"confidential",
]


# ============================================================================
# STYLING
# ============================================================================
def inject_custom_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}

        .stApp {{
            background: radial-gradient(circle at 20% 0%, #10182B 0%, {COLORS['bg']} 45%);
        }}

        section[data-testid="stSidebar"] {{
            background: {COLORS['bg_secondary']};
        }}

        /* Glass card */
        .glass-card {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['card_border']};
            border-radius: 16px;
            padding: 20px 22px;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 24px rgba(0,0,0,0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(59,130,246,0.15);
        }}

        /* Header */
        .soc-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 24px 28px; margin-bottom: 24px;
            background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.04));
            border: 1px solid {COLORS['card_border']};
            border-radius: 20px;
        }}
        .soc-title {{
            font-size: 30px; font-weight: 800; color: {COLORS['text']};
            letter-spacing: -0.5px; margin: 0;
        }}
        .soc-subtitle {{
            font-size: 14px; color: {COLORS['text_muted']}; margin-top: 4px;
            font-weight: 500;
        }}
        .status-pill {{
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(16,185,129,0.12); color: {COLORS['success']};
            padding: 6px 14px; border-radius: 999px; font-size: 12px;
            font-weight: 600; border: 1px solid rgba(16,185,129,0.3);
        }}
        .dot {{
            width: 7px; height: 7px; border-radius: 50%; background: currentColor;
            box-shadow: 0 0 8px currentColor;
        }}

        /* KPI card */
        .kpi-card {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['card_border']};
            border-radius: 16px; padding: 18px 20px;
            height: 100%;
        }}
        .kpi-icon {{ font-size: 20px; opacity: 0.85; }}
        .kpi-value {{
            font-size: 30px; font-weight: 800; color: {COLORS['text']};
            margin: 6px 0 2px 0; font-family: 'JetBrains Mono', monospace;
        }}
        .kpi-label {{ font-size: 12px; color: {COLORS['text_muted']}; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.5px; }}
        .kpi-sub {{ font-size: 11px; color: {COLORS['text_muted']}; margin-top: 4px; }}

        /* Badges */
        .badge {{
            display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
        }}
        .badge-low {{ background: rgba(16,185,129,0.15); color: {COLORS['success']}; border: 1px solid rgba(16,185,129,0.4); }}
        .badge-elevated {{ background: rgba(245,158,11,0.15); color: {COLORS['warning']}; border: 1px solid rgba(245,158,11,0.4); }}
        .badge-critical {{ background: rgba(239,68,68,0.15); color: {COLORS['critical']}; border: 1px solid rgba(239,68,68,0.4); }}

        .phrase-tag {{
            display: inline-block; background: rgba(239,68,68,0.18); color: #FCA5A5;
            border: 1px solid rgba(239,68,68,0.4); padding: 1px 7px; border-radius: 6px;
            font-weight: 600; font-size: 0.92em;
        }}

        .section-title {{
            font-size: 18px; font-weight: 700; color: {COLORS['text']};
            margin: 28px 0 14px 0; display: flex; align-items: center; gap: 8px;
        }}

        .timeline-item {{
            display: flex; align-items: center; gap: 14px; padding: 10px 0;
            border-bottom: 1px solid {COLORS['card_border']};
        }}
        .timeline-icon {{
            width: 32px; height: 32px; border-radius: 50%; display: flex;
            align-items: center; justify-content: center; background: rgba(59,130,246,0.15);
            border: 1px solid rgba(59,130,246,0.3); font-size: 15px;
        }}

        div[data-testid="stForm"] {{
            background: {COLORS['card']}; border: 1px solid {COLORS['card_border']};
            border-radius: 16px; padding: 20px;
        }}

        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['info']});
            color: white; border: none; border-radius: 10px; font-weight: 600;
            padding: 10px 24px; transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            box-shadow: 0 4px 20px rgba(59,130,246,0.4); transform: translateY(-1px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# DATABASE (unchanged logic)
# ============================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            caller_number TEXT,
            transcript TEXT,
            audio_score REAL,
            transcript_score REAL,
            caller_score REAL,
            final_risk_score REAL,
            tier TEXT,
            matched_categories TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_call(caller_number, transcript, audio_score, transcript_score,
             caller_score, final_risk_score, tier, matched_categories):
    """Transcript is encrypted before storage only decrypted when displayed."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO call_log
        
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            caller_number,
            encrypt_text(transcript),
            audio_score,
            transcript_score,
            caller_score,
            final_risk_score,
            tier,
            ", ".join(matched_categories),
        ),
    )
    conn.commit()
    conn.close()


def load_calls() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM call_log ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df["transcript"] = df["transcript"].apply(lambda t: decrypt_text(t) if t else "")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource
def load_audio_model():
    return AudioSpoofClassifier()


# ============================================================================
# PIPELINE 
# ============================================================================
def analyze_uploaded_call(uploaded_file, caller_number: str):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    timeline = []
    t_start = time.time()

    try:
        validation = validate_audio_file(tmp_path)
    except AudioValidationError as e:
        st.error(f"Rejected: {e}")
        return None

    try:
        with st.spinner("Running audio spoof detection..."):
            audio_model = load_audio_model()
            audio_score = score_audio_file(audio_model, tmp_path)
    except Exception as e:
        st.error(
            f"This audio file could not be decoded ({e}). "
            "It may be corrupted or use an unsupported FLAC encoding. "
            "Try a different file."
        )
        return None
    timeline.append({"step": "Audio Analysis Complete", "icon": "🎧",
                      "status": "done", "elapsed": time.time() - t_start})

    with st.spinner("Transcribing call audio..."):
        transcript = transcribe_audio(tmp_path)
    timeline.append({"step": "Transcript Generated", "icon": "📝",
                      "status": "done", "elapsed": time.time() - t_start})

    transcript_score = rule_based_score(transcript)
    categories = matched_categories(transcript)
    timeline.append({"step": "Threat Categories Detected", "icon": "🔍",
                      "status": "done" if categories else "clear",
                      "elapsed": time.time() - t_start})

    caller_result = score_caller(caller_number)
    fusion_result = fuse_scores(audio_score, transcript_score, caller_result.risk_score)
    timeline.append({"step": "Fusion Score Generated", "icon": "⚡",
                      "status": "done", "elapsed": time.time() - t_start})

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
    timeline.append({"step": "Final Decision Logged", "icon": "✅",
                      "status": "done", "elapsed": time.time() - t_start})

    return {
        "caller_number": caller_number,
        "transcript": transcript,
        "audio_score": audio_score,
        "transcript_score": transcript_score,
        "caller_score": caller_result.risk_score,
        "caller_trust_level": caller_result.trust_level,
        "final_risk_score": fusion_result.final_risk_score,
        "tier": fusion_result.tier.value,
        "categories": categories,
        "explanation": fusion_result.explanation,
        "timeline": timeline,
        "duration_seconds": validation.duration_seconds,
    }


# ============================================================================
# UI HELPERS
# ============================================================================
def badge_class(tier: str) -> str:
    return {"log_only": "badge-low", "on_screen_warning": "badge-elevated",
            "require_callback_verification": "badge-critical"}.get(tier, "badge-low")


def tier_badge(tier: str) -> str:
    meta = TIER_META.get(tier, TIER_META["log_only"])
    return f'<span class="badge {badge_class(tier)}">{meta["icon"]} {meta["label"]}</span>'


def recommended_action(tier: str) -> str:
    return {
        "log_only": "No action required — continue passive monitoring.",
        "on_screen_warning": "Display an on-screen warning to the call recipient; recommend manual review.",
        "require_callback_verification": "Do not trust this call. Require independent callback verification before proceeding.",
    }.get(tier, "No action required.")


def highlight_transcript(text: str) -> str:
    """Cosmetic-only keyword highlighting for the analyst view. Does not
    affect the real transcript_score, which is computed separately by
    rule_based_score()."""
    if not text:
        return "<i>No speech detected.</i>"
    escaped = text
    for pattern in SUSPICIOUS_PHRASES:
        escaped = re.sub(
            f"({pattern})",
            r'<span class="phrase-tag">\1</span>',
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped


def gauge_chart(value: float, title: str, color: str, height: int = 200) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 26, "color": COLORS["text"]}},
        title={"text": title, "font": {"size": 13, "color": COLORS["text_muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text_muted"], "tickfont": {"size": 9}},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(16,185,129,0.12)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.12)"},
                {"range": [70, 100], "color": "rgba(239,68,68,0.12)"},
            ],
        },
    ))
    fig.update_layout(
        height=height, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": COLORS["text"]},
    )
    return fig


def threat_level_gauge(score: float) -> go.Figure:
    if score >= 0.7:
        color = COLORS["critical"]
    elif score >= 0.4:
        color = COLORS["warning"]
    else:
        color = COLORS["success"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 40, "color": COLORS["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text_muted"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(16,185,129,0.15)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": score * 100},
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": COLORS["text"]},
    )
    return fig


# ============================================================================
# SECTIONS
# ============================================================================
def render_header(df: pd.DataFrame):
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    calls_today = 0
    if not df.empty:
        today = datetime.now(timezone.utc).date()
        calls_today = (df["timestamp"].dt.date == today).sum()

    model_status = "Loaded" if "_audio_model_loaded" in st.session_state else "Idle (loads on first analysis)"

    st.markdown(
        f"""
        <div class="soc-header">
            <div>
                <p class="soc-title">🛡️ CallTraige Security Operations Dashboard</p>
                <p class="soc-subtitle">AI-Powered Voice Phishing Detection &amp; Call Risk Intelligence</p>
            </div>
            <div style="text-align:right;">
                <div class="status-pill"><span class="dot"></span> System Operational</div>
                <div style="color:{COLORS['text_muted']}; font-size:12px; margin-top:8px; font-family:'JetBrains Mono',monospace;">
                    {now_utc}
                </div>
                <div style="color:{COLORS['text_muted']}; font-size:12px; margin-top:2px;">
                    Model: {model_status} &nbsp;•&nbsp; Calls today: {calls_today}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame):
    st.markdown('<div class="section-title">📊 Executive Overview</div>', unsafe_allow_html=True)

    total = len(df)
    high_risk = int((df["final_risk_score"] >= 0.4).sum()) if total else 0
    verify = int((df["tier"] == "require_callback_verification").sum()) if total else 0
    avg_risk = f"{df['final_risk_score'].mean():.2f}" if total else "—"
    spoof_detections = int((df["audio_score"] >= 0.5).sum()) if total else 0
    scam_transcripts = int((df["transcript_score"] > 0).sum()) if total else 0

    kpis = [
        ("📞", "Total Calls Processed", total, "All-time"),
        ("⚠️", "High-Risk Calls", high_risk, "Score ≥ 0.40"),
        ("🔒", "Verification Required", verify, "Critical tier"),
        ("📈", "Average Risk Score", avg_risk, "Across all calls"),
        ("🎙️", "Audio Spoof Detections", spoof_detections, "Audio score ≥ 0.50"),
        ("🚨", "Scam Language Detected", scam_transcripts, "Transcript flags > 0"),
    ]

    cols = st.columns(6)
    for col, (icon, label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def compute_threat_level(df: pd.DataFrame) -> tuple[float, str]:
    if df.empty:
        return 0.0, "NO DATA"
    recent = df.head(20)
    score = recent["final_risk_score"].mean()
    if (recent["tier"] == "require_callback_verification").any():
        label = "CRITICAL"
    elif (recent["tier"] == "on_screen_warning").any():
        label = "ELEVATED"
    else:
        label = "LOW"
    return score, label


def render_threat_command_center(df: pd.DataFrame):
    st.markdown('<div class="section-title">🎯 Risk Command Center</div>', unsafe_allow_html=True)
    score, label = compute_threat_level(df)
    label_color = {"CRITICAL": COLORS["critical"], "ELEVATED": COLORS["warning"],
                   "LOW": COLORS["success"], "NO DATA": COLORS["text_muted"]}[label]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(threat_level_gauge(score), use_container_width=True, config={"displayModeBar": False}, key="threat_level_gauge")
    with c2:
        st.markdown(
            f"""
            <div class="glass-card" style="height:100%;">
                <div style="font-size:13px; color:{COLORS['text_muted']}; font-weight:600;">CURRENT THREAT LEVEL</div>
                <div style="font-size:42px; font-weight:800; color:{label_color}; margin:6px 0;">{label}</div>
                <div style="color:{COLORS['text_muted']}; font-size:13px; line-height:1.6;">
                    Based on the fused risk scores of the most recent 20 logged calls.
                    A <b>CRITICAL</b> rating means at least one recent call required callback
                    verification; <b>ELEVATED</b> means at least one triggered an on-screen
                    warning; <b>LOW</b> means all recent calls were log-only.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analysis_workspace():
    st.markdown('<div class="section-title">🔬 Analysis Workspace</div>', unsafe_allow_html=True)

    step_cols = st.columns(4)
    steps = ["① Upload Recording", "② Caller Information", "③ Run Analysis", "④ View Results"]
    for col, step in zip(step_cols, steps):
        col.markdown(
            f'<div style="text-align:center; padding:10px; color:{COLORS["text_muted"]}; '
            f'font-size:12px; font-weight:600;">{step}</div>',
            unsafe_allow_html=True,
        )

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Drop a call recording here or click to browse (.wav or .flac)",
            type=["wav", "flac"],
        )
        caller_number = st.text_input("Caller phone number", value="+61400000000")
        submitted = st.form_submit_button("🚀 Run Analysis")

    if submitted:
        if uploaded_file is None:
            st.warning("Please choose an audio file first.")
        else:
            st.session_state["_audio_model_loaded"] = True
            result = analyze_uploaded_call(uploaded_file, caller_number)
            if result:
                st.session_state["last_result"] = result


def render_results(result: dict):
    st.markdown('<div class="section-title">📋 Threat Assessment</div>', unsafe_allow_html=True)

    tier = result["tier"]
    meta = TIER_META[tier]
    signal_strength = abs(result["final_risk_score"] - 0.5) * 2 * 100

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="color:{COLORS['text_muted']}; font-size:12px; font-weight:600;">FUSED RISK SCORE</div>
                <div style="font-size:52px; font-weight:800; color:{meta['color']};">
                    {result['final_risk_score']:.2f}
                </div>
                {tier_badge(tier)}
                <hr style="border-color:{COLORS['card_border']}; margin:14px 0;">
                <div style="font-size:12px; color:{COLORS['text_muted']};">RECOMMENDED ACTION</div>
                <div style="font-size:14px; color:{COLORS['text']}; margin-top:4px;">{recommended_action(tier)}</div>
                <div style="font-size:12px; color:{COLORS['text_muted']}; margin-top:12px;">
                    SIGNAL STRENGTH: {signal_strength:.0f}%
                    <span style="opacity:0.7;">(distance from the 0.5 decision midpoint — not a calibrated confidence interval)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(gauge_chart(result["audio_score"], "Audio Spoofing", COLORS["info"]),
                         use_container_width=True, config={"displayModeBar": False}, key="result_gauge_audio")
        g2.plotly_chart(gauge_chart(result["transcript_score"], "Transcript Risk", COLORS["warning"]),
                         use_container_width=True, config={"displayModeBar": False}, key="result_gauge_transcript")
        g3.plotly_chart(gauge_chart(result["caller_score"], "Caller Reputation", COLORS["accent"]),
                         use_container_width=True, config={"displayModeBar": False}, key="result_gauge_caller")

    # Timeline
    st.markdown('<div class="section-title">⏱️ Threat Analysis Timeline</div>', unsafe_allow_html=True)
    for step in result["timeline"]:
        st.markdown(
            f"""
            <div class="timeline-item">
                <div class="timeline-icon">{step['icon']}</div>
                <div style="flex:1;">
                    <div style="color:{COLORS['text']}; font-weight:600; font-size:14px;">{step['step']}</div>
                </div>
                <div style="color:{COLORS['text_muted']}; font-size:12px; font-family:'JetBrains Mono',monospace;">
                    +{step['elapsed']:.2f}s
                </div>
                <div style="color:{COLORS['success']}; font-size:16px;">✓</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result["categories"]:
        st.markdown('<div class="section-title">🏷️ Threat Indicators</div>', unsafe_allow_html=True)
        tags = " ".join(f'<span class="phrase-tag">{c}</span>' for c in result["categories"])
        st.markdown(tags, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📄 Transcript</div>', unsafe_allow_html=True)
    st.caption("Suspicious-phrase highlighting below is a display aid only and does not affect the transcript risk score.")
    st.markdown(
        f'<div class="glass-card">{highlight_transcript(result["transcript"])}</div>',
        unsafe_allow_html=True,
    )


def render_call_history(df: pd.DataFrame):
    st.markdown('<div class="section-title">📁 Call History</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    search = fc1.text_input("🔍 Search by caller number", "")
    tier_filter = fc2.selectbox("Filter by tier", ["All"] + list(TIER_META.keys()))
    sort_by = fc3.selectbox("Sort by", ["Time (newest)", "Risk score (highest)", "Risk score (lowest)"])

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["caller_number"].str.contains(search, case=False, na=False)]
    if tier_filter != "All":
        filtered = filtered[filtered["tier"] == tier_filter]
    if sort_by == "Risk score (highest)":
        filtered = filtered.sort_values("final_risk_score", ascending=False)
    elif sort_by == "Risk score (lowest)":
        filtered = filtered.sort_values("final_risk_score", ascending=True)
    else:
        filtered = filtered.sort_values("timestamp", ascending=False)

    if filtered.empty:
        st.info("No calls match the current filters.")
        return filtered

    header_cols = st.columns([1, 2, 2, 1.5, 1.5, 2.5])
    for c, label in zip(header_cols, ["ID", "Time", "Caller", "Risk Score", "Threat Level", "Action Required"]):
        c.markdown(f'<div style="color:{COLORS["text_muted"]}; font-size:11px; font-weight:700; text-transform:uppercase;">{label}</div>',
                    unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        meta = TIER_META.get(row["tier"], TIER_META["log_only"])
        cols = st.columns([1, 2, 2, 1.5, 1.5, 2.5])
        cols[0].markdown(f"`INC-{row['id']:04d}`")
        cols[1].markdown(f"<span style='font-size:13px; color:{COLORS['text_muted']}'>{row['timestamp'].strftime('%Y-%m-%d %H:%M')}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"<span style='font-family:JetBrains Mono,monospace; font-size:13px;'>{row['caller_number']}</span>", unsafe_allow_html=True)
        cols[3].markdown(f"<b style='color:{meta['color']}'>{row['final_risk_score']:.2f}</b>", unsafe_allow_html=True)
        cols[4].markdown(tier_badge(row["tier"]), unsafe_allow_html=True)
        cols[5].markdown(f"<span style='font-size:12px;'>{recommended_action(row['tier'])}</span>", unsafe_allow_html=True)

    return filtered


def render_investigation_panel(df: pd.DataFrame):
    st.markdown('<div class="section-title">🕵️ Call Investigation Panel</div>', unsafe_allow_html=True)

    selected_id = st.selectbox(
        "Select an incident to investigate:",
        options=df["id"],
        format_func=lambda i: f"INC-{i:04d} — {df.loc[df['id'] == i, 'timestamp'].values[0]}",
    )
    row = df[df["id"] == selected_id].iloc[0]
    meta = TIER_META.get(row["tier"], TIER_META["log_only"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Executive Summary", "Threat Indicators", "Model Scores", "Transcript Analysis", "Risk Explanation"]
    )

    with tab1:
        st.markdown(
            f"""
            <div class="glass-card">
                <p><b>Incident:</b> INC-{row['id']:04d}</p>
                <p><b>Caller:</b> {row['caller_number']}</p>
                <p><b>Time:</b> {row['timestamp']}</p>
                <p><b>Threat Level:</b> {tier_badge(row['tier'])}</p>
                <p><b>Recommended Action:</b> {recommended_action(row['tier'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        if row["matched_categories"]:
            tags = " ".join(f'<span class="phrase-tag">{c.strip()}</span>' for c in row["matched_categories"].split(","))
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.info("No transcript-based threat categories were triggered for this call.")

    with tab3:
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(gauge_chart(row["audio_score"], "Audio Spoofing", COLORS["info"]),
                         use_container_width=True, config={"displayModeBar": False}, key=f"inv_gauge_audio_{selected_id}")
        g2.plotly_chart(gauge_chart(row["transcript_score"], "Transcript Risk", COLORS["warning"]),
                         use_container_width=True, config={"displayModeBar": False}, key=f"inv_gauge_transcript_{selected_id}")
        g3.plotly_chart(gauge_chart(row["caller_score"], "Caller Reputation", COLORS["accent"]),
                         use_container_width=True, config={"displayModeBar": False}, key=f"inv_gauge_caller_{selected_id}")

    with tab4:
        st.caption("Suspicious-phrase highlighting is a display aid only; it does not affect the transcript risk score.")
        st.markdown(f'<div class="glass-card">{highlight_transcript(row["transcript"])}</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown(
            f"""
            <div class="glass-card">
                <p style="color:{COLORS['text_muted']};">
                    Final risk score of <b style="color:{meta['color']}">{row['final_risk_score']:.2f}</b>
                    is a weighted fusion of the audio ({row['audio_score']:.2f}), transcript
                    ({row['transcript_score']:.2f}), and caller-reputation ({row['caller_score']:.2f}) scores,
                    resulting in the <b>{meta['label']}</b> tier.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_visualizations(df: pd.DataFrame):
    st.markdown('<div class="section-title">📈 Threat Intelligence Visualizations</div>', unsafe_allow_html=True)

    v1, v2 = st.columns(2)

    with v1:
        fig = go.Figure(data=[go.Histogram(
            x=df["final_risk_score"], nbinsx=15, marker_color=COLORS["accent"],
        )])
        fig.update_layout(
            title="Risk Score Distribution", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text"]}, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with v2:
        tier_counts = df["tier"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=[TIER_META.get(t, {"label": t})["label"] for t in tier_counts.index],
            values=tier_counts.values, hole=0.55,
            marker_colors=[TIER_META.get(t, {"color": COLORS["accent"]})["color"] for t in tier_counts.index],
        )])
        fig.update_layout(
            title="Threat Tier Distribution", height=300,
            paper_bgcolor="rgba(0,0,0,0)", font={"color": COLORS["text"]},
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    v3, v4 = st.columns(2)

    with v3:
        fig = go.Figure(data=[go.Scatter(
            x=df.sort_values("timestamp")["timestamp"],
            y=df.sort_values("timestamp")["final_risk_score"],
            mode="lines+markers", line={"color": COLORS["info"], "width": 2},
            marker={"size": 6},
        )])
        fig.update_layout(
            title="Risk Score Over Time", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text"]}, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with v4:
        all_categories = []
        for cats in df["matched_categories"].dropna():
            all_categories.extend([c.strip() for c in cats.split(",") if c.strip()])
        if all_categories:
            cat_counts = pd.Series(all_categories).value_counts()
            fig = go.Figure(data=[go.Bar(
                x=cat_counts.values, y=cat_counts.index, orientation="h",
                marker_color=COLORS["warning"],
            )])
            fig.update_layout(
                title="Threat Category Breakdown", height=300,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text"]}, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No transcript threat categories have been triggered yet.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.set_page_config(page_title="CallTraige SOC Dashboard", layout="wide", page_icon="🛡️")
    inject_custom_css()

    init_db()
    init_caller_db()
    df = load_calls()

    render_header(df)
    render_analysis_workspace()

    if "last_result" in st.session_state:
        render_results(st.session_state["last_result"])
        st.divider()

    if df.empty:
        st.info("No calls logged yet. Upload a call recording above to get started.")
        return

    render_kpis(df)
    render_threat_command_center(df)
    filtered = render_call_history(df)
    if not filtered.empty:
        render_investigation_panel(filtered)
    render_visualizations(df)


if __name__ == "__main__":
    main()