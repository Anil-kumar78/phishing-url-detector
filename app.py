"""
Phishing URL Detector — Streamlit App
Run with:  streamlit run app.py
"""

import os
# Limit OpenBLAS / MKL threads BEFORE any numpy/sklearn import to prevent memory crash
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import pickle
import sqlite3
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from features.extract_features import extract_features

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "phishing_model.pkl")
META_PATH  = os.path.join(PROJECT_ROOT, "model", "model_metadata.pkl")
DB_PATH    = os.path.join(PROJECT_ROOT, "history.db")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    min-height: 100vh;
}

/* Hero title */
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
}

.hero-subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Result cards */
.result-phishing {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(239,68,68,0.15);
}

.result-legitimate {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.08));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(34,197,94,0.15);
}

.result-icon {
    font-size: 4rem;
    text-align: center;
    display: block;
    margin-bottom: 0.5rem;
}

.result-label {
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
}

.result-label-phishing  { color: #f87171; }
.result-label-legit     { color: #4ade80; }

.result-confidence {
    text-align: center;
    font-size: 1rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* Feature table */
.feature-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.5rem;
}

/* URL input styling */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(96,165,250,0.6) !important;
    box-shadow: 0 0 0 2px rgba(96,165,250,0.2) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.4) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #60a5fa !important;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── SQLite History DB ─────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            url       TEXT,
            result    TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def log_to_db(url: str, result: str, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO history (timestamp, url, result, confidence) VALUES (?, ?, ?, ?)",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), url, result, round(confidence, 4))
    )
    conn.commit()
    conn.close()


def fetch_history(limit: int = 100) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp, url, result, confidence FROM history ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    meta = pickle.load(open(META_PATH, "rb")) if os.path.exists(META_PATH) else {}
    return clf, meta


# ── Gauge chart ──────────────────────────────────────────────────────────────
def make_gauge(confidence: float, is_phishing: bool) -> go.Figure:
    color = "#ef4444" if is_phishing else "#22c55e"
    label = "PHISHING" if is_phishing else "LEGITIMATE"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": f"<b>{label}</b>", "font": {"size": 16, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar":  {"color": color, "thickness": 0.75},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(34,197,94,0.12)"},
                {"range": [40, 70], "color": "rgba(234,179,8,0.12)"},
                {"range": [70, 100],"color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": confidence * 100,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
    )
    return fig


# ── Feature importance bar chart ──────────────────────────────────────────────
def make_feature_chart(feat_dict: dict, meta: dict) -> go.Figure:
    importances = meta.get("feature_importances", {})
    names  = list(feat_dict.keys())
    values = list(feat_dict.values())
    imps   = [importances.get(n, 0) for n in names]

    # Sort by importance
    combined = sorted(zip(names, values, imps), key=lambda x: -x[2])
    names_s  = [c[0] for c in combined]
    values_s = [c[1] for c in combined]
    imps_s   = [c[2] for c in combined]

    # Colour bars: red if non-zero value and high importance, else muted
    bar_colors = []
    for v, imp in zip(values_s, imps_s):
        if v != 0 and imp > 0.03:
            bar_colors.append("rgba(248,113,113,0.85)")
        elif v != 0:
            bar_colors.append("rgba(96,165,250,0.7)")
        else:
            bar_colors.append("rgba(71,85,105,0.5)")

    fig = go.Figure(go.Bar(
        x=values_s,
        y=names_s,
        orientation="h",
        marker_color=bar_colors,
        text=[str(v) for v in values_s],
        textposition="outside",
        textfont={"color": "#94a3b8", "size": 11},
    ))
    fig.update_layout(
        height=max(350, len(names) * 26),
        margin=dict(l=10, r=60, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(color="#94a3b8", tickfont={"size": 11}),
        font_color="#f1f5f9",
        showlegend=False,
    )
    return fig


# ── History pie chart ─────────────────────────────────────────────────────────
def make_history_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["result"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.5,
        marker_colors=["#22c55e" if l == "Legitimate" else "#ef4444" for l in counts.index],
        textfont={"size": 13},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8")),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
init_db()
clf, meta = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Phishing Detector")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if clf is not None and meta:
        st.markdown("**Model Information**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy",  f"{meta.get('accuracy', 0)*100:.1f}%")
            st.metric("Precision", f"{meta.get('precision', 0)*100:.1f}%")
        with col2:
            st.metric("Recall",   f"{meta.get('recall', 0)*100:.1f}%")
            st.metric("F1-Score", f"{meta.get('f1_score', 0)*100:.1f}%")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("**Algorithm**")
        st.code(meta.get("algorithm", "RandomForestClassifier"), language=None)

        st.markdown("**Training Set**")
        st.caption(f"{meta.get('train_samples', 0):,} samples")

        st.markdown("**Feature Count**")
        st.caption(f"{len(meta.get('feature_names', []))} features")
    else:
        st.warning("⚠️ Model not found.\nRun training first:")
        st.code("python model/train_model.py", language="bash")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("**Quick Test URLs**")
    st.caption("Copy-paste to try:")
    examples = {
        "✅ Legitimate": "https://www.github.com/user/repo",
        "🚨 Phishing #1": "http://paypal-secure-login.update-account.xyz/verify/billing",
        "🚨 Phishing #2": "http://192.168.1.1/login/verify?cmd=account&update=1",
        "🚨 Shortener":   "http://bit.ly/3xABCDE",
    }
    for label, url in examples.items():
        st.markdown(f"**{label}**")
        st.code(url, language=None)


# ── Main content ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  URL Checker", "📊  History"])

# ═══════ TAB 1: URL CHECKER ═══════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="hero-title">🛡️ Phishing URL Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">AI-powered URL safety analysis — no page visit required</div>',
        unsafe_allow_html=True
    )

    if clf is None:
        st.error(
            "**Model not trained yet.**\n\n"
            "Run this command first:\n```bash\npython model/train_model.py\n```"
        )
        st.stop()

    # URL input
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        url_input = st.text_input(
            label="URL",
            placeholder="https://example.com  or  http://suspicious-site.xyz/login/verify",
            label_visibility="collapsed",
        )
    with col_btn:
        check_btn = st.button("🔍 Check URL", use_container_width=True)

    if check_btn and url_input.strip():
        url = url_input.strip()

        with st.spinner("Analysing URL..."):
            try:
                # Feature extraction
                feat_dict = extract_features(url)
                feat_array = np.array(list(feat_dict.values())).reshape(1, -1)

                # Prediction
                pred  = clf.predict(feat_array)[0]
                proba = clf.predict_proba(feat_array)[0]
                phish_conf = proba[1]
                legit_conf = proba[0]

                is_phishing = bool(pred == 1)
                confidence  = phish_conf if is_phishing else legit_conf

                # Log to DB
                log_to_db(url, "Phishing" if is_phishing else "Legitimate", confidence)

            except Exception as e:
                st.error(f"Error analysing URL: {e}")
                st.stop()

        # ── Result display ─────────────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if is_phishing:
                st.markdown(f"""
                <div class="result-phishing">
                    <span class="result-icon">🚨</span>
                    <div class="result-label result-label-phishing">PHISHING DETECTED</div>
                    <div class="result-confidence">Confidence: {phish_conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.error(
                    "⚠️ **Warning:** This URL shows strong phishing indicators. "
                    "Do **not** enter personal information on this site."
                )
            else:
                st.markdown(f"""
                <div class="result-legitimate">
                    <span class="result-icon">✅</span>
                    <div class="result-label result-label-legit">LEGITIMATE</div>
                    <div class="result-confidence">Confidence: {legit_conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.success(
                    "✅ **Looks safe.** No phishing indicators detected. "
                    "Always verify the domain independently."
                )

        with col_gauge:
            st.plotly_chart(
                make_gauge(confidence, is_phishing),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Probability breakdown ──────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Legitimate Prob",  f"{legit_conf*100:.1f}%")
        c2.metric("🔴 Phishing Prob",    f"{phish_conf*100:.1f}%")
        c3.metric("📏 URL Length",       f"{feat_dict['url_length']} chars")
        c4.metric("🔢 Subdomains",       feat_dict["num_subdomains"])

        # ── Feature breakdown ─────────────────────────────────────────────
        with st.expander("🔬 Feature Breakdown (click to expand)", expanded=True):
            col_tbl, col_chart = st.columns([1, 2])

            with col_tbl:
                st.markdown("**Extracted Features**")
                feat_df = pd.DataFrame(
                    [{"Feature": k, "Value": v} for k, v in feat_dict.items()]
                )
                # Colour-code boolean flags
                def highlight_flag(val):
                    if isinstance(val, (int, float)) and val == 1:
                        return "color: #f87171; font-weight: 600"
                    return ""
                st.dataframe(
                    feat_df.style.applymap(highlight_flag, subset=["Value"]),
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                )

            with col_chart:
                st.markdown("**Feature Values vs. Importance**")
                st.caption("Red bars = high-importance features with non-zero values")
                st.plotly_chart(
                    make_feature_chart(feat_dict, meta),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    elif check_btn:
        st.warning("Please enter a URL to check.")


# ═══════ TAB 2: HISTORY ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 URL Check History")

    hist_df = fetch_history(200)

    if hist_df.empty:
        st.info("No URLs checked yet. Use the **URL Checker** tab to get started.")
    else:
        # Summary metrics
        total    = len(hist_df)
        phishing = (hist_df["result"] == "Phishing").sum()
        legit    = (hist_df["result"] == "Legitimate").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Checked",   total)
        c2.metric("🔴 Phishing",     phishing)
        c3.metric("🟢 Legitimate",   legit)
        c4.metric("Detection Rate",  f"{phishing/total*100:.1f}%" if total else "N/A")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        col_pie, col_table = st.columns([1, 2])

        with col_pie:
            st.markdown("**Result Distribution**")
            st.plotly_chart(
                make_history_pie(hist_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_table:
            st.markdown("**Recent Checks**")
            # Style result column
            def style_result(val):
                if val == "Phishing":
                    return "color: #f87171; font-weight: 600"
                return "color: #4ade80; font-weight: 600"

            styled = hist_df.style.applymap(style_result, subset=["result"])
            st.dataframe(styled, use_container_width=True, height=380, hide_index=True)

        if st.button("🗑️ Clear History", key="clear_hist"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM history")
            conn.commit()
            conn.close()
            st.rerun()
