"""
reports/generate_report.py
===========================
Phishing URL Detector — Evaluation Report Generator

Loads the trained model + metadata and generates:
  - reports/confusion_matrix.png
  - reports/feature_importance.png
  - reports/metrics_summary.txt

Usage:
    python reports/generate_report.py
"""

import sys
# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import pickle
import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
META_PATH  = os.path.join(PROJECT_ROOT, "model", "model_metadata.pkl")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT     = "#60a5fa"
RED_ACCENT = "#f87171"
GREEN_ACC  = "#4ade80"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "font.family":       "DejaVu Sans",
})


def load_metadata():
    if not os.path.exists(META_PATH):
        print("[ERROR] model_metadata.pkl not found. Run: python model/train_model.py")
        sys.exit(1)
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


# ── 1. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(meta: dict):
    cm = np.array(meta["confusion_matrix"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Cell colours and values
    cell_data = [
        # (row, col, value, bg_color, label, label_color)
        (1, 0, tn, "#1e3a2f", "TN", GREEN_ACC),   # top-left  = Legit/Legit
        (1, 1, fp, "#3a1e1e", "FP", RED_ACCENT),   # top-right = Legit/Phish
        (0, 0, fn, "#3a1e1e", "FN", RED_ACCENT),   # bot-left  = Phish/Legit
        (0, 1, tp, "#1e3a2f", "TP", GREEN_ACC),    # bot-right = Phish/Phish
    ]

    for (row, col, val, bg, lbl, lbl_color) in cell_data:
        rect = mpatches.FancyBboxPatch(
            (col + 0.04, row + 0.04), 0.92, 0.92,
            boxstyle="round,pad=0.02",
            facecolor=bg, edgecolor="none",
        )
        ax.add_patch(rect)
        ax.text(col + 0.5, row + 0.58, f"{val:,}",
                ha="center", va="center", fontsize=26,
                fontweight="bold", color=TEXT_COLOR)
        ax.text(col + 0.5, row + 0.28, lbl,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color=lbl_color)
        pct = val / total * 100 if total > 0 else 0
        ax.text(col + 0.5, row + 0.14, f"{pct:.1f}%",
                ha="center", va="center", fontsize=9, color="#94a3b8")

    # Axis labels
    for col, lbl in enumerate(["Legitimate", "Phishing"]):
        ax.text(col + 0.5, 2.08, lbl, ha="center", va="bottom",
                fontsize=11, color=TEXT_COLOR, fontweight="600")
    for row, lbl in enumerate(["Phishing", "Legitimate"]):
        ax.text(-0.06, row + 0.5, lbl, ha="right", va="center",
                fontsize=11, color=TEXT_COLOR, fontweight="600", rotation=90)

    ax.text(1.0, 2.22, "Predicted", ha="center", va="bottom",
            fontsize=12, color="#60a5fa", fontweight="bold")
    ax.text(-0.32, 1.0, "Actual", ha="center", va="center",
            fontsize=12, color="#60a5fa", fontweight="bold", rotation=90)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold",
                 color=TEXT_COLOR, pad=30)

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  [OK] Saved: {out}")


# ── 2. Feature Importance Chart ───────────────────────────────────────────────
def plot_feature_importance(meta: dict):
    imp_dict   = meta["feature_importances"]
    feat_names = meta["feature_names"]

    imps = [imp_dict.get(f, 0) for f in feat_names]
    sorted_pairs = sorted(zip(feat_names, imps), key=lambda x: x[1])

    names  = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    # Colour: top-5 features highlighted in accent blue, rest in muted
    colors = []
    threshold = sorted(values)[-5] if len(values) >= 5 else 0
    for v in values:
        colors.append(ACCENT if v >= threshold else "#374151")

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.65)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, color=TEXT_COLOR)

    ax.set_xlabel("Feature Importance (Gini)", fontsize=12, labelpad=10)
    ax.set_title("Random Forest — Feature Importance\n(Phishing URL Detection)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_xlim(0, max(values) * 1.18)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.grid(axis="x", alpha=0.2, color=GRID_COLOR)
    ax.grid(axis="y", visible=False)

    legend_patches = [
        mpatches.Patch(color=ACCENT,   label="Top-5 most important features"),
        mpatches.Patch(color="#374151", label="Other features"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10,
              facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "feature_importance.png")
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  [✓] Saved: {out}")


# ── 3. Metrics Bar Chart ──────────────────────────────────────────────────────
def plot_metrics(meta: dict):
    metrics = {
        "Accuracy":  meta.get("accuracy",  0),
        "Precision": meta.get("precision", 0),
        "Recall":    meta.get("recall",    0),
        "F1-Score":  meta.get("f1_score",  0),
    }

    names  = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]
    colors = [ACCENT, "#a78bfa", "#34d399", "#fbbf24"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    bars = ax.bar(names, values, color=colors, edgecolor="none", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=TEXT_COLOR)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=12, labelpad=10)
    ax.set_title("Model Evaluation Metrics", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR, pad=12)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.2, color=GRID_COLOR)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "metrics_chart.png")
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  [✓] Saved: {out}")


# ── 4. Text Metrics Summary ───────────────────────────────────────────────────
def write_metrics_summary(meta: dict):
    cm   = np.array(meta["confusion_matrix"])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    content = f"""
==============================================================
  PHISHING URL DETECTOR — MODEL EVALUATION SUMMARY
  Student  : Anil Kumar
  Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==============================================================

  Algorithm   : {meta.get('algorithm', 'RandomForestClassifier')}
  Estimators  : {meta.get('n_estimators', 200)}
  Features    : {len(meta.get('feature_names', []))}
  Train Samples: {meta.get('train_samples', 0):,}
  Test  Samples: {meta.get('test_samples',  0):,}

  ── Evaluation Metrics ──────────────────────────────────────
  Accuracy    : {meta.get('accuracy',  0)*100:.4f}%
  Precision   : {meta.get('precision', 0)*100:.4f}%
  Recall      : {meta.get('recall',    0)*100:.4f}%
  F1-Score    : {meta.get('f1_score',  0)*100:.4f}%

  ── Confusion Matrix ────────────────────────────────────────
                    Predicted
                    Legit   Phishing
  Actual  Legit    {tn:>6,}   {fp:>6,}    (TN / FP)
          Phishing {fn:>6,}   {tp:>6,}    (FN / TP)

  True  Positives (TP): {tp:,}   — Phishing correctly detected
  True  Negatives (TN): {tn:,}   — Legitimate correctly identified
  False Positives (FP): {fp:,}   — Legitimate flagged as Phishing
  False Negatives (FN): {fn:,}   — Phishing missed

  False Positive Rate : {fp/(fp+tn)*100:.2f}%
  False Negative Rate : {fn/(fn+tp)*100:.2f}%

  ── Top-10 Feature Importances ──────────────────────────────
"""
    imp_dict = meta.get("feature_importances", {})
    sorted_imp = sorted(imp_dict.items(), key=lambda x: -x[1])[:10]
    for rank, (feat, imp) in enumerate(sorted_imp, 1):
        bar = "█" * int(imp * 300)
        content += f"  {rank:>2}. {feat:<35} {imp:.4f}  {bar}\n"

    content += "\n" + "=" * 62 + "\n"

    out = os.path.join(REPORT_DIR, "metrics_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [✓] Saved: {out}")
    print(content)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("\n[INFO] Loading model metadata...")
    meta = load_metadata()

    print("\n[INFO] Generating reports...\n")
    plot_confusion_matrix(meta)
    plot_feature_importance(meta)
    plot_metrics(meta)
    write_metrics_summary(meta)

    print("\n✅ All reports generated in: reports/")
    print("   - confusion_matrix.png")
    print("   - feature_importance.png")
    print("   - metrics_chart.png")
    print("   - metrics_summary.txt")
