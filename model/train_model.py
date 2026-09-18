"""
Train a Random Forest classifier on URL features for phishing detection.

Usage:
    python model/train_model.py

Outputs:
    model/phishing_model.pkl   – trained model
    model/model_metadata.pkl   – accuracy, feature names, etc.
"""

import os
import sys
import time
import pickle
import warnings

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from features.extract_features import extract_features, feature_names

DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "phishing_dataset.csv")
MODEL_PATH  = os.path.join(PROJECT_ROOT, "model", "phishing_model.pkl")
META_PATH   = os.path.join(PROJECT_ROOT, "model", "model_metadata.pkl")


# ── 1. Load / generate dataset ───────────────────────────────────────────────
def load_dataset():
    if not os.path.exists(DATA_PATH):
        print("[INFO] Dataset not found. Generating synthetic dataset...")
        from data.generate_dataset import generate_dataset
        generate_dataset(n_legit=5000, n_phishing=5000, output_path=DATA_PATH)

    print(f"[INFO] Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Support both 'label' and 'result' column names
    label_col = None
    for candidate in ["label", "result", "class", "phishing", "target"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(f"Could not find label column. Columns: {df.columns.tolist()}")

    # Support both 'url' and 'urls' column names
    url_col = None
    for candidate in ["url", "urls", "address", "link"]:
        if candidate in df.columns:
            url_col = candidate
            break

    if url_col is None:
        # Dataset may already be pre-extracted (all numeric)
        print("[INFO] No URL column found — assuming dataset is already feature-extracted.")
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values
        y = df[label_col].values
        # Encode string labels if needed
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
        return X, y, feature_cols

    print(f"[INFO] URL column: '{url_col}', Label column: '{label_col}'")
    print(f"[INFO] Dataset size: {len(df)} rows")
    print(f"[INFO] Label distribution:\n{df[label_col].value_counts()}\n")

    # Drop rows with missing URLs
    df = df.dropna(subset=[url_col])

    print("[INFO] Extracting features from URLs (this may take a moment)...")
    start = time.time()

    feature_list = []
    failed = 0
    for i, url in enumerate(df[url_col].astype(str)):
        try:
            feature_list.append(extract_features(url))
        except Exception:
            feature_list.append({k: 0 for k in feature_names()})
            failed += 1
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(f"  Processed {i+1}/{len(df)} URLs... ({elapsed:.1f}s)")

    print(f"[INFO] Feature extraction complete. Failed: {failed}")

    X = pd.DataFrame(feature_list).values
    y = df[label_col].values

    # Encode string labels: -1/1 → 0/1, or 'phishing'/'legitimate' → 1/0
    if y.dtype == object or (hasattr(y, 'dtype') and y.dtype.kind not in 'iuf'):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(int)
        # Remap -1 → 0 if dataset uses -1 for legitimate
        if -1 in y:
            y = np.where(y == -1, 0, y)

    return X, y, feature_names()


# ── 2. Train model ───────────────────────────────────────────────────────────
def train(X, y, feat_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[INFO] Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Phishing"]))

    # Feature importance
    importances = clf.feature_importances_
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    print("  Top-10 Feature Importances:")
    for name, imp in feat_imp[:10]:
        bar = "#" * int(imp * 200)
        print(f"    {name:<35} {imp:.4f}  {bar}")

    print("=" * 60)

    metadata = {
        "algorithm":       "RandomForestClassifier",
        "n_estimators":    200,
        "accuracy":        acc,
        "precision":       prec,
        "recall":          rec,
        "f1_score":        f1,
        "confusion_matrix": cm.tolist(),
        "feature_names":   feat_names,
        "feature_importances": dict(zip(feat_names, importances.tolist())),
        "train_samples":   len(X_train),
        "test_samples":    len(X_test),
    }

    return clf, metadata


# ── 3. Save model ────────────────────────────────────────────────────────────
def save(clf, metadata):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"\n[INFO] Model saved  -> {MODEL_PATH}")
    print(f"[INFO] Metadata saved -> {META_PATH}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, feat_names = load_dataset()
    clf, metadata = train(X, y, feat_names)
    save(clf, metadata)
    print("\n✅  Training complete! Run the app with:  streamlit run app.py")
