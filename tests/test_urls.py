"""
tests/test_urls.py
==================
Phishing URL Detector — Test Suite
Runs 15 curated URLs through the trained model and reports
expected vs. actual results, plus a pass/fail summary.

Usage:
    python tests/test_urls.py

Output:
    Console table + reports/test_results.txt
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

import numpy as np
from features.extract_features import extract_features

MODEL_PATH  = os.path.join(PROJECT_ROOT, "model", "phishing_model.pkl")
META_PATH   = os.path.join(PROJECT_ROOT, "model", "model_metadata.pkl")
REPORT_DIR  = os.path.join(PROJECT_ROOT, "reports")
RESULT_FILE = os.path.join(REPORT_DIR, "test_results.txt")

# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (url, expected_label)  — 0 = Legitimate, 1 = Phishing
TEST_CASES = [
    # ── Legitimate URLs ──────────────────────────────────────────────────────
    ("https://www.google.com",                                         0, "Google homepage — clean HTTPS, no suspicious keywords"),
    ("https://github.com/user/repo",                                   0, "GitHub repo — standard structure"),
    ("https://www.wikipedia.org/wiki/Phishing",                        0, "Wikipedia article — trusted domain"),
    ("https://stackoverflow.com/questions/tagged/python",              0, "Stack Overflow — well-known developer site"),
    ("https://www.microsoft.com/en-us/windows",                        0, "Microsoft official — standard HTTPS"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ",                   0, "YouTube — legitimate query params"),
    ("https://www.linkedin.com/in/profile",                            0, "LinkedIn profile — legitimate path"),

    # ── Phishing URLs ────────────────────────────────────────────────────────
    ("http://paypal-secure-login.update-account.xyz/verify/billing",   1, "Brand impersonation + suspicious TLD"),
    ("http://192.168.1.1/login/verify?cmd=account&update=1",           1, "Raw IP address + suspicious keywords"),
    ("http://bit.ly/3xABCDE",                                          1, "URL shortener — hides destination"),
    ("http://amazon.com.phish-site.tk/signin/confirm",                 1, "amazon.com as subdomain trick"),
    ("http://secure.paypal-verify.ml/webscr?cmd=_login-submit",       1, "Fake PayPal — suspicious keywords + bad TLD"),
    ("http://netflix-account-suspended.top/update/billing",            1, "Brand + 'suspended' keyword + bad TLD"),
    ("http://login.microsoft.com.verify-account.xyz/password/reset",  1, "Microsoft impersonation subdomain chain"),
    ("http://185.220.101.50/download/install/setup.exe",               1, "IP-based malware distribution URL"),
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found. Run: python model/train_model.py")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    return clf, meta


def run_tests(clf) -> list:
    results = []
    for url, expected, note in TEST_CASES:
        try:
            feats = extract_features(url)
            feat_array = np.array(list(feats.values())).reshape(1, -1)
            pred   = clf.predict(feat_array)[0]
            proba  = clf.predict_proba(feat_array)[0]
            conf   = proba[int(pred)]
            actual = int(pred)
            passed = (actual == expected)
        except Exception as e:
            actual = -1
            conf   = 0.0
            passed = False
            note   = f"ERROR: {e}"

        results.append({
            "url":      url,
            "expected": expected,
            "actual":   actual,
            "confidence": conf,
            "passed":   passed,
            "note":     note,
        })
    return results


def label_str(val: int) -> str:
    return "Phishing  " if val == 1 else "Legitimate"


def print_and_save(results: list, meta: dict):
    os.makedirs(REPORT_DIR, exist_ok=True)

    header = (
        "=" * 110 + "\n"
        "  PHISHING URL DETECTOR — TEST RESULTS\n"
        f"  Student : Anil Kumar\n"
        f"  Date    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Model   : {meta.get('algorithm', 'RandomForestClassifier')}\n"
        f"  Accuracy: {meta.get('accuracy', 0)*100:.2f}%\n"
        "=" * 110
    )

    col_hdr = (
        f"  {'#':<3} {'Expected':<12} {'Actual':<12} {'Conf':>6}  {'Pass':^5}  "
        f"{'URL':<55} Note"
    )
    sep = "-" * 110

    rows = []
    for i, r in enumerate(results, 1):
        tick = "PASS" if r["passed"] else "FAIL"
        rows.append(
            f"  {i:<3} {label_str(r['expected']):<12} {label_str(r['actual']):<12} "
            f"{r['confidence']*100:>5.1f}%  [{tick}]  "
            f"{r['url'][:54]:<55} {r['note'][:40]}"
        )

    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    summary = (
        "\n" + "=" * 110 + "\n"
        f"  SUMMARY:  {passed}/{total} passed   |   {failed} failed\n"
        f"  Pass Rate: {passed/total*100:.1f}%\n"
        + "=" * 110
    )

    # FP / FN analysis
    fp = [r for r in results if r["expected"] == 0 and r["actual"] == 1]
    fn = [r for r in results if r["expected"] == 1 and r["actual"] == 0]

    fp_fn = "\n  False Positive Analysis (Legitimate URL classified as Phishing):\n"
    if fp:
        for r in fp:
            fp_fn += f"    ⚠  {r['url']}\n"
    else:
        fp_fn += "    None — no false positives on this test set.\n"

    fp_fn += "\n  False Negative Analysis (Phishing URL classified as Legitimate):\n"
    if fn:
        for r in fn:
            fp_fn += f"    ⚠  {r['url']}\n"
    else:
        fp_fn += "    None — no false negatives on this test set.\n"

    full_output = "\n".join([header, col_hdr, sep] + rows + [summary, fp_fn])

    print(full_output)

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(full_output)

    print(f"\n  Results saved -> {RESULT_FILE}")


if __name__ == "__main__":
    print("\n[INFO] Loading model...")
    clf, meta = load_model()

    print(f"[INFO] Running {len(TEST_CASES)} test cases...\n")
    results = run_tests(clf)

    print_and_save(results, meta)
