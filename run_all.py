"""
run_all.py
==========
Phishing URL Detector — Full Pipeline Runner

Runs the entire project pipeline in order:
  1. Generate dataset (if not present)
  2. Train the model
  3. Run the 15-URL test suite
  4. Generate evaluation report charts

Usage:
    python run_all.py

After completion:
    streamlit run app.py
"""

import os
import sys
import subprocess
import time

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

BANNER = r"""
  ____  _     _     _     _              _   _ ____  _       
 |  _ \| |__ (_)___| |__ (_)_ __   __ _| | | |  _ \| |      
 | |_) | '_ \| / __| '_ \| | '_ \ / _` | | | | |_) | |      
 |  __/| | | | \__ \ | | | | | | | (_| | |_| |  _ <| |___   
 |_|   |_| |_|_|___/_| |_|_|_| |_|\__, |\___/|_| \_\_____|  
                                   |___/                      
           Phishing URL Detector — Full Pipeline Runner
           Student: Anil Kumar
"""


def run_step(step_num: int, description: str, cmd: list, cwd: str) -> bool:
    print(f"\n{'-'*60}")
    print(f"  STEP {step_num}: {description}")
    print(f"{'-'*60}")
    start = time.time()

    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✅ Step {step_num} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌ Step {step_num} FAILED (exit code {result.returncode})")
        return False


def main():
    print(BANNER)
    print("=" * 62)

    steps_ok = 0

    # ── Step 1: Train model (also auto-generates dataset if needed) ──────────
    ok = run_step(
        step_num=1,
        description="Train Random Forest model (generates dataset if needed)",
        cmd=[sys.executable, os.path.join(PROJECT_ROOT, "model", "train_model.py")],
        cwd=PROJECT_ROOT,
    )
    if ok:
        steps_ok += 1

    # ── Step 2: Run test suite ───────────────────────────────────────────────
    ok = run_step(
        step_num=2,
        description="Run 15-URL test suite",
        cmd=[sys.executable, os.path.join(PROJECT_ROOT, "tests", "test_urls.py")],
        cwd=PROJECT_ROOT,
    )
    if ok:
        steps_ok += 1

    # ── Step 3: Generate evaluation report ──────────────────────────────────
    ok = run_step(
        step_num=3,
        description="Generate evaluation report (confusion matrix, charts)",
        cmd=[sys.executable, os.path.join(PROJECT_ROOT, "reports", "generate_report.py")],
        cwd=PROJECT_ROOT,
    )
    if ok:
        steps_ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  PIPELINE COMPLETE: {steps_ok}/3 steps succeeded")
    print("=" * 62)

    if steps_ok == 3:
        print("""
  Everything is ready! To launch the web application run:

      streamlit run app.py

  Then open: http://localhost:8501
""")
    else:
        print("\n  ⚠  Some steps failed. Check the output above for details.\n")


if __name__ == "__main__":
    main()
