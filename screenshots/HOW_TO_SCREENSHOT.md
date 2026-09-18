# 📸 Screenshot Guide — Phishing URL Detector

Save all screenshots **in this `screenshots/` folder** with the names listed below.

---

## Required Screenshots

| Filename | What to Capture |
|----------|-----------------|
| `01_app_homepage.png` | The Streamlit app main page (URL Checker tab, before any URL is entered) |
| `02_legitimate_result.png` | Result for a legitimate URL, e.g. `https://www.google.com` — showing green "LEGITIMATE" card |
| `03_phishing_result.png` | Result for a phishing URL, e.g. `http://paypal-secure-login.update-account.xyz/verify/billing` — showing red "PHISHING DETECTED" card |
| `04_feature_breakdown.png` | The expanded "Feature Breakdown" section showing the feature table and bar chart |
| `05_history_tab.png` | The History tab showing the pie chart and recent URL checks table |
| `06_sidebar_metrics.png` | Sidebar showing model accuracy, precision, recall, F1-score |
| `07_test_results.png` | Terminal/console output after running `python tests/test_urls.py` |
| `08_confusion_matrix.png` | The `reports/confusion_matrix.png` image (already generated) |
| `09_feature_importance.png` | The `reports/feature_importance.png` image (already generated) |
| `10_metrics_chart.png` | The `reports/metrics_chart.png` image (already generated) |

---

## How to Take Screenshots

### Windows
- **Snipping Tool**: Press `Windows + Shift + S` → select area → save as PNG
- **Print Screen**: Press `PrtScn` → paste into Paint → save

### Recommended Tool
Use the **Snipping Tool** (`Windows + Shift + S`) to capture specific regions.

---

## Steps to Capture App Screenshots

1. Run the model training if not done already:
   ```bash
   python model/train_model.py
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at **http://localhost:8501**

4. For each screenshot listed above, perform the action in the app and capture.

---

## Test URLs to Use for Screenshots

| URL | Type |
|-----|------|
| `https://www.google.com` | Legitimate |
| `https://github.com/user/repo` | Legitimate |
| `http://paypal-secure-login.update-account.xyz/verify/billing` | Phishing |
| `http://192.168.1.1/login/verify?cmd=account&update=1` | Phishing |
| `http://bit.ly/3xABCDE` | Phishing |
