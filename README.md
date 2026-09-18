# 🛡️ Phishing URL Detection System

An AI-powered, machine-learning based phishing URL detection tool built entirely in Python.  
Analyzes URLs using **22 lexical and structural features** and classifies them as  
**Legitimate** or **Phishing** — without visiting the page or making external API calls.

---

## 👨‍💻 Student / Team Details

| Field | Details |
|-------|---------|
| **Student Name** | Anil Kumar |
| **Project Title** | Phishing URL Detection System |
| **Domain** | Cybersecurity — Threat Detection / Machine Learning |
| **Company / Institute** | YHills Edutech |
| **Guide / Trainer** | Sakshi Pandey |
| **Submission Date** | October 2026 |

---

## 📁 Project Structure

```
phishing-url-detector/
├── app.py                        # Streamlit web application (main entry point)
├── run_all.py                    # One-command pipeline: train → test → report
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── history.db                    # Auto-created: SQLite scan history
│
├── src/                          # Source package marker
│   └── __init__.py
│
├── data/
│   ├── generate_dataset.py       # Synthetic dataset generator (fallback)
│   └── phishing_dataset.csv      # Training/test dataset (10,000 URLs)
│
├── features/
│   └── extract_features.py       # 22 URL feature extraction functions
│
├── model/
│   ├── train_model.py            # Model training script
│   ├── phishing_model.pkl        # Saved trained model (created after training)
│   └── model_metadata.pkl        # Accuracy metrics + feature importances
│
├── tests/
│   └── test_urls.py              # 15 curated test URLs with expected results
│
├── reports/
│   ├── generate_report.py        # Generates charts + metrics summary
│   ├── confusion_matrix.png      # Created after running generate_report.py
│   ├── feature_importance.png    # Created after running generate_report.py
│   ├── metrics_chart.png         # Created after running generate_report.py
│   └── metrics_summary.txt       # Text summary of all evaluation metrics
│
└── screenshots/
    └── HOW_TO_SCREENSHOT.md      # Guide on what screenshots to capture
```

---

## ⚙️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Web UI | Streamlit 1.28+ |
| ML Model | scikit-learn — RandomForestClassifier |
| Data | pandas, numpy |
| Visualization | Plotly (UI), matplotlib, seaborn (reports) |
| URL Parsing | urllib.parse, tldextract |
| Storage | SQLite (scan history) |
| Serialization | pickle / joblib |

---

## 🖥️ System Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.9 or higher |
| RAM | 2 GB |
| Disk Space | 200 MB (including dataset + model) |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| Browser | Chrome, Firefox, Edge (for Streamlit UI) |

---

## 📦 Installation

### Step 1 — Clone or extract the project

```bash
# If cloning from repository:
git clone <repo-url>
cd phishing-url-detector

# Or simply extract the ZIP and cd into it
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### Option A — Full pipeline (recommended for first run)

```bash
python run_all.py
```

This single command:
1. Trains the Random Forest model (generates dataset if not present)
2. Runs 15 test cases and prints expected vs actual results
3. Generates confusion matrix, feature importance chart, and metrics summary

Then launch the web app:

```bash
python -m streamlit run app.py
```

### Option B — Step by step

```bash
# Step 1: Train the model
python model/train_model.py

# Step 2: Run test suite
python tests/test_urls.py

# Step 3: Generate report charts
python reports/generate_report.py

# Step 4: Launch the web application
python -m streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔬 Sample Input

**Input**: A raw URL string in the text field

```
http://paypal-secure-login.update-account.xyz/verify/billing
```

---

## 📤 Expected Output

**Result**: `🚨 PHISHING DETECTED` with confidence percentage

The system also displays:
- Confidence gauge chart (0–100%)
- Probability breakdown (legitimate vs phishing)
- Table of all 22 extracted features with values
- Feature importance bar chart (red = high-risk features active)

**For a legitimate URL** (`https://www.google.com`):  
→ `✅ LEGITIMATE` with ~95%+ confidence

**For a phishing URL** (`http://paypal-secure-login.update-account.xyz/verify/billing`):  
→ `🚨 PHISHING DETECTED` with ~90%+ confidence

---

## 🔬 Features Extracted (22 total)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `url_length` | Total character length of the URL |
| 2 | `domain_length` | Length of the hostname/domain |
| 3 | `path_length` | Length of the URL path |
| 4 | `count_dots` | Number of `.` characters |
| 5 | `count_hyphens` | Number of `-` characters |
| 6 | `count_underscores` | Number of `_` characters |
| 7 | `count_at` | Number of `@` characters |
| 8 | `count_digits` | Total digit count in URL |
| 9 | `count_slashes` | Number of `/` characters |
| 10 | `count_question_marks` | Number of `?` characters |
| 11 | `count_equals` | Number of `=` characters |
| 12 | `count_ampersands` | Number of `&` characters |
| 13 | `count_percent` | Number of `%` characters (URL encoding) |
| 14 | `uses_https` | 1 if HTTPS scheme, else 0 |
| 15 | `is_ip_address` | 1 if domain is raw IPv4/IPv6 address |
| 16 | `has_at_symbol` | 1 if `@` appears (browser redirects after `@`) |
| 17 | `has_double_slash_redirect` | 1 if `//` appears in path |
| 18 | `uses_non_standard_port` | 1 if non-80/443 port is specified |
| 19 | `is_url_shortener` | 1 if domain is a known URL shortener |
| 20 | `num_subdomains` | Count of subdomain levels |
| 21 | `digit_to_letter_ratio` | Ratio of digits to letters in URL |
| 22 | `suspicious_keyword_count` | Count of phishing keywords found |
| 23 | `has_suspicious_keyword` | Binary flag: any suspicious keyword present |

---

## 📊 Dataset Source

| Property | Value |
|----------|-------|
| **Source** | Synthetically generated (`data/generate_dataset.py`) |
| **Size** | 10,000 URLs (5,000 legitimate + 5,000 phishing) |
| **Columns** | `url` (string), `label` (0 = legitimate, 1 = phishing) |
| **Format** | CSV |
| **Split** | 80% training / 20% testing (stratified) |
| **Balance** | Balanced class weights used during training |

**Alternative real datasets:**
- Kaggle: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
- UCI: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+website

Place either as `data/phishing_dataset.csv` with columns `url` and `label`.

---

## 🧪 Test URLs (15 Cases)

| # | URL | Expected | Category |
|---|-----|----------|----------|
| 1 | `https://www.google.com` | ✅ Legitimate | Clean HTTPS, trusted domain |
| 2 | `https://github.com/user/repo` | ✅ Legitimate | Standard repo URL |
| 3 | `https://www.wikipedia.org/wiki/Phishing` | ✅ Legitimate | Trusted domain |
| 4 | `https://stackoverflow.com/questions/tagged/python` | ✅ Legitimate | Developer site |
| 5 | `https://www.microsoft.com/en-us/windows` | ✅ Legitimate | Official HTTPS |
| 6 | `https://www.youtube.com/watch?v=dQw4w9WgXcQ` | ✅ Legitimate | Legitimate query params |
| 7 | `https://www.linkedin.com/in/profile` | ✅ Legitimate | LinkedIn profile |
| 8 | `http://paypal-secure-login.update-account.xyz/verify/billing` | 🚨 Phishing | Brand impersonation + bad TLD |
| 9 | `http://192.168.1.1/login/verify?cmd=account&update=1` | 🚨 Phishing | Raw IP + suspicious keywords |
| 10 | `http://bit.ly/3xABCDE` | 🚨 Phishing | URL shortener hides destination |
| 11 | `http://amazon.com.phish-site.tk/signin/confirm` | 🚨 Phishing | amazon.com as subdomain trick |
| 12 | `http://secure.paypal-verify.ml/webscr?cmd=_login-submit` | 🚨 Phishing | Fake PayPal + suspicious TLD |
| 13 | `http://netflix-account-suspended.top/update/billing` | 🚨 Phishing | Brand + 'suspended' + bad TLD |
| 14 | `http://login.microsoft.com.verify-account.xyz/password/reset` | 🚨 Phishing | Microsoft impersonation chain |
| 15 | `http://185.220.101.50/download/install/setup.exe` | 🚨 Phishing | IP-based malware URL |

---

## ⚠️ Known Limitations

1. **Lexical analysis only** — The system does not visit or render the URL page. It cannot detect phishing based on page content, login forms, or visual spoofing.
2. **Short legitimate domains** may occasionally trigger false positives due to URL structure overlap with phishing patterns.
3. **Zero-day phishing domains** using clean HTTPS and no suspicious keywords may be missed (false negatives).
4. **URL shorteners** are flagged categorically — some legitimate use cases (e.g., marketing links) may be over-flagged.
5. **Dataset** is synthetically generated; a real-world dataset (Kaggle/UCI) will yield better generalization.
6. **No WHOIS / DNS lookup** — domain age, registrar, and DNS reputation are not checked.
7. **No SSL certificate validation** — HTTPS presence is a positive signal but does not confirm legitimacy.

---

## 🤖 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Estimators | 200 decision trees |
| Max Features | `sqrt` |
| Class Weight | `balanced` |
| Random State | 42 |
| Train/Test Split | 80% / 20% (stratified) |
| Expected Accuracy | ~95–98% (synthetic dataset) |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.28 | Web UI |
| `scikit-learn` | ≥1.3 | RandomForest ML model |
| `pandas` | ≥2.0 | Data loading and manipulation |
| `numpy` | ≥1.24 | Numerical arrays |
| `joblib` | ≥1.3 | Model serialization |
| `tldextract` | ≥3.4 | Accurate TLD/subdomain parsing |
| `plotly` | ≥5.15 | Interactive charts in UI |
| `matplotlib` | ≥3.7 | Report chart generation |
| `seaborn` | ≥0.12 | Styled report charts |
| `requests` | ≥2.31 | HTTP utilities |

---

## ⚠️ Disclaimer

This tool performs **lexical and structural analysis only** — it does not visit URLs or make any network requests to the target.  
High confidence scores indicate likely phishing, but this tool is **not a substitute** for full security analysis.  
Always verify URLs through multiple trusted sources before accessing them.

---

## 📚 References

1. Phishing.org — "What is Phishing?" — https://www.phishing.org/what-is-phishing
2. APWG — Anti-Phishing Working Group Reports — https://apwg.org
3. scikit-learn RandomForestClassifier — https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
4. Kaggle Phishing Dataset — https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
5. UCI PhiUSIIL Phishing URL Dataset — https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+website
6. Streamlit Documentation — https://docs.streamlit.io
7. tldextract library — https://github.com/john-kurkowski/tldextract
8. Mohammad, R.M., Thabtah, F., McCluskey, L. (2014) — "Predicting phishing websites based on self-structuring neural network" — Neural Computing and Applications
