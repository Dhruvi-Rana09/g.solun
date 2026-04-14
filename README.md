# 🛡️ Unbiased AI Decision

> **Google Solution Challenge 2026** — Ensuring Fairness and Detecting Bias in Automated Decisions

## 🎯 Problem Statement

Computer programs now make life-changing decisions about who gets a job, a bank loan, or medical care. When these systems learn from historically biased data, they silently repeat and amplify discrimination at scale.

**This tool gives organisations a clear, accessible way to:**
- Inspect datasets and AI models for hidden unfairness
- Measure, flag, and understand bias using 4 industry-standard fairness metrics
- Apply automatic mitigation (Reweighing) and compare before vs after
- Download compliance-ready PDF reports
- Understand the real-world impact of bias with plain-English AI analysis

---

## ✨ Features

| Feature | Description |
|---|---|
| **📊 Dataset Audit** | Upload any CSV, select a sensitive column + outcome, and detect bias in seconds |
| **🤖 Model Audit** | Train a classifier and audit the fairness of its *predictions* (not just historical data) |
| **💡 AI Insights** | Optional Gemini integration for plain-English root cause analysis |
| **🔧 Reweighing** | Apply the Kamiran & Calders (2012) reweighing algorithm and see before/after improvement |
| **📄 PDF Report** | Compliance-ready PDF with metrics, AI insights, root causes, and legal checklist |
| **📋 Audit History** | Timeline view of all past audits with bias score trend |
| **⚠️ Proxy Warnings** | Detect features that correlate with the protected attribute (indirect discrimination) |

### Fairness Metrics Computed

| Metric | Threshold | Legal Framework |
|---|---|---|
| Disparate Impact | ≥ 0.80 | EEOC 80% Rule |
| Demographic Parity Difference | < 0.10 | EU AI Act |
| Equal Opportunity Difference | < 0.10 | ISO/IEC 42001 |
| Average Odds Difference | < 0.10 | ECOA / Fair Credit |

---

## 🚀 Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/Dhruvi-Rana09/g.solun.git
cd g.solun

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Add your Gemini API key for AI insights
# Enter it in the sidebar of the app, or set as environment variable:
# set GEMINI_API_KEY=your_key_here   (Windows)
# export GEMINI_API_KEY=your_key_here  (macOS/Linux)

# 6. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🏗️ Architecture

```
g.solun/
├── app.py                    # Main Streamlit application (4 pages)
├── core/
│   ├── inspector.py          # Dataset health checks & sensitive col detection
│   ├── metrics.py            # 4 fairness metrics (Disparate Impact, Parity, etc.)
│   ├── scorer.py             # Severity scoring (Low / Medium / High / Critical)
│   ├── mitigator.py          # Reweighing pre-processing mitigation
│   ├── model_auditor.py      # Train + audit ML model predictions
│   ├── gemini_insights.py    # Gemini AI analysis (with rule-based fallback)
│   └── reporter.py           # PDF report generation (fpdf2)
├── requirements.txt
└── README.md
```

---

## 📐 How the Metrics Work

### Disparate Impact
`DI = (positive rate of least-favoured group) / (positive rate of most-favoured group)`

EEOC 80% rule: DI must be ≥ 0.80. A value of 0.60 means the disadvantaged group is selected at only 60% the rate of the advantaged group.

### Demographic Parity Difference
`DPD = |P(decision=1 | group A) - P(decision=1 | group B)|`

Should be < 0.10 (10% gap). Measures raw selection rate inequality.

### Equal Opportunity Difference
`EOD = |TPR(group A) - TPR(group B)|`

Measures whether qualified people from different groups are treated equally.

### Average Odds Difference
`AOD = (|TPR_gap| + |FPR_gap|) / 2`

Captures two-sided unfairness — both excessive false rejections and false approvals.

---

## ⚖️ Compliance Frameworks Addressed

- **EU AI Act 2024** (Article 10 — Data governance for high-risk AI)
- **EEOC 80% Rule** (Uniform Guidelines on Employee Selection)
- **US Equal Credit Opportunity Act (ECOA)**
- **ISO/IEC 42001** (AI Management Systems)

---

## 🧪 Test with Sample Data

A pre-built biased hiring dataset generator is included:

```bash
python generate_sample_data.py
# Creates: sample_hiring_data.csv (500 rows, gender-biased hiring outcomes)
```

In the app: select `gender` as sensitive column, `hired` as label, `1` as positive label.

---

## 🤝 Contributing

Pull requests welcome! Please open an issue first to discuss changes.

---

*Built with ❤️ for the Google Solution Challenge 2026*
