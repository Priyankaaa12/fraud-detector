# Real-Time Financial Fraud Detection with AI Explanations

> An end-to-end ML system that detects fraudulent transactions and generates plain-English explanations using LLMs — built for financial compliance teams.

## What This Project Does

Most fraud detection systems just return a score. This system tells you **why**.

A compliance officer sees:
> *"This transaction is flagged primarily because the amount ($4,200) is 21x higher than this cardholder's typical range. Combined with a 3 AM timestamp and the account being accessed for the first time in 0 days (immediate repeat usage), these signals are consistent with an account takeover pattern. Recommend immediate card freeze pending verification."*

Not just: **Score: 87/100**

## Architecture

```
Transaction JSON
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  FastAPI    │───▶│  LightGBM   │───▶│  SHAP Explainer  │
│  Backend    │    │  Model       │    │  (Feature Values) │
└─────────────┘    └──────────────┘    └──────────────────┘
      │                                         │
      │                                         ▼
      │                               ┌──────────────────┐
      │                               │   Claude API     │
      │                               │  (LLM Narrative) │
      │                               └──────────────────┘
      │                                         │
      ▼                                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                     │
│  Live Feed │ Alerts │ Analysis │ Risk Charts            │
└─────────────────────────────────────────────────────────┘
```

## Results

| Metric | Score |
|--------|-------|
| AUC-ROC | > 0.93 |
| Average Precision | > 0.75 |
| Inference Time | < 200ms |
| LLM Response Time | < 2s |

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| ML Model | LightGBM | Best on tabular data, fast CPU training |
| Explainability | SHAP (TreeExplainer) | Fast, exact, industry standard |
| LLM | Claude API | Best narrative quality |
| API | FastAPI | Auto-docs, type validation, async |
| Dashboard | Streamlit | Fast to build, great for demos |
| Database | SQLite | Zero config, portable |

## Quick Start

### 1. Clone and set up
```bash
git clone https://github.com/yourusername/fraud-detector.git
cd fraud-detector
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Get the dataset
Download IEEE-CIS Fraud Detection from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) and place CSV files in `data/`.

### 3. Add your API key
Create `.env`:
```
ANTHROPIC_API_KEY=your_key_here
```

### 4. Train the model (~10 min on CPU)
```bash
python src/train.py
```

### 5. Start everything
```bash
# Terminal 1: API
uvicorn api.main:app --reload

# Terminal 2: Dashboard  
streamlit run dashboard/app.py

# Terminal 3: Live simulator
python src/simulator.py
```

### 6. Open the dashboard
Navigate to `http://localhost:8501`

## API Usage

```bash
# Test a transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-001",
    "TransactionAmt": 4200.0,
    "hour": 3,
    "day_of_week": 6,
    "D1": 0,
    "C1": 50
  }'
```

Response:
```json
{
  "fraud_probability": 0.87,
  "risk_score": 87,
  "risk_level": "CRITICAL",
  "action": "BLOCK_IMMEDIATELY",
  "explanation": "This transaction shows multiple high-risk signals...",
  "top_factors": [...]
}
```

Full API docs: `http://localhost:8000/docs`

## Project Structure

```
fraud-detector/
├── data/                  # Dataset (not committed to git)
├── src/
│   ├── preprocess.py      # Data cleaning, feature engineering, SMOTE
│   ├── train.py           # LightGBM training pipeline
│   ├── explain.py         # SHAP explainability engine
│   ├── llm.py             # Claude API integration
│   └── simulator.py       # Transaction stream generator
├── api/
│   ├── main.py            # FastAPI endpoints
│   └── models.py          # Pydantic request/response models
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── models/                # Saved model artifacts (not committed)
├── .env                   # API keys (not committed)
├── requirements.txt
└── README.md
```

## What I Learned

- **Explainable AI matters more than accuracy** — a 98% accurate black box is less useful than a 93% accurate model you can explain to regulators
- **Class imbalance is the main challenge** in fraud detection — SMOTE + threshold tuning beat accuracy as a metric
- **LLMs add real value** when given structured inputs (SHAP values) rather than raw data
- **FastAPI + Streamlit** is a practical stack for ML demos in job interviews

## System Requirements

- Windows / macOS / Linux
- Python 3.10+
- 8GB RAM (no GPU required)
- ~1GB disk space

---


