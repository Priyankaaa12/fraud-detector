from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import sys
import sqlite3
from datetime import datetime
from typing import Optional
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from explain import FraudExplainer

from api.models import TransactionInput, PredictionResponse, HealthResponse

app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL = None
EXPLAINER = None
LLM_EXPLAINER = None
DB_PATH = "fraud_alerts.db"


@app.on_event("startup")
async def load_models():
    global MODEL, EXPLAINER, LLM_EXPLAINER
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    try:
        print("🔄 Loading fraud detection model...")
        MODEL = joblib.load(os.path.join(model_dir, 'fraud_model.pkl'))
        print("✅ Model loaded")
        print("🔄 Loading SHAP explainer...")
        EXPLAINER = joblib.load(os.path.join(model_dir, 'explainer.pkl'))
        print("✅ SHAP explainer loaded")
        try:
            from llm import FraudLLMExplainer
            LLM_EXPLAINER = FraudLLMExplainer()
            print("✅ LLM explainer loaded")
        except Exception as e:
            print(f"⚠️  LLM explainer not available: {e}")
        _init_db()
        print("✅ Database initialized")
        print("\n🚀 Fraud Detection API is ready!")
    except FileNotFoundError:
        print("⚠️  Model files not found. Run python src/train.py first.")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT, amount REAL, fraud_probability REAL,
        risk_score INTEGER, risk_level TEXT, action TEXT,
        explanation TEXT, top_factor TEXT, timestamp TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


@app.get("/")
async def root():
    return {"service": "Fraud Detection API", "status": "running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=MODEL is not None,
        explainer_loaded=EXPLAINER is not None, llm_available=LLM_EXPLAINER is not None,
        timestamp=datetime.now().isoformat())


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput, include_llm: bool = True):
    if MODEL is None or EXPLAINER is None:
        return _demo_response(transaction)
    try:
        shap_result = EXPLAINER.explain_transaction(transaction.to_feature_dict(), top_n=5)
        action = shap_result['risk_level']
        action_reason = f"Based on risk score of {shap_result['risk_score']}/100"
        llm_explanation = None

        if LLM_EXPLAINER and include_llm:
            try:
                llm_result = LLM_EXPLAINER.generate_explanation(
                    shap_result, transaction_id=transaction.transaction_id or 'UNKNOWN',
                    amount=transaction.TransactionAmt or 0)
                llm_explanation = llm_result['explanation']
                action = llm_result['action']
                action_reason = llm_result['action_reason']
            except Exception as e:
                llm_explanation = f"Explanation unavailable: {str(e)}"

        response = PredictionResponse(
            transaction_id=transaction.transaction_id or f"TXN-{datetime.now().strftime('%H%M%S')}",
            fraud_probability=shap_result['fraud_probability'],
            risk_score=shap_result['risk_score'],
            risk_level=shap_result['risk_level'],
            action=action, action_reason=action_reason,
            explanation=llm_explanation or _fallback_explanation(shap_result),
            top_factors=shap_result['top_factors'],
            base_probability=shap_result['base_probability'],
            timestamp=datetime.now().isoformat())

        _save_alert(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/alerts")
async def get_recent_alerts(limit: int = 50, risk_level: Optional[str] = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM alerts"
        params = []
        if risk_level:
            query += " WHERE risk_level = ?"
            params.append(risk_level.upper())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = conn.cursor()
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return {"count": len(rows), "alerts": [dict(zip(columns, r)) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("""SELECT COUNT(*),
            SUM(CASE WHEN risk_level='CRITICAL' THEN 1 ELSE 0 END),
            SUM(CASE WHEN risk_level='HIGH' THEN 1 ELSE 0 END),
            SUM(CASE WHEN risk_level='MEDIUM' THEN 1 ELSE 0 END),
            SUM(CASE WHEN risk_level='LOW' THEN 1 ELSE 0 END),
            COALESCE(SUM(amount),0), COALESCE(AVG(fraud_probability),0)
            FROM alerts WHERE created_at >= datetime('now','-24 hours')""").fetchone()
        conn.close()
        return {"period": "last_24_hours", "total_alerts": row[0],
            "by_risk_level": {"CRITICAL": row[1], "HIGH": row[2], "MEDIUM": row[3], "LOW": row[4]},
            "total_flagged_amount": round(row[5], 2), "avg_fraud_probability": round(row[6], 4)}
    except Exception as e:
        return {"error": str(e), "total_alerts": 0}


def _save_alert(response: PredictionResponse):
    try:
        conn = sqlite3.connect(DB_PATH)
        top_factor = response.top_factors[0]['description'] if response.top_factors else ''
        conn.execute("""INSERT INTO alerts
            (transaction_id, amount, fraud_probability, risk_score, risk_level,
             action, explanation, top_factor, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (response.transaction_id, 0, response.fraud_probability, response.risk_score,
             response.risk_level, response.action, response.explanation, top_factor, response.timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB save error: {e}")


def _fallback_explanation(shap_result: dict) -> str:
    factors = shap_result.get('top_factors', [])
    if not factors:
        return f"Fraud probability: {shap_result['fraud_probability']:.1%}."
    return (f"Flagged with {shap_result['fraud_probability']:.1%} fraud probability. "
            f"Primary risk: {factors[0]['description']}. Level: {shap_result['risk_level']}.")


def _demo_response(transaction: TransactionInput) -> PredictionResponse:
    import random
    prob = random.uniform(0.1, 0.95)
    return PredictionResponse(
        transaction_id=transaction.transaction_id or "DEMO-001",
        fraud_probability=round(prob, 4), risk_score=int(prob * 100),
        risk_level="HIGH" if prob > 0.6 else "MEDIUM",
        action="REVIEW_URGENT" if prob > 0.6 else "MONITOR",
        action_reason="Demo mode", explanation="Demo mode: Run python src/train.py first.",
        top_factors=[], base_probability=0.035, timestamp=datetime.now().isoformat())
