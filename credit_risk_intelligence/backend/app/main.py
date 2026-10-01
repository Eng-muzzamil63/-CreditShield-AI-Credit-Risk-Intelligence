from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml import FEATURES, META, get_engine

app = FastAPI(title="CreditShield AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Applicant(BaseModel):
    age: int = Field(35, ge=18, le=90)
    annual_income: float = Field(80000, gt=0)
    loan_amount: float = Field(180000, gt=0)
    employment_years: float = Field(6, ge=0)
    credit_history_months: float = Field(84, ge=1)
    debt_to_income: float = Field(0.28, ge=0, le=1)
    credit_utilization: float = Field(0.32, ge=0, le=1)
    late_payments_12m: int = Field(0, ge=0, le=30)
    credit_inquiries_6m: int = Field(1, ge=0, le=30)
    savings_balance: float = Field(25000, ge=0)
    existing_debt: float = Field(15000, ge=0)
    dependents: int = Field(1, ge=0, le=12)
    monthly_expenses: float = Field(3500, gt=0)
    home_ownership: str = "Rent"
    loan_purpose: str = "Personal"
    employment_type: str = "Salaried"

@app.get("/")
def root():
    return {"name": "CreditShield AI", "status": "ok", "message": "Credit risk intelligence API"}

@app.get("/api/health")
def health():
    engine = get_engine()
    return {"status": "healthy", "model": engine.model_name, "rows": len(engine.customers)}

@app.get("/api/overview")
def overview():
    engine = get_engine()
    df = engine.customers.copy()
    probs = engine.model.predict_proba(df[FEATURES])[:, 1]
    high = int((probs >= 0.35).sum())
    critical = int((probs >= 0.60).sum())
    return {
        "portfolio": {
            "applications": len(df),
            "high_risk": high,
            "critical": critical,
            "avg_default_probability": round(float(probs.mean()), 4),
            "observed_default_rate": round(float(df.default.mean()), 4),
        },
        "model": {
            "name": engine.model_name,
            "roc_auc": engine.champion_metrics["roc_auc"],
            "accuracy": engine.champion_metrics["accuracy"],
            "precision": engine.champion_metrics["precision"],
            "recall": engine.champion_metrics["recall"],
            "f1": engine.champion_metrics["f1"],
        },
        "risk_distribution": [
            {"label": "Low", "count": int((probs < 0.15).sum())},
            {"label": "Moderate", "count": int(((probs >= 0.15) & (probs < 0.35)).sum())},
            {"label": "High", "count": int(((probs >= 0.35) & (probs < 0.60)).sum())},
            {"label": "Critical", "count": critical},
        ],
        "feature_importance": engine.feature_importance,
        "comparison": engine.comparison,
    }

@app.get("/api/applicants")
def applicants(
    risk: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
):
    engine = get_engine()
    df = engine.customers.copy()
    probs = engine.model.predict_proba(df[FEATURES])[:, 1]
    df["default_probability"] = probs
    df["risk_band"] = ["Low" if p < 0.15 else "Moderate" if p < 0.35 else "High" if p < 0.60 else "Critical" for p in probs]
    if risk:
        df = df[df.risk_band.str.lower() == risk.lower()]
    if search:
        s = search.lower()
        df = df[df.apply(lambda r: s in str(r.applicant_id).lower() or s in str(r.applicant_name).lower(), axis=1)]
    df = df.sort_values("default_probability", ascending=False).head(limit)
    cols = ["applicant_id", "applicant_name", "annual_income", "loan_amount", "debt_to_income", "credit_utilization", "late_payments_12m", "default_probability", "risk_band", "loan_purpose", "employment_type"]
    return df[cols].round(4).to_dict(orient="records")

@app.get("/api/applicants/{applicant_id}")
def applicant_detail(applicant_id: str):
    engine = get_engine()
    df = engine.customers
    match = df[df.applicant_id == applicant_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Applicant not found")
    raw = match.iloc[0]
    payload = {k: raw[k] for k in FEATURES}
    scored = engine.score(payload)
    return {"applicant_id": applicant_id, "applicant_name": raw.applicant_name, "profile": {k: raw[k].item() if hasattr(raw[k], "item") else raw[k] for k in FEATURES}, **scored}

@app.post("/api/score")
def score(payload: Applicant):
    engine = get_engine()
    return engine.score(payload.model_dump())
