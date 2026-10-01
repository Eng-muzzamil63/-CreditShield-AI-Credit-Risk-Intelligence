from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = np.random.default_rng(42)

NUMERIC = [
    "age", "annual_income", "loan_amount", "employment_years",
    "credit_history_months", "debt_to_income", "credit_utilization",
    "late_payments_12m", "credit_inquiries_6m", "savings_balance",
    "existing_debt", "dependents", "monthly_expenses",
]
CATEGORICAL = ["home_ownership", "loan_purpose", "employment_type"]
FEATURES = NUMERIC + CATEGORICAL

@dataclass
class RiskEngine:
    model: Pipeline
    model_name: str
    champion_metrics: dict[str, float]
    comparison: list[dict[str, float | str]]
    feature_importance: list[dict[str, Any]]
    customers: pd.DataFrame
    train_rate: float

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = pd.DataFrame([payload])[FEATURES]
        prob = float(self.model.predict_proba(row)[0, 1])
        band = "Low" if prob < 0.15 else "Moderate" if prob < 0.35 else "High" if prob < 0.60 else "Critical"
        decision = "Eligible for standard review" if prob < 0.15 else "Manual underwriting recommended" if prob < 0.35 else "Enhanced review required" if prob < 0.60 else "High-risk application — senior review"
        contributions = explain_row(self.model, row)
        return {
            "default_probability": round(prob, 4),
            "risk_band": band,
            "decision": decision,
            "drivers": contributions,
        }


def generate_data(n: int = 6500) -> pd.DataFrame:
    age = RNG.integers(21, 70, n)
    income = np.clip(RNG.lognormal(mean=11.05, sigma=0.48, size=n), 18000, 320000)
    employment_years = np.clip(RNG.gamma(2.8, 2.2, n), 0, 35)
    credit_history = np.clip((age - 18) * 11 + RNG.normal(0, 35, n), 6, 600)
    loan = np.clip(income * RNG.uniform(0.45, 3.8, n) + RNG.normal(0, 9000, n), 3000, 500000)
    dti = np.clip(RNG.beta(2.4, 5.2, n) * 0.95 + 0.02, 0.01, 0.98)
    utilization = np.clip(RNG.beta(2.2, 3.5, n), 0.01, 0.99)
    late = RNG.poisson(0.75, n)
    inquiries = RNG.poisson(1.35, n)
    savings = np.clip(RNG.lognormal(9.1, 1.0, n), 0, 900000)
    existing_debt = np.clip(income * RNG.uniform(0.05, 2.4, n), 0, 550000)
    dependents = RNG.integers(0, 5, n)
    expenses = np.clip(income / 12 * RNG.uniform(0.35, 0.92, n), 600, 24000)
    home = RNG.choice(["Own", "Mortgage", "Rent"], n, p=[0.27, 0.34, 0.39])
    purpose = RNG.choice(["Home", "Auto", "Education", "Business", "Personal"], n, p=[0.24, 0.22, 0.17, 0.17, 0.20])
    emp_type = RNG.choice(["Salaried", "Self-employed", "Contract", "Hourly"], n, p=[0.56, 0.17, 0.17, 0.10])

    # Business-like synthetic default process: higher leverage, delinquency and utilization increase risk;
    # stable income, savings and credit depth reduce risk.
    logit = (
        -5.25
        + 2.9 * dti
        + 2.0 * utilization
        + 0.34 * late
        + 0.13 * inquiries
        + 0.0000042 * loan
        + 0.0000022 * existing_debt
        + 0.000018 * expenses
        - 0.0000027 * savings
        - 0.0015 * credit_history
        - 0.025 * employment_years
        + 0.018 * dependents
        + (home == "Rent") * 0.38
        + (emp_type == "Contract") * 0.22
        + (emp_type == "Hourly") * 0.34
        + (purpose == "Personal") * 0.18
        + (purpose == "Business") * 0.07
        + RNG.normal(0, 0.7, n)
    )
    p = 1 / (1 + np.exp(-logit))
    default = RNG.binomial(1, np.clip(p, 0.01, 0.88))

    df = pd.DataFrame({
        "age": age, "annual_income": income.round(0), "loan_amount": loan.round(0),
        "employment_years": employment_years.round(1), "credit_history_months": credit_history.round(0),
        "debt_to_income": dti.round(3), "credit_utilization": utilization.round(3),
        "late_payments_12m": late, "credit_inquiries_6m": inquiries,
        "savings_balance": savings.round(0), "existing_debt": existing_debt.round(0),
        "dependents": dependents, "monthly_expenses": expenses.round(0),
        "home_ownership": home, "loan_purpose": purpose, "employment_type": emp_type,
        "default": default,
    })
    df.insert(0, "applicant_id", [f"CR-{100000+i}" for i in range(n)])
    df.insert(1, "applicant_name", [f"Applicant {i+1:04d}" for i in range(n)])
    return df


def build_models(df: pd.DataFrame) -> tuple[RiskEngine, dict[str, Any]]:
    X = df[FEATURES]
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, stratify=y, random_state=42
    )

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1400, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=260, max_depth=10, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
    }
    trained: dict[str, Pipeline] = {}
    comparison = []
    for name, estimator in models.items():
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        comparison.append({
            "model": name,
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        })
        trained[name] = pipe

    champion_info = max(comparison, key=lambda x: x["roc_auc"])
    champion_name = str(champion_info["model"])
    champion = trained[champion_name]
    importances = feature_importance(champion)
    engine = RiskEngine(
        model=champion,
        model_name=champion_name,
        champion_metrics=champion_info,
        comparison=comparison,
        feature_importance=importances,
        customers=df,
        train_rate=float(y.mean()),
    )
    return engine, {"rows": len(df), "defaults": int(y.sum()), "default_rate": float(y.mean())}


def feature_importance(pipe: Pipeline) -> list[dict[str, Any]]:
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    if hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        vals = np.asarray(model.feature_importances_)
    order = np.argsort(vals)[::-1]
    out = []
    for i in order[:14]:
        n = names[i].replace("num__", "").replace("cat__", "").replace("_", " ")
        out.append({"feature": n.title(), "importance": round(float(vals[i]), 4)})
    return out


def explain_row(pipe: Pipeline, row: pd.DataFrame) -> list[dict[str, Any]]:
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]
    transformed = pre.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = list(pre.get_feature_names_out())
    x = transformed[0]
    if hasattr(model, "coef_"):
        signed = x * model.coef_[0]
    else:
        signed = x * model.feature_importances_
    idx = np.argsort(np.abs(signed))[::-1]
    result = []
    for i in idx:
        if len(result) >= 6:
            break
        raw = names[i].replace("num__", "").replace("cat__", "")
        impact = float(signed[i])
        result.append({
            "feature": raw.replace("_", " ").title(),
            "direction": "increases risk" if impact > 0 else "reduces risk",
            "impact": round(abs(impact), 4),
        })
    return result


def get_engine() -> RiskEngine:
    global ENGINE
    if ENGINE is None:
        df = generate_data()
        ENGINE, META = build_models(df)
    return ENGINE

ENGINE: RiskEngine | None = None
META: dict[str, Any] = {}
