# CreditShield AI — Credit Risk Intelligence

A portfolio-ready ML product for practical credit-risk analytics. It demonstrates classification, model comparison, probability scoring, feature engineering, explainability, portfolio monitoring, and API deployment.

## Stack
- Python 3.12+
- FastAPI
- Pandas / NumPy
- scikit-learn
- Logistic Regression + Random Forest
- Next.js / React / TypeScript

## ML flow
Synthetic borrower data → data generation → feature engineering via preprocessing pipeline → model comparison → held-out evaluation → champion selection by ROC-AUC → probability scoring → local feature contribution explanation → REST API → dashboard.

## Run backend
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs

## Run frontend
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:3000

## Demo note
The dataset is synthetic and intentionally designed to create realistic risk relationships for a portfolio demonstration. It is not trained on real borrower data and should not be used for real lending decisions.
