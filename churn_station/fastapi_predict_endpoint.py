"""
Stretch goal (optional, extra credit): FastAPI endpoint wrapping the trained churn pipeline.

Run locally with:
    pip install fastapi uvicorn
    uvicorn app:app --reload

Then POST a customer's raw JSON data to /predict, e.g.:
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85"
    }'
"""
import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from churn_utils import ChurnFeatureEngineer  # noqa: F401  (needed to unpickle the pipeline)

app = FastAPI(title="Churn Prediction API")

# Allows a browser-based UI (e.g. the SIGNAL console, or any local dev server) to call this
# API from a different origin. "*" is fine for local development; if you deploy this beyond
# your own machine, replace it with the specific origin(s) that should be allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Production model (Stacking Ensemble) -- used for the headline probability
PRODUCTION_MODEL = joblib.load("final_churn_prediction_pipeline.joblib")

# XGBoost individual model -- used only to compute the top-3 SHAP contributing factors
# (see Part 3 notebook for why: TreeExplainer is fast/exact for tree models, unlike the
# heterogeneous Stacking Ensemble, which would need a much slower model-agnostic explainer).
_all_pipelines = joblib.load("all_tuned_individual_models.joblib")
XGB_PIPELINE = _all_pipelines["XGBoost"]
_fe_step = XGB_PIPELINE.named_steps["feature_engineering"]
_prep_step = XGB_PIPELINE.named_steps["preprocessing"]
_clf_step = XGB_PIPELINE.named_steps["clf"]
_explainer = shap.TreeExplainer(_clf_step)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  # kept as str -- matches the raw dataset's quirk, cleaned by the pipeline


@app.post("/predict")
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])

    churn_probability = float(PRODUCTION_MODEL.predict_proba(X)[:, 1][0])

    X_fe = _fe_step.transform(X)
    X_arr = _prep_step.transform(X_fe)
    if hasattr(X_arr, "toarray"):
        X_arr = X_arr.toarray()
    feature_names = [f.split("__", 1)[-1] for f in _prep_step.get_feature_names_out()]
    X_shap_df = pd.DataFrame(X_arr, columns=feature_names)

    sv = _explainer(X_shap_df).values[0]
    top3_idx = sorted(range(len(sv)), key=lambda i: abs(sv[i]), reverse=True)[:3]
    top_factors = [
        {"feature": feature_names[i], "shap_value": round(float(sv[i]), 4)}
        for i in top3_idx
    ]

    return {
        "churn_probability": round(churn_probability, 4),
        "top_3_contributing_factors": top_factors,
    }