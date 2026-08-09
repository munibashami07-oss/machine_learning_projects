"""
Shared preprocessing utilities for the Churn Prediction & Explainable Retention Engine.

Kept in its own importable module (rather than defined inline in a notebook's __main__)
so that the pickled/joblib'd pipeline can be loaded from any other notebook, script, or
a FastAPI/Flask service (see stretch goal) without an "AttributeError: Can't get attribute"
unpickling failure.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = [
    "PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
]

RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
ENGINEERED_NUMERIC = ["NumServices", "ChargePerService"]
NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "TenureGroup",
]


def count_services(frame: pd.DataFrame) -> pd.Series:
    def _count(row):
        n = 0
        for c in SERVICE_COLS:
            if row[c] == "Yes":
                n += 1
        if row.get("InternetService", "No") != "No":
            n += 1
        return n
    return frame.apply(_count, axis=1)


def tenure_group(frame: pd.DataFrame) -> pd.Series:
    bins = [-1, 12, 36, np.inf]
    labels = ["New(0-12mo)", "Established(12-36mo)", "Loyal(36mo+)"]
    return pd.cut(frame["tenure"], bins=bins, labels=labels)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleans TotalCharges and adds the 3 engineered features. Stateless -> safe on new data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # 1. Clean TotalCharges: blank/whitespace strings for tenure==0 customers -> 0.0
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )
        X["TotalCharges"] = X["TotalCharges"].fillna(0.0)

        # 2. NumServices
        X["NumServices"] = count_services(X)

        # 3. TenureGroup
        X["TenureGroup"] = tenure_group(X).astype(str)

        # 4. ChargePerService
        X["ChargePerService"] = X["MonthlyCharges"] / (X["NumServices"] + 1)

        return X

    def get_feature_names_out(self, input_features=None):
        return None
