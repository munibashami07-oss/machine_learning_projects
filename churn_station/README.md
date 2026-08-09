# Customer Churn Prediction & Explainable Retention Engine

**Live Pakistan ML Internship — Week 1 of 4 (Complicated Tier)**

A production-style churn prediction pipeline for a telecom company: predicts which customers
will churn, explains *why* (SHAP), and estimates the business impact of acting on those
predictions.

## Project Structure

The project is split into 3 notebooks, each building on the last. Run them **in order** — each
one loads artifacts saved by the previous one.

```
Part1_Data_Preparation.ipynb          Data cleaning, EDA, feature engineering, preprocessing pipeline
Part2_Modeling_Evaluation.ipynb       5 models, imbalance handling, nested-CV tuning, evaluation
Part3_Explainability_Business.ipynb   SHAP explainability + business impact estimate

churn_utils.py                        Shared feature-engineering code (imported by all 3 notebooks
                                       and by the API — keeps the pipeline importable/picklable)
telco_customer_churn_raw_dataset.csv  Source dataset (IBM Telco Customer Churn, via Kaggle)

final_churn_prediction_pipeline.joblib   The deployable model (Stacking Ensemble) — raw customer
                                          data in, churn probability out
all_tuned_individual_models.joblib       All 5 tuned models, kept for SHAP (Part 3 uses the
                                          XGBoost one specifically — see Part 3 for why)

fastapi_predict_endpoint.py           Stretch goal: POST /predict API (probability + top-3 SHAP factors)
                                       Now includes CORS middleware so a browser-based UI (like
                                       signal_churn_console.html below) can call it directly.

signal_churn_console.html             Standalone 3D UI ("SIGNAL") for the API above. Open it in a
                                       browser while uvicorn is running -- it POSTs a customer
                                       profile to /predict and renders the churn probability on an
                                       animated 3D risk beacon (Three.js), plus the top contributing
                                       SHAP factors as directional bars. No build step, no install --
                                       just double-click the file.

Written_Summary_Imbalance_Model_Business.md   Half-to-one-page summary: imbalance decision,
                                                best model, business recommendation

screenshots/
  01_churn_class_distribution_eda.png
  02_model_comparison_metrics_table.png
  03_calibration_curve_stacking_ensemble.png
  04a_shap_global_feature_importance_bar.png
  04b_shap_global_feature_importance_beeswarm.png
  05a_shap_local_highest_risk_customer.png
  05b_shap_local_borderline_customer.png
  05c_shap_local_lowest_risk_customer.png
```

## Setup

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn jupyter joblib
```

## Running the Notebooks

All 3 notebooks already have their outputs saved (plots, tables, printed results) — you can just
open and read them without re-running anything.

If you want to re-run them from scratch:

1. Keep all files in this folder together (notebooks + `churn_utils.py` + the CSV).
2. **Note:** Part 1's data-load cell reads `WA_Fn-UseC_-Telco-Customer-Churn.csv` (the dataset's
   original filename), but this folder's copy has been renamed to
   `telco_customer_churn_raw_dataset.csv` for clarity. Either rename the CSV back, or edit that
   one line in Part 1 to match, before re-running.
3. Open with Jupyter / JupyterLab / VS Code and run **Part 1 → Part 2 → Part 3**, in that order.
   Each notebook writes its outputs to an `artifacts/` and `outputs/` subfolder that the next
   notebook reads from.

⚠️ Part 2's hyperparameter search grids were kept intentionally small (this project was built on
a single-CPU machine) — the methodology (nested CV, SMOTE vs. class-weight comparison, 5-model
comparison) is fully real, just not an exhaustive grid. Expect Part 2 to take a few minutes to
re-run; Parts 1 and 3 are quick.

## Running the API (Stretch Goal)

```bash
pip install fastapi uvicorn
uvicorn fastapi_predict_endpoint:app --reload
```

Server runs at `http://127.0.0.1:8000`. Interactive docs (easiest way to test): `http://127.0.0.1:8000/docs`.

## Running the 3D UI (SIGNAL console)

With the API running (above), just open `signal_churn_console.html` in a browser — double-click
it, or drag it into a browser window. No server, no build step. Fill in a customer profile on the
left and click **Analyze customer**; it POSTs to `http://127.0.0.1:8000/predict` (editable in the
top-right field if your server runs on a different port) and renders:

- An animated 3D "risk beacon" (Three.js) that shifts from green → amber → red with the predicted
  churn probability
- The top 3 SHAP factors for that specific customer, as directional bars (red = pushes toward
  churn, green = pushes toward retention)

If the beacon shows a connection error, it means the browser couldn't reach the API — check that
`uvicorn` is still running in its terminal, and that the URL in the top-right field matches it.

Or via curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85"
}'
```

Returns something like:
```json
{
  "churn_probability": 0.6633,
  "top_3_contributing_factors": [
    {"feature": "tenure", "shap_value": 0.7482},
    {"feature": "Contract_Month-to-month", "shap_value": 0.5447},
    {"feature": "InternetService_Fiber optic", "shap_value": -0.2225}
  ]
}
```

## Results Summary

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Stacking Ensemble (production)** | **0.629** | 0.554 | 0.589 | **0.842** | 0.639 |
| Gradient Boosting | 0.550 | 0.660 | 0.600 | 0.841 | 0.646 |
| XGBoost | 0.560 | 0.663 | 0.607 | 0.841 | **0.648** |
| Random Forest | 0.533 | **0.757** | 0.625 | 0.840 | 0.637 |
| Logistic Regression | 0.500 | 0.783 | 0.610 | 0.838 | 0.624 |

Targeting the top 10% highest-risk customers (per the production model) captures actual churners
at **2.78x lift** over random targeting, projecting to roughly **$148,000/year** in retained
revenue under a conservative 30%-success-rate retention campaign assumption.

Full reasoning, caveats, and the imbalance-handling decision are in
`Written_Summary_Imbalance_Model_Business.md`."# machine_learning" 
