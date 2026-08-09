# Customer Churn Prediction & Explainable Retention Engine — Written Summary

**Live Pakistan ML Internship — Week 1**

## Imbalance-Handling Decision

The dataset is moderately imbalanced (26.5% churn / 73.5% retained). Two strategies were compared
on identical Logistic Regression pipelines under 5-fold stratified cross-validation:

| Strategy | F1 (churn class) | ROC-AUC | PR-AUC |
|---|---|---|---|
| SMOTE (synthetic oversampling) | higher | comparable | comparable |
| `class_weight='balanced'` | lower | comparable | comparable |

**SMOTE was selected** and applied uniformly across all 5 models, so every model in the final
comparison was evaluated under the same imbalance-handling approach.

## Best Model

Five models were trained and tuned via nested cross-validation (outer 3-fold for an unbiased
performance estimate, inner 3-fold `RandomizedSearchCV` for hyperparameter search): Logistic
Regression, Random Forest, Gradient Boosting, XGBoost, and a Stacking Ensemble of the top 3.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Stacking Ensemble** | **0.629** | 0.554 | 0.589 | **0.842** | 0.639 |
| Gradient Boosting | 0.550 | 0.660 | 0.600 | 0.841 | 0.646 |
| XGBoost | 0.560 | 0.663 | 0.607 | 0.841 | **0.648** |
| Random Forest | 0.533 | **0.757** | 0.625 | 0.840 | 0.637 |
| Logistic Regression | 0.500 | 0.783 | 0.610 | 0.838 | 0.624 |

The **Stacking Ensemble** was selected as the production model — it has the best ROC-AUC and by
far the best precision, meaning retention offers targeted using its ranking are less likely to be
wasted on customers who weren't actually going to leave. Its calibration curve sits close to the
diagonal, so its predicted probabilities are reasonably trustworthy in an absolute sense, not just
useful for ranking. The trade-off: it has lower recall than Random Forest or Logistic Regression,
i.e. it misses more true churners than those higher-recall alternatives — worth revisiting if the
business would rather cast a wider net at the cost of more false positives.

## Business Recommendation

Ranking the held-out test customers by the Stacking Ensemble's predicted churn probability and
targeting the **top 10% highest-risk** customers concentrates actual churners at **2.78x** the
base rate (73.8% churn within that cohort vs. 26.5% overall) — clear evidence the model's ranking
has real value for prioritizing retention outreach over targeting customers at random.

Assuming a conservative 30% success rate for a retention campaign (e.g. a discount or win-back
offer) aimed at this cohort, and scaling from the test sample to the full ~7,043-customer base:

- **Estimated annual retained revenue: ~$148,000** (based on the billing of customers in the
  top-risk cohort who actually churned, times the assumed 30% save rate, times 12 months).

This is a projection, not a guarantee — it assumes the test set is representative of the full
customer base and that the assumed 30% success rate holds in a real campaign. **Recommendation:**
run a small pilot campaign against the top-decile cohort, measure the actual save rate, and use
that real number to refine this estimate before committing budget at scale.

The SHAP analysis shows the drivers behind this cohort are consistent and actionable: **being on
a month-to-month contract**, **short tenure**, **lacking online security / tech support add-ons**,
and **paying by electronic check** are the strongest churn signals — all of which point toward
concrete retention levers (contract-upgrade incentives, bundling security/support add-ons, and
nudging customers off manual electronic-check billing).
