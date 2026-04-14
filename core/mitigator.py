import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from core.metrics import compute_metrics


def apply_reweighing(df: pd.DataFrame, label_col: str,
                     sensitive_col: str, positive_label) -> dict:
    """
    Apply the Reweighing pre-processing bias mitigation strategy.

    How it works:
      Each training sample is assigned a weight so that the (group, label)
      combination is represented at the rate expected under independence.
      A fresh logistic regression is then trained with these weights,
      and the resulting predictions are used to compute new fairness metrics.
    """
    y = (df[label_col] == positive_label).astype(int)
    s = df[sensitive_col]
    n = len(df)

    # Compute sample weights (Reweighing — Kamiran & Calders 2012)
    weights = pd.Series(1.0, index=df.index)
    for group in s.unique():
        for outcome in [0, 1]:
            mask = (s == group) & (y == outcome)
            if mask.sum() == 0:
                continue
            # Expected rate under independence: P(group) * P(outcome)
            expected = (s == group).mean() * (y == outcome).mean()
            actual = mask.mean()
            if actual > 0:
                weights[mask] = round(expected / actual, 6)

    # Build feature matrix (numeric only, fill NaN with median)
    features = df.drop(columns=[label_col, sensitive_col])
    features = features.select_dtypes(include=[np.number])

    if features.shape[1] == 0:
        return {"error": "No numeric feature columns found to retrain on. "
                         "Add at least one numeric column (e.g. score, income)."}

    features = features.fillna(features.median(numeric_only=True))

    # Check we have enough samples
    if len(features) < 20:
        return {"error": "Dataset too small for reweighing (need at least 20 rows)."}

    # Train logistic regression with fairness weights
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(features, y, sample_weight=weights)
    new_preds = model.predict(features)

    # Compute new metrics on the mitigated predictions
    df_temp = df.copy()
    df_temp[label_col] = new_preds
    # Use the actual positive_label — fix for the original bug where 1 was hardcoded
    new_metrics = compute_metrics(df_temp, label_col, sensitive_col, positive_label)

    # Calculate improvement summary
    old_metrics = compute_metrics(df, label_col, sensitive_col, positive_label)
    improvements = {}
    for m_name in old_metrics["metrics"]:
        old_val = old_metrics["metrics"][m_name]["value"]
        new_val = new_metrics["metrics"][m_name]["value"]
        old_pass = old_metrics["metrics"][m_name]["pass"]
        new_pass = new_metrics["metrics"][m_name]["pass"]
        improvements[m_name] = {
            "old_value": old_val,
            "new_value": new_val,
            "old_pass": old_pass,
            "new_pass": new_pass,
            "improved": (not old_pass and new_pass) or (new_val < old_val
                         if m_name != "disparate_impact" else new_val > old_val)
        }

    return {
        "strategy": "Reweighing",
        "description": (
            "Sample weights were adjusted so that each (protected group × outcome) "
            "combination influences the model proportionally to its expected frequency "
            "under statistical independence. A new logistic regression was then trained "
            "on the reweighted dataset."
        ),
        "weights_range": {
            "min": round(float(weights.min()), 3),
            "max": round(float(weights.max()), 3)
        },
        "new_metrics": new_metrics,
        "improvements": improvements,
        "features_used": list(features.columns)
    }