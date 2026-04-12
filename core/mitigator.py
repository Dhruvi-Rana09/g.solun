import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from core.metrics import compute_metrics

def apply_reweighing(df: pd.DataFrame, label_col: str,
                     sensitive_col: str, positive_label) -> dict:
    y = (df[label_col] == positive_label).astype(int)
    s = df[sensitive_col]
    weights = pd.Series(1.0, index=df.index)

    for group in s.unique():
        for outcome in [0, 1]:
            mask = (s == group) & (y == outcome)
            expected = (s == group).mean() * (y == outcome).mean()
            actual = mask.mean()
            if actual > 0:
                weights[mask] = round(expected / actual, 4)

    features = df.drop(columns=[label_col, sensitive_col])
    features = features.select_dtypes(include=[np.number]).fillna(0)

    if features.shape[1] == 0:
        return {"error": "No numeric feature columns found to retrain on."}

    model = LogisticRegression(max_iter=1000)
    model.fit(features, y, sample_weight=weights)
    new_preds = model.predict(features)

    df_temp = df.copy()
    df_temp[label_col] = new_preds
    new_metrics = compute_metrics(df_temp, label_col, sensitive_col, 1)

    return {
        "strategy": "Reweighing",
        "description": "Sample weights adjusted so all groups influence the model equally.",
        "new_metrics": new_metrics
    }