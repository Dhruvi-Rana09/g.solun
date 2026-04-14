"""
Model Audit module — trains a simple classifier on the dataset
and evaluates fairness of its *predictions* (not just historical labels).
This addresses the challenge requirement: "inspect software models".
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from core.metrics import compute_metrics


MODEL_OPTIONS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


def audit_model(df: pd.DataFrame, label_col: str, sensitive_col: str,
                positive_label, model_name: str = "Random Forest",
                test_size: float = 0.3) -> dict:
    """
    Trains a model on the dataset (excluding the sensitive attribute),
    then evaluates fairness of its predictions on the held-out test set.

    Returns:
        dict with model performance, fairness metrics, and feature importances.
    """
    y_raw = df[label_col]
    is_binary = y_raw.nunique() <= 2

    # Encode label
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    pos_encoded = le.transform([positive_label])[0]

    # Build feature matrix (drop label + sensitive to see proxy-free baseline)
    features_drop = [label_col, sensitive_col]
    features_with_sensitive = df.drop(columns=[label_col]).copy()
    features_no_sensitive = df.drop(columns=features_drop).copy()

    # Encode categorical features
    def encode_df(feat_df):
        for col in feat_df.select_dtypes(include=["object", "category"]).columns:
            feat_df[col] = LabelEncoder().fit_transform(feat_df[col].astype(str))
        return feat_df.fillna(feat_df.median(numeric_only=True))

    X_with = encode_df(features_with_sensitive.copy())
    X_without = encode_df(features_no_sensitive.copy())

    if X_without.shape[1] == 0:
        return {"error": "No numeric feature columns available (excluding sensitive col and label)."}

    # Train / test split (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_without, y, test_size=test_size, random_state=42, stratify=y
    )
    # Keep sensitive col for fairness eval on test set
    s_te = df.loc[X_te.index, sensitive_col]

    # Select and train model
    if model_name not in MODEL_OPTIONS:
        model_name = "Random Forest"
    model = MODEL_OPTIONS[model_name]
    model.fit(X_tr, y_tr)

    # Predictions on test set
    y_pred = model.predict(X_te)
    acc = round(accuracy_score(y_te, y_pred), 4)

    # Build a temp dataframe with predictions for fairness metrics
    df_te = X_te.copy()
    df_te[sensitive_col] = s_te.values
    df_te["__prediction__"] = le.inverse_transform(y_pred)
    df_te["__actual__"] = le.inverse_transform(y_te)

    # Fairness metrics on model PREDICTIONS
    try:
        fairness = compute_metrics(df_te, "__prediction__", sensitive_col, positive_label)
    except Exception as e:
        fairness = {"error": str(e)}

    # Feature importances
    feat_importances = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_importances = dict(sorted(
            zip(X_without.columns.tolist(), importances.tolist()),
            key=lambda x: x[1], reverse=True
        ))
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        feat_importances = dict(sorted(
            zip(X_without.columns.tolist(), coefs.tolist()),
            key=lambda x: x[1], reverse=True
        ))

    # Sensitive attribute correlation check (using full dataset)
    correlation_warnings = []
    for feat_col in X_without.columns:
        try:
            corr = df[feat_col].corr(pd.Categorical(df[sensitive_col]).codes.astype(float)
                                     if df[sensitive_col].dtype == object
                                     else df[sensitive_col].astype(float))
            if abs(corr) > 0.3:
                correlation_warnings.append({
                    "feature": feat_col,
                    "correlation": round(float(corr), 3),
                    "risk": "High proxy risk" if abs(corr) > 0.5 else "Moderate proxy risk"
                })
        except Exception:
            continue

    return {
        "model_name": model_name,
        "accuracy": acc,
        "test_size": len(X_te),
        "train_size": len(X_tr),
        "features_used": list(X_without.columns),
        "fairness_metrics": fairness,
        "feature_importances": feat_importances,
        "proxy_warnings": sorted(correlation_warnings, key=lambda x: abs(x["correlation"]), reverse=True),
        "note": (
            "Fairness metrics computed on MODEL PREDICTIONS (not historical labels). "
            "This shows the bias the model will introduce when deployed."
        )
    }
