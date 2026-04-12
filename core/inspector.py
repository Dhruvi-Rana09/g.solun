import pandas as pd

SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "age", "ethnicity",
    "religion", "nationality", "disability", "marital"
]

def inspect_dataset(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    total_cols = len(df.columns)

    null_pct = (df.isnull().sum() / total_rows * 100).round(2).to_dict()

    likely_sensitive = [
        col for col in df.columns
        if any(kw in col.lower() for kw in SENSITIVE_KEYWORDS)
    ]

    class_balance = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        class_balance[col] = df[col].value_counts(normalize=True).round(3).to_dict()

    avg_null = sum(null_pct.values()) / total_cols if total_cols > 0 else 0
    health_score = round(max(0, 100 - avg_null), 1)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "columns": list(df.columns),
        "null_pct": null_pct,
        "likely_sensitive": likely_sensitive,
        "class_balance": class_balance,
        "health_score": health_score
    }