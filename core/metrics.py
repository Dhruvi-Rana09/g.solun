import pandas as pd
import numpy as np

def compute_metrics(df: pd.DataFrame, label_col: str,
                    sensitive_col: str, positive_label) -> dict:

    y = (df[label_col] == positive_label).astype(int).values
    s = df[sensitive_col]
    groups = s.unique()

    group_rates = {}
    for g in groups:
        mask = s == g
        group_rates[str(g)] = round(float(y[mask].mean()), 4)

    rates = list(group_rates.values())
    max_rate = max(rates)
    min_rate = min(rates)

    # Metric 1: Disparate Impact
    di = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0
    di_pass = 0.8 <= di <= 1.25

    # Metric 2: Demographic Parity Difference
    dp = round(abs(max_rate - min_rate), 4)
    dp_pass = dp < 0.1

    # Metric 3: Equal Opportunity Difference
    tpr = {}
    for g in groups:
        mask = (s == g) & (y == 1)
        tpr[str(g)] = round(float(y[mask].mean()), 4) if mask.sum() > 0 else 0.0
    tpr_vals = list(tpr.values())
    eo = round(abs(max(tpr_vals) - min(tpr_vals)), 4)
    eo_pass = eo < 0.1

    # Metric 4: Average Odds Difference
    fpr = {}
    for g in groups:
        mask = (s == g) & (y == 0)
        fpr[str(g)] = round(float(y[mask].mean()), 4) if mask.sum() > 0 else 0.0
    fpr_vals = list(fpr.values())
    ao = round((abs(max(tpr_vals) - min(tpr_vals)) + abs(max(fpr_vals) - min(fpr_vals))) / 2, 4)
    ao_pass = ao < 0.1

    return {
        "group_rates": group_rates,
        "tpr_per_group": tpr,
        "fpr_per_group": fpr,
        "metrics": {
            "disparate_impact": {
                "value": di, "pass": di_pass,
                "range": "0.8 – 1.25",
                "explain": f"Least-favoured group gets outcomes at {di*100:.1f}% the rate of the best-off group."
            },
            "demographic_parity": {
                "value": dp, "pass": dp_pass,
                "range": "< 0.10",
                "explain": f"There is a {dp*100:.1f}% gap in positive outcome rates between groups."
            },
            "equal_opportunity": {
                "value": eo, "pass": eo_pass,
                "range": "< 0.10",
                "explain": f"Among qualified people, {eo*100:.1f}% fewer from the disadvantaged group get a positive decision."
            },
            "average_odds": {
                "value": ao, "pass": ao_pass,
                "range": "< 0.10",
                "explain": f"Error rates differ by {ao*100:.1f}% on average between groups."
            }
        }
    }