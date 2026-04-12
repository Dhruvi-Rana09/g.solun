def compute_severity(metrics: dict) -> dict:
    m = metrics["metrics"]
    score = 0

    if not m["disparate_impact"]["pass"]:   score += 30
    if not m["demographic_parity"]["pass"]: score += 25
    if not m["equal_opportunity"]["pass"]:  score += 20
    if not m["average_odds"]["pass"]:       score += 15

    score = min(score, 100)

    if score <= 25:   severity, color = "Low",      "🟢"
    elif score <= 50: severity, color = "Medium",   "🟡"
    elif score <= 75: severity, color = "High",     "🟠"
    else:             severity, color = "Critical", "🔴"

    return {"score": score, "severity": severity, "color": color}


def generate_recommendations(metrics: dict, sensitive_col: str) -> list:
    m = metrics["metrics"]
    recs = []
    if not m["disparate_impact"]["pass"]:
        recs.append(f"Apply Reweighing to balance '{sensitive_col}' group influence on the model.")
    if not m["demographic_parity"]["pass"]:
        recs.append(f"Check if '{sensitive_col}' is directly or indirectly driving decisions.")
    if not m["equal_opportunity"]["pass"]:
        recs.append("Use Threshold Optimizer so qualified people from all groups get equal chance.")
    if not m["average_odds"]["pass"]:
        recs.append("Review false positive rates — some groups may be unfairly penalised.")
    if not recs:
        recs.append("All metrics pass. Keep monitoring with each new dataset version.")
    return recs