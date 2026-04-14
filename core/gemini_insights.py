"""
Gemini AI-powered insights for bias audit results.
Gracefully falls back to rule-based insights if no API key is configured.
"""
import os


def _build_prompt(inspection: dict, bias_result: dict, severity: dict,
                  sensitive_col: str, label_col: str) -> str:
    m = bias_result["metrics"]
    group_rates = bias_result["group_rates"]
    adv = bias_result.get("advantaged_group", "")
    disadv = bias_result.get("disadvantaged_group", "")

    metrics_summary = "\n".join([
        f"- {name.replace('_', ' ').title()}: {data['value']} "
        f"({'PASS' if data['pass'] else 'FAIL'}) — {data['explain']}"
        for name, data in m.items()
    ])

    group_summary = "\n".join([
        f"- {group}: {rate*100:.1f}% positive outcome rate"
        for group, rate in group_rates.items()
    ])

    return f"""You are an AI fairness expert analysing an automated decision-making dataset.

AUDIT RESULTS:
- Dataset: {inspection['total_rows']} rows, {inspection['total_cols']} columns
- Protected attribute: '{sensitive_col}'
- Outcome variable: '{label_col}'
- Advantaged group: {adv}
- Disadvantaged group: {disadv}
- Overall bias severity: {severity['severity']} (score: {severity['score']}/100)

GROUP OUTCOME RATES:
{group_summary}

FAIRNESS METRICS:
{metrics_summary}

Please provide a structured analysis in this EXACT JSON format (no markdown, pure JSON):
{{
  "plain_summary": "2-3 sentence plain-English summary of what the bias means in real human terms",
  "root_causes": ["3 specific likely root causes of this bias in the data or decision process"],
  "human_impact": "1-2 sentence description of real-world harm this bias causes to affected people",
  "action_items": ["3 specific, actionable steps the organisation should take immediately"],
  "compliance_risk": "1 sentence on the legal/regulatory risk if this bias is not addressed",
  "analogy": "A simple 1-sentence analogy explaining this bias to a non-technical audience"
}}"""


def get_gemini_insights(inspection: dict, bias_result: dict, severity: dict,
                        sensitive_col: str, label_col: str) -> dict:
    """
    Generate AI-powered insights using Google Gemini.
    Returns a dict with keys: plain_summary, root_causes, human_impact,
    action_items, compliance_risk, analogy, source.
    Falls back to rule-based insights if Gemini is unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")

    if api_key and api_key != "your_gemini_api_key_here":
        try:
            return _call_gemini(api_key, inspection, bias_result, severity,
                                sensitive_col, label_col)
        except Exception as e:
            # Fall through to rule-based fallback
            print(f"[Gemini] API call failed: {e}. Using fallback insights.")

    return _rule_based_insights(bias_result, severity, sensitive_col, label_col)


def _call_gemini(api_key: str, inspection: dict, bias_result: dict,
                 severity: dict, sensitive_col: str, label_col: str) -> dict:
    import json
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = _build_prompt(inspection, bias_result, severity, sensitive_col, label_col)
    response = model.generate_content(prompt)

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    result = json.loads(text)
    result["source"] = "gemini"
    return result


def _rule_based_insights(bias_result: dict, severity: dict,
                         sensitive_col: str, label_col: str) -> dict:
    """
    Rule-based fallback insights when Gemini is not available.
    """
    m = bias_result["metrics"]
    adv = bias_result.get("advantaged_group", "the advantaged group")
    disadv = bias_result.get("disadvantaged_group", "the disadvantaged group")
    sev = severity["severity"]
    score = severity["score"]

    di_val = m["disparate_impact"]["value"]
    dp_val = m["demographic_parity"]["value"]

    # Plain summary
    plain_summary = (
        f"This dataset shows {sev.lower()} bias in '{label_col}' decisions "
        f"based on '{sensitive_col}'. "
        f"The '{disadv}' group receives positive outcomes at only "
        f"{di_val*100:.1f}% the rate of the '{adv}' group — "
        f"a gap of {dp_val*100:.1f} percentage points."
    )

    # Root causes
    root_causes = [
        f"Historical data reflects past discrimination against '{disadv}' in this domain.",
        f"Proxy variables (e.g. education, location, experience) may correlate with "
        f"'{sensitive_col}' and act as indirect channels for bias.",
        f"Selection or survivorship bias in how the training data was collected "
        f"may under-represent '{disadv}' in positive outcomes."
    ]

    # Human impact
    human_impact = (
        f"Real people from the '{disadv}' group are being systematically denied "
        f"'{label_col}' at a higher rate than equally qualified '{adv}' counterparts, "
        f"causing measurable harm to livelihoods, opportunities, and dignity."
    )

    # Action items
    action_items = []
    if not m["disparate_impact"]["pass"]:
        action_items.append(
            f"Apply Reweighing mitigation to balance '{sensitive_col}' group "
            f"influence during model training (use the 'Apply Fix' button above)."
        )
    if not m["demographic_parity"]["pass"]:
        action_items.append(
            f"Audit all features for correlation with '{sensitive_col}' — "
            f"remove or adjust proxy variables before retraining."
        )
    if not m["equal_opportunity"]["pass"]:
        action_items.append(
            "Deploy a Threshold Optimizer post-processing step to equalise "
            "true positive rates across all protected groups."
        )
    if not action_items:
        action_items = [
            "Continue monitoring bias metrics as new data arrives.",
            "Document this audit trail for regulatory compliance.",
            "Set up automated bias alerts if score exceeds 25/100."
        ]

    # Compliance risk
    if score >= 70:
        compliance_risk = (
            f"A Disparate Impact ratio of {di_val:.2f} constitutes a prima facie "
            f"violation of the EEOC 80% Rule and may expose the organisation to "
            f"EU AI Act Article 10 enforcement actions. Immediate remediation required."
        )
    elif score >= 40:
        compliance_risk = (
            f"The current bias score of {score}/100 is a compliance risk under "
            f"ECOA and the EU AI Act for high-risk AI systems. Document mitigation steps."
        )
    else:
        compliance_risk = (
            "Bias metrics are within acceptable ranges. Maintain audit logs "
            "to demonstrate ongoing compliance under ISO/IEC 42001."
        )

    # Analogy
    analogy = (
        f"Imagine two equally skilled job applicants: one from '{adv}' and "
        f"one from '{disadv}' — the system currently gives the '{adv}' applicant "
        f"a {di_val*100:.0f}% higher chance of success for the same qualifications."
    )

    return {
        "plain_summary": plain_summary,
        "root_causes": root_causes,
        "human_impact": human_impact,
        "action_items": action_items,
        "compliance_risk": compliance_risk,
        "analogy": analogy,
        "source": "rule-based"
    }
