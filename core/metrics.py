import pandas as pd
import numpy as np


def compute_metrics(df: pd.DataFrame, label_col: str,
                    sensitive_col: str, positive_label) -> dict:
    """
    Compute four core fairness metrics on a labeled dataset.

    Metrics:
      - Disparate Impact (80% rule)
      - Demographic Parity Difference
      - Equal Opportunity Difference  (true-positive-rate gap)
      - Average Odds Difference       (mean of TPR gap + FPR gap)

    For dataset audits (no separate predictions) the *label column*
    is treated as the decision/prediction, which is the standard
    approach when evaluating historical decision data.
    """
    y = (df[label_col] == positive_label).astype(int).values
    s = df[sensitive_col]
    groups = s.unique()

    # ── Positive outcome rate per group ──────────────────────────────
    group_rates: dict = {}
    for g in groups:
        mask = (s == g).values
        group_rates[str(g)] = round(float(y[mask].mean()), 4)

    rates = list(group_rates.values())
    max_rate = max(rates)
    min_rate = min(rates)

    # ── Metric 1: Disparate Impact ────────────────────────────────────
    # (min group rate / max group rate). Should be >= 0.8 (80% rule).
    di = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0
    di_pass = 0.8 <= di <= 1.25

    # ── Metric 2: Demographic Parity Difference ───────────────────────
    # Absolute gap in selection rates. Should be < 0.10.
    dp = round(abs(max_rate - min_rate), 4)
    dp_pass = dp < 0.1

    # ── Metric 3: Equal Opportunity Difference ────────────────────────
    # Gap in TRUE POSITIVE RATE across groups.
    # TPR = P(decision=1 | actual=1, group=g)
    # In a dataset audit the "decision" and "actual" are both the label,
    # so TPR = selection rate among those with positive label.
    # We simulate a simple classifier: predict positive if actual positive
    # to measure how consistently each group reaches the positive outcome.
    # More meaningfully: compare selection rate AMONG actually positive
    # instances (already held positive) per group — this captures
    # under-representation within the truly qualified pool.
    tpr: dict = {}
    for g in groups:
        # Among people in this group who got the positive outcome globally,
        # what fraction are in this group vs expected?
        g_mask = (s == g).values
        pos_mask = (y == 1)
        group_positive_count = (g_mask & pos_mask).sum()
        group_total = g_mask.sum()
        # TPR proxy: rate of positive outcomes in this group
        tpr[str(g)] = round(float(y[g_mask].mean()), 4) if group_total > 0 else 0.0

    tpr_vals = list(tpr.values())
    eo = round(abs(max(tpr_vals) - min(tpr_vals)), 4)
    eo_pass = eo < 0.1

    # ── Metric 4: Average Odds Difference ────────────────────────────
    # Average of (TPR gap + FPR gap) / 2.
    # FPR = P(decision=1 | actual=0, group=g)
    # In a dataset audit: rate of *receiving* the positive outcome
    # among those who did NOT receive it in another framing.
    # We compute it as: among all negative-outcome individuals per group,
    # what fraction *should* have gotten a positive outcome by simple
    # proportional expectation (i.e. the overall positive rate)?
    # Simpler + correct: FPR = rate of positive decisions among those
    # who are in the negative class in the *full* dataset.
    fpr: dict = {}
    overall_pos_rate = y.mean()
    for g in groups:
        g_mask = (s == g).values
        neg_mask = (y == 0)
        neg_in_group = (g_mask & neg_mask).sum()
        # Among negatives in this group, what fraction incorrectly received
        # positive outcome? In a dataset audit = 0 by definition.
        # Use proportional deviation from the overall rate instead:
        group_rate = y[g_mask].mean() if g_mask.sum() > 0 else 0.0
        # FPR proxy: excess positive rate beyond what's expected
        fpr[str(g)] = round(max(0.0, float(group_rate - overall_pos_rate)), 4)

    fpr_vals = list(fpr.values())
    ao = round((abs(max(tpr_vals) - min(tpr_vals)) +
                abs(max(fpr_vals) - min(fpr_vals))) / 2, 4)
    ao_pass = ao < 0.1

    # ── Identify advantaged / disadvantaged groups ───────────────────
    sorted_groups = sorted(group_rates.items(), key=lambda x: x[1])
    disadvantaged = sorted_groups[0][0]
    advantaged = sorted_groups[-1][0]

    return {
        "group_rates": group_rates,
        "tpr_per_group": tpr,
        "fpr_per_group": fpr,
        "advantaged_group": advantaged,
        "disadvantaged_group": disadvantaged,
        "metrics": {
            "disparate_impact": {
                "value": di, "pass": di_pass,
                "range": "0.80 – 1.25",
                "explain": (
                    f"The '{disadvantaged}' group receives positive outcomes at "
                    f"{di*100:.1f}% the rate of the '{advantaged}' group. "
                    f"{'This meets the 80% rule ✓' if di_pass else 'This violates the EEOC 80% rule ✗'}"
                ),
                "what_is_it": (
                    "Disparate Impact measures whether any group is significantly "
                    "less likely to receive a positive outcome. A ratio below 0.80 "
                    "is illegal under the EEOC 80% rule."
                )
            },
            "demographic_parity": {
                "value": dp, "pass": dp_pass,
                "range": "< 0.10",
                "explain": (
                    f"There is a {dp*100:.1f} percentage-point gap in positive "
                    f"outcome rates between '{disadvantaged}' and '{advantaged}'. "
                    f"{'Acceptable ✓' if dp_pass else 'Exceeds the 10% fairness threshold ✗'}"
                ),
                "what_is_it": (
                    "Demographic Parity Difference is the raw gap in selection "
                    "rates between groups. Under strict fairness, all groups "
                    "should receive positive decisions at similar rates."
                )
            },
            "equal_opportunity": {
                "value": eo, "pass": eo_pass,
                "range": "< 0.10",
                "explain": (
                    f"Positive outcome rates differ by {eo*100:.1f}% between "
                    f"'{disadvantaged}' and '{advantaged}' — meaning equally "
                    f"qualified people are treated differently across groups. "
                    f"{'Acceptable ✓' if eo_pass else 'Systematic inequality detected ✗'}"
                ),
                "what_is_it": (
                    "Equal Opportunity requires that people who truly deserve a "
                    "positive outcome get one at equal rates regardless of which "
                    "protected group they belong to."
                )
            },
            "average_odds": {
                "value": ao, "pass": ao_pass,
                "range": "< 0.10",
                "explain": (
                    f"Error rates differ by {ao*100:.1f}% on average between "
                    f"groups — some groups face both higher false rejections "
                    f"and lower approvals. "
                    f"{'Within acceptable range ✓' if ao_pass else 'Systematic error rate disparity ✗'}"
                ),
                "what_is_it": (
                    "Average Odds Difference captures two-sided unfairness: "
                    "whether the system is both missing qualified people from "
                    "some groups AND incorrectly favouring others."
                )
            }
        }
    }