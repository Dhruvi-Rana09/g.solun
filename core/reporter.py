from fpdf import FPDF
from datetime import datetime
import re


def _s(text: str) -> str:
    """Strip characters outside Latin-1 (emojis etc.) so fpdf Helvetica font won't crash."""
    return re.sub(r'[^\x00-\xff]', '', str(text))


class AuditPDF(FPDF):
    """Custom PDF class with header and footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        self.set_fill_color(32, 33, 36)
        self.rect(0, 0, 210, 16, "F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_y(4)
        # Using new_x and new_y to ensure x is reset properly
        self.cell(0, 8, "UNBIASED AI DECISION -- AUDIT REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10,
                  _s(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
                     f"Page {self.page_no()} of {{nb}}  |  Google Solution Challenge 2026"),
                  align="C")

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(60, 64, 67)
        self.set_fill_color(241, 243, 244)
        self.cell(0, 9, _s(f"  {text}"), new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)

    def body_text(self, text: str, color=(0, 0, 0)):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*color)
        # Use explicit width to avoid x-position issues
        self.multi_cell(self.epw, 6, _s(text), new_x="LMARGIN", new_y="NEXT")

    def kv_row(self, key: str, value: str, bold_value: bool = False):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(55, 7, _s(f"  {key}:"), new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "B" if bold_value else "", 10)
        self.set_text_color(0, 0, 0)
        self.cell(0, 7, _s(value), new_x="LMARGIN", new_y="NEXT")


def generate_pdf(inspection: dict, bias_result: dict, severity: dict,
                 recommendations: list, sensitive_col: str, label_col: str,
                 insights: dict = None) -> str:
    """
    Generate a comprehensive, styled PDF audit report.
    Returns the path to the saved PDF file.
    """
    pdf = AuditPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Cover info ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(32, 33, 36)
    pdf.ln(4)
    pdf.cell(0, 12, "AI Bias Audit Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6,
             _s(f"Protected Attribute: {sensitive_col}   |   "
                f"Outcome Variable: {label_col}   |   "
                f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}"),
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Severity banner
    sev = severity["severity"]
    score = severity["score"]
    banner_colors = {
        "Low": (24, 128, 56),
        "Medium": (234, 179, 8),
        "High": (242, 153, 0),
        "Critical": (217, 48, 37)
    }
    r, g, b = banner_colors.get(sev, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, _s(f"  BIAS SEVERITY: {sev.upper()}   --   SCORE: {score}/100"),
             new_x="LMARGIN", new_y="NEXT", fill=True, align="C")
    pdf.ln(6)

    # ── 1. Executive Summary (AI Insights) ──────────────────────────
    pdf.section_title("1. Executive Summary")
    if insights and "plain_summary" in insights:
        pdf.body_text(insights["plain_summary"])
        pdf.ln(2)
        if "analogy" in insights:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(26, 115, 232)
            pdf.multi_cell(pdf.epw, 6, _s(f'  "{insights["analogy"]}"'), new_x="LMARGIN", new_y="NEXT")
        if "human_impact" in insights:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(180, 30, 30)
            pdf.multi_cell(pdf.epw, 6, _s(f"  Human Impact: {insights['human_impact']}"), new_x="LMARGIN", new_y="NEXT")
    else:
        sev_desc = {
            "Low": "Bias levels are within acceptable thresholds. Continue monitoring.",
            "Medium": "Moderate bias detected. Review features and consider mitigation.",
            "High": "Significant bias detected. Mitigation required before deployment.",
            "Critical": "Severe bias detected. Do not deploy this system without remediation."
        }
        pdf.body_text(sev_desc.get(sev, "Bias audit complete."))
    pdf.ln(4)

    # ── 2. Dataset Health ────────────────────────────────────────────
    pdf.section_title("2. Dataset Health")
    pdf.kv_row("Total Rows", f"{inspection['total_rows']:,}")
    pdf.kv_row("Total Columns", str(inspection["total_cols"]))
    pdf.kv_row("Health Score", f"{inspection['health_score']}/100")
    if inspection.get("likely_sensitive"):
        pdf.kv_row("Sensitive Cols Detected",
                   ", ".join(inspection["likely_sensitive"]))
    pdf.ln(4)

    # Group outcome rates table
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(32, 33, 36)
    pdf.cell(0, 7, "  Positive Outcome Rate by Group:", new_x="LMARGIN", new_y="NEXT")
    adv = bias_result.get("advantaged_group", "")
    disadv = bias_result.get("disadvantaged_group", "")
    for group, rate in bias_result["group_rates"].items():
        tag = " (most favoured)" if group == adv else (
              " (least favoured)" if group == disadv else "")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        bar_width = max(1, min(70, int(rate * 70)))  # cap at 70mm so it never overflows
        pdf.cell(30, 6, _s(f"    {group}"), new_x="RIGHT", new_y="TOP")
        pdf.set_fill_color(26, 115, 232)
        pdf.cell(bar_width, 5, "", fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(60, 6, _s(f"  {rate*100:.1f}%{tag}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── 3. Fairness Metrics ──────────────────────────────────────────
    pdf.section_title("3. Fairness Metrics")
    for name, data in bias_result["metrics"].items():
        status = "PASS" if data["pass"] else "FAIL"
        r2, g2, b2 = (24, 128, 56) if data["pass"] else (217, 48, 37)

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(r2, g2, b2)
        pdf.cell(10, 7, "  ", new_x="RIGHT", new_y="TOP")
        pdf.cell(80, 7, _s(name.replace("_", " ").title()), new_x="RIGHT", new_y="TOP")
        pdf.cell(20, 7, _s(status), new_x="RIGHT", new_y="TOP")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7,
                 _s(f"Value: {data['value']}   (Acceptable: {data['range']})"),
                 new_x="LMARGIN", new_y="NEXT")

        if "what_is_it" in data:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(100, 100, 100)
            # Use pdf.epw and explicit margin reset to avoid potential x-pos bugs
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 5, _s(f"       What is it? {data['what_is_it']}"), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 5, _s(f"       {data['explain']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    pdf.ln(2)

    # ── 4. Root Causes & Action Items (AI Insights) ──────────────────
    if insights and "root_causes" in insights:
        pdf.section_title("4. AI-Identified Root Causes")
        for i, cause in enumerate(insights["root_causes"], 1):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(pdf.epw, 6, _s(f"  {i}. {cause}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        pdf.section_title("5. Action Items")
        for i, item in enumerate(insights.get("action_items", recommendations), 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(26, 115, 232)
            pdf.multi_cell(pdf.epw, 6, _s(f"  {i}. {item}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    else:
        pdf.section_title("4. Recommendations")
        for i, rec in enumerate(recommendations, 1):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(pdf.epw, 6, _s(f"  {i}. {rec}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── 5 / 6. Compliance Checklist ──────────────────────────────────
    section_n = "6" if insights else "5"
    pdf.section_title(f"{section_n}. Compliance Checklist")

    m = bias_result["metrics"]
    di_val = m["disparate_impact"]["value"]
    compliance_items = [
        ("EU AI Act 2024 (Art. 10)",
         "Bias assessment completed for high-risk AI system",
         True),
        ("EEOC 80% Rule",
         f"Disparate Impact = {di_val} ({'PASS' if m['disparate_impact']['pass'] else 'FAIL'})",
         m["disparate_impact"]["pass"]),
        ("US ECOA / Fair Credit",
         "Demographic group outcome analysis completed",
         True),
        ("ISO/IEC 42001",
         "Full audit trail saved with metric evidence",
         True),
        ("Demographic Parity",
         f"Gap = {m['demographic_parity']['value']} ({'PASS' if m['demographic_parity']['pass'] else 'FAIL'})",
         m["demographic_parity"]["pass"]),
    ]

    if insights and "compliance_risk" in insights:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(180, 30, 30)
        pdf.multi_cell(pdf.epw, 6, _s(f"  WARNING: {insights['compliance_risk']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    for law, note, passed in compliance_items:
        r3, g3, b3 = (24, 128, 56) if passed else (217, 48, 37)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(r3, g3, b3)
        pdf.cell(12, 6, "  [OK]" if passed else "  [!!]", new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(55, 6, _s(law), new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6, _s(note), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Save
    path = f"bias_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(path)
    return path