from fpdf import FPDF
from datetime import datetime

def generate_pdf(inspection, bias_result, severity,
                 recommendations, sensitive_col, label_col) -> str:
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(79, 70, 229)
    pdf.cell(0, 12, "Unbiased AI Decision - Audit Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Sensitive: {sensitive_col}  |  Outcome: {label_col}", ln=True, align="C")
    pdf.ln(6)

    # Dataset health
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 27, 75)
    pdf.cell(0, 8, "1. Dataset Health", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"  Rows: {inspection['total_rows']}   Columns: {inspection['total_cols']}   Health Score: {inspection['health_score']}/100", ln=True)
    pdf.ln(4)

    # Metrics
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 27, 75)
    pdf.cell(0, 8, "2. Bias Metrics", ln=True)
    for name, data in bias_result["metrics"].items():
        status = "PASS" if data["pass"] else "FAIL"
        r, g, b = (34, 197, 94) if data["pass"] else (239, 68, 68)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(r, g, b)
        pdf.cell(70, 7, f"  {name.replace('_', ' ').title()}", ln=False)
        pdf.cell(20, 7, status, ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, f"Value: {data['value']}  (Acceptable: {data['range']})", ln=True)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, f"    -> {data['explain']}")
    pdf.ln(4)

    # Severity
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 27, 75)
    pdf.cell(0, 8, "3. Overall Severity", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(239, 68, 68)
    pdf.cell(0, 8, f"  {severity['severity']}  -  Bias Score: {severity['score']}/100", ln=True)
    pdf.ln(4)

    # Recommendations
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 27, 75)
    pdf.cell(0, 8, "4. Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    for i, rec in enumerate(recommendations, 1):
        pdf.multi_cell(0, 6, f"  {i}. {rec}")
    pdf.ln(4)

    # Compliance
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 27, 75)
    pdf.cell(0, 8, "5. Compliance Checklist", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    m = bias_result["metrics"]
    items = [
        ("EU AI Act 2024", "Bias assessment completed for high-risk AI system"),
        ("EEOC 80% Rule",  f"Disparate Impact = {m['disparate_impact']['value']} - {'PASS' if m['disparate_impact']['pass'] else 'FAIL'}"),
        ("ECOA",           "Demographic group outcome analysis completed"),
        ("ISO/IEC 42001",  "Audit saved locally with full metric trail"),
    ]
    for law, note in items:
        pdf.cell(40, 6, f"  [{law}]", ln=False)
        pdf.cell(0, 6, note, ln=True)

    path = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(path)
    return path