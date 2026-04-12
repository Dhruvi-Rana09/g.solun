import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

from core.inspector import inspect_dataset
from core.metrics import compute_metrics
from core.scorer import compute_severity, generate_recommendations
from core.mitigator import apply_reweighing
from core.reporter import generate_pdf

# ── Local history helpers ──────────────────────────────────────────────────
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def save_audit_local(data: dict):
    filename = f"{HISTORY_DIR}/audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def load_all_audits() -> list:
    audits = []
    for fname in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(HISTORY_DIR, fname)) as f:
                audits.append(json.load(f))
    return audits

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Unbiased AI Decision",
    page_icon="🛡️",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Unbiased AI")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home", "🔍 Run Audit", "📋 Audit History"])
    st.markdown("---")
    st.caption("Google Solution Challenge")
    st.caption("Detecting bias in AI decisions")

# ── HOME ───────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🛡️ Unbiased AI Decision")
    st.subheader("Detect and fix hidden bias in AI-powered decisions")
    st.markdown("""
    AI systems making decisions about **hiring, loans, and medical care**
    often learn from historically biased data and silently amplify
    discrimination at scale.

    **This tool lets you:**
    - Upload any decision dataset (CSV)
    - Detect bias across demographic groups in seconds
    - Get plain-English explanations of what's wrong
    - Apply an automatic fix and see before vs after
    - Download a compliance-ready PDF report
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Fairness Metrics", "4 Core Metrics")
    col2.metric("Fix Strategy", "Reweighing")
    col3.metric("Compliance", "EU AI Act · EEOC · ECOA")

# ── RUN AUDIT ──────────────────────────────────────────────────────────────
elif page == "🔍 Run Audit":
    st.title("🔍 Run a Bias Audit")

    st.header("Step 1 — Upload Dataset")
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

        with st.expander("Preview first 10 rows"):
            st.dataframe(df.head(10))

        inspection = inspect_dataset(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{inspection['total_rows']:,}")
        col2.metric("Total Columns", inspection["total_cols"])
        col3.metric("Data Health Score", f"{inspection['health_score']}/100")

        if inspection["likely_sensitive"]:
            st.info(f"🔍 Auto-detected sensitive columns: **{', '.join(inspection['likely_sensitive'])}**")

        st.header("Step 2 — Configure Audit")
        col1, col2, col3 = st.columns(3)
        sensitive_col  = col1.selectbox("Protected / Sensitive Column", df.columns)
        label_col      = col2.selectbox("Outcome / Label Column", df.columns)
        unique_labels  = df[label_col].unique().tolist()
        positive_label = col3.selectbox("Positive Label (the 'good' outcome)", unique_labels)

        st.header("Step 3 — Run Audit")
        if st.button("🚀 Run Bias Audit", type="primary", use_container_width=True):

            with st.spinner("Analysing dataset for bias..."):
                bias_result     = compute_metrics(df, label_col, sensitive_col, positive_label)
                severity        = compute_severity(bias_result)
                recommendations = generate_recommendations(bias_result, sensitive_col)

            # Severity banner
            sev = severity["severity"]
            banner = {"Low": st.success, "Medium": st.warning,
                      "High": st.warning, "Critical": st.error}
            banner[sev](f"{severity['color']} **Bias Severity: {sev}** — Score: {severity['score']}/100")

            # Group outcome chart
            st.subheader("📊 Positive Outcome Rate by Group")
            group_df = pd.DataFrame({
                "Group": list(bias_result["group_rates"].keys()),
                "Positive Rate": list(bias_result["group_rates"].values())
            }).set_index("Group")
            st.bar_chart(group_df)

            # Metrics table
            st.subheader("📋 Fairness Metrics")
            rows = []
            for name, data in bias_result["metrics"].items():
                rows.append({
                    "Metric":           name.replace("_", " ").title(),
                    "Value":            data["value"],
                    "Acceptable Range": data["range"],
                    "Status":           "✅ PASS" if data["pass"] else "❌ FAIL",
                    "Plain English":    data["explain"]
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Recommendations
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"**{i}.** {rec}")

            # Mitigation
            st.subheader("🔧 Apply Fix — Reweighing")
            st.caption("Reweighing gives underrepresented groups more influence during model training.")
            if st.button("Apply Reweighing & Compare"):
                with st.spinner("Applying fix and recomputing metrics..."):
                    mit = apply_reweighing(df, label_col, sensitive_col, positive_label)

                if "error" in mit:
                    st.error(mit["error"])
                else:
                    st.success("✅ Fix applied! Before vs After:")
                    new_m = mit["new_metrics"]["metrics"]
                    compare = []
                    for name, data in bias_result["metrics"].items():
                        after = new_m[name]
                        compare.append({
                            "Metric":        name.replace("_", " ").title(),
                            "Before Value":  data["value"],
                            "After Value":   after["value"],
                            "Before Status": "✅" if data["pass"] else "❌",
                            "After Status":  "✅" if after["pass"] else "❌"
                        })
                    st.dataframe(pd.DataFrame(compare), use_container_width=True, hide_index=True)

            # PDF report
            st.subheader("📄 Download Report")
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF..."):
                    pdf_path = generate_pdf(
                        inspection, bias_result, severity,
                        recommendations, sensitive_col, label_col
                    )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF", f,
                        file_name="bias_audit_report.pdf",
                        mime="application/pdf"
                    )

            # Save locally
            save_audit_local({
                "sensitive_col":  str(sensitive_col),
                "label_col":      str(label_col),
                "positive_label": str(positive_label),
                "severity":       severity["severity"],
                "score":          severity["score"],
                "rows":           inspection["total_rows"],
                "health_score":   inspection["health_score"],
                "metrics": {
                    k: {"value": v["value"], "pass": v["pass"]}
                    for k, v in bias_result["metrics"].items()
                },
                "recommendations": recommendations,
                "created_at": datetime.now().isoformat()
            })
            st.toast("✅ Audit saved to history!", icon="💾")

# ── HISTORY ────────────────────────────────────────────────────────────────
elif page == "📋 Audit History":
    st.title("📋 Audit History")
    audits = load_all_audits()

    if not audits:
        st.info("No audits yet. Run your first audit from the Run Audit page!")
    else:
        st.caption(f"Showing {len(audits)} past audit(s) — saved locally in /history folder")
        for a in audits:
            icon = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}.get(a.get("severity"), "⚪")
            with st.expander(f"{icon} {a.get('severity')} — Score: {a.get('score')}/100 | Sensitive: `{a.get('sensitive_col')}` | {a.get('created_at', '')[:10]}"):
                col1, col2 = st.columns(2)
                col1.metric("Bias Score", f"{a.get('score')}/100")
                col2.metric("Data Health", f"{a.get('health_score')}/100")
                st.write("**Metrics:**")
                for mname, mdata in a.get("metrics", {}).items():
                    status = "✅" if mdata["pass"] else "❌"
                    st.write(f"{status} {mname.replace('_', ' ').title()}: `{mdata['value']}`")
                st.write("**Recommendations:**")
                for rec in a.get("recommendations", []):
                    st.write(f"• {rec}")