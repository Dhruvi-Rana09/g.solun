import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from core.inspector import inspect_dataset
from core.metrics import compute_metrics
from core.scorer import compute_severity, generate_recommendations
from core.mitigator import apply_reweighing
from core.reporter import generate_pdf
from core.gemini_insights import get_gemini_insights
from core.model_auditor import audit_model, MODEL_OPTIONS

# ── 1. Page Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Unbiased AI | Governance Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. Premium SaaS UI CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Font & Global Reset */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* 1. Hide Streamlit Branding & Chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; display: none !important; }
.stDeployButton { display: none !important; }

/* Force Sidebar Toggle Buttons to be absolutely visible */
[data-testid="collapsedControl"], 
[data-testid="collapsedControl"] * {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}
/* Always use dark toggle icons for the light theme we are building */
[data-testid="collapsedControl"] svg,
section[data-testid="stSidebar"] button[kind="header"] svg {
    color: #0F172A !important;
    fill: #0F172A !important;
}

/* Refined Status Widget (Make it discrete but visible) */
div[data-testid="stStatusWidget"] {
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(6px) !important;
    border-radius: 6px !important;
    border: 1px solid #E2E8F0 !important;
    padding: 4px 10px !important;
    bottom: 20px !important;
    right: 20px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}

/* 2. App Background & Layout */
.stApp {
    background-color: #F1F5F9; /* More standard SaaS gray-blue */
}
.main .block-container {
    padding-top: 2.5rem !important;
    padding-left: 3.5rem !important;
    padding-right: 3.5rem !important;
    max-width: 1300px;
}

/* 3. Sidebar Makeover - Light Mode Edition */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #64748B;
}

/* Logo Area */
.sidebar-logo-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 3.5rem;
    padding: 0 10px;
}
.logo-badge {
    background: #6366F1; /* Indigo SaaS accent */
    color: white;
    padding: 7px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    box-shadow: 0 4px 12px rgba(99,102,241,0.25);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.logo-text {
    color: #0F172A !important;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: -0.02em;
}

/* Navigation List spacing */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    padding-top: 1rem;
}

/* HIDDEN RADIO BUTTONS for Navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important; /* Hide the radio circle */
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 12px 16px !important;
    border-radius: 10px !important;
    background-color: transparent !important;
    margin-bottom: 6px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #F8FAFC !important;
    border-color: #F1F5F9 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
    background-color: rgba(99,102,241,0.06) !important;
    border-color: rgba(99,102,241,0.12) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] p {
    color: #6366F1 !important;
    font-weight: 600 !important;
}

/* Sidebar Footer */
.sidebar-footer {
    position: fixed;
    bottom: 30px;
    left: 20px;
    color: #94A3B8;
    font-size: 0.7rem;
    font-weight: 500;
    line-height: 1.5;
}

/* 4. Cards (.saas-card and .metric-card-container) */
.saas-card, .metric-card-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important;
    padding: 2rem;
    box-shadow: 
        0 4px 6px -1px rgba(0, 0, 0, 0.02), 
        0 2px 4px -1px rgba(0, 0, 0, 0.01);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 2rem;
}
.saas-card:hover, .metric-card-container:hover {
    transform: translateY(-4px);
    box-shadow: 
        0 20px 25px -5px rgba(0, 0, 0, 0.04), 
        0 10px 10px -5px rgba(0, 0, 0, 0.02);
}

/* 5. KPI Metric Cards */
.metric-card-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 140px;
}
.metric-title {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 12px;
}
.metric-value {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #0F172A;
    letter-spacing: -0.03em !important;
    line-height: 1;
}
.metric-subtitle {
    font-size: 0.8rem !important;
    color: #94A3B8;
    font-weight: 500;
    margin-top: 10px;
}

/* 6. Subtle Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-success { background: #ECFDF5; color: #059669; border: 1px solid #D1FAE5; }
.badge-warning { background: #FFFBEB; color: #D97706; border: 1px solid #FEF3C7; }
.badge-error   { background: #FEF2F2; color: #DC2626; border: 1px solid #FEE2E2; }
.badge-neutral { background: #F8FAFC; color: #475569; border: 1px solid #E2E8F0; }

/* 7. Tables */
.audit-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 1.5rem 0;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
}
.audit-table th {
    background: #F8FAFC;
    padding: 14px 18px;
    text-align: left;
    font-size: 0.725rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #475569;
    border-bottom: 2px solid #E2E8F0;
    letter-spacing: 0.05em;
}
.audit-table td {
    padding: 16px 18px;
    color: #334155;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.9rem;
    line-height: 1.5;
}
.audit-table tr:last-child td {
    border-bottom: none;
}
.audit-table tr:hover {
    background-color: #FBFCFE;
}

/* 8. Buttons */
.stButton button {
    border-radius: 10px !important;
    padding: 0.6rem 1.75rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton button[kind="primary"] {
    background: #4F46E5 !important; /* Indigo center */
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
}
.stButton button[kind="primary"]:hover {
    background: #4338CA !important;
    box-shadow: 0 8px 16px rgba(79, 70, 229, 0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton button[kind="secondary"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #334155 !important;
}
.stButton button[kind="secondary"]:hover {
    border-color: #6366F1 !important;
    color: #6366F1 !important;
    background: #F8FAFC !important;
}

/* 9. File Uploader */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #E2E8F0 !important;
    border-radius: 16px !important;
    background: #FFFFFF !important;
    transition: all 0.3s ease;
    padding: 3rem 2rem !important;
    text-align: center;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #6366F1 !important;
    background: #F5F7FF !important;
    transform: scale(1.01);
}

/* 10. Subtle Severity Strip */
.severity-strip {
    height: 52px;
    border-radius: 12px;
    padding: 0 1.5rem;
    display: flex;
    align-items: center;
    font-weight: 700;
    font-size: 1rem;
    margin: 2rem 0;
    border: 1px solid transparent;
    border-left-width: 6px !important;
}
.sev-low      { background: #ECFDF5; border-color: #D1FAE5; border-left-color: #10B981 !important; color: #065F46; }
.sev-medium   { background: #FFFBEB; border-color: #FEF3C7; border-left-color: #F59E0B !important; color: #92400E; }
.sev-high     { background: #FFF7ED; border-color: #FFEDD5; border-left-color: #F97316 !important; color: #9A3412; }
.sev-critical { background: #FEF2F2; border-color: #FEE2E2; border-left-color: #EF4444 !important; color: #991B1B; }

/* 11. Insight Cards */
.insight-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #6366F1;
    border-radius: 12px;
    padding: 1.75rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.insight-header {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #64748B;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
}

/* 12. Dashboard Feature Blocks */
.feature-block {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
}
.feature-block:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08);
    border-color: #6366F1;
}

/* Typography Overrides */
.stMarkdown h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin-top: 2.5rem !important;
    margin-bottom: 1rem !important;
    letter-spacing: -0.01em !important;
}
.stMarkdown p {
    line-height: 1.6;
    color: #475569;
}
</style>
""", unsafe_allow_html=True)

# ── 3. UI Helper Components ────────────────────────────────────────────────

def ui_page_header(title, subtitle=""):
    """Consistent SaaS page header."""
    sub = f'<div style="font-size:0.925rem;color:#64748B;margin-bottom:2rem;">{subtitle}</div>' if subtitle else ''
    st.markdown(f'<div style="font-size:1.85rem;font-weight:800;color:#0F172A;letter-spacing:-0.04em;">{title}</div>{sub}', unsafe_allow_html=True)

def ui_card(title, value, icon="", subtitle="", status="neutral"):
    """KPI Metric card."""
    st.markdown(f"""
    <div class="metric-card-container">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def ui_alert(message, alert_type="info"):
    """Custom alert box."""
    st.markdown(f"""
    <div class="saas-alert alert-{alert_type}">
        <div style="font-size: 1.2em;">{'ℹ️' if alert_type=='info' else '🚨'}</div>
        <div style="font-weight: 500;">{message}</div>
    </div>
    """, unsafe_allow_html=True)

def ui_badge(text, badge_type="neutral"):
    """Inline status badge."""
    return f'<span class="badge badge-{badge_type}">{text}</span>'

def style_fig(fig):
    """Global styling applier for Plotly charts."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color="#64748B"),
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(gridcolor='#F1F5F9', zeroline=False),
        yaxis=dict(gridcolor='#F1F5F9', zeroline=False),
        legend=dict(font=dict(size=10))
    )
    fig.update_traces(marker_line_width=0)
    return fig

# ── 4. State & Helpers ─────────────────────────────────────────────────────

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def save_audit_local(data: dict):
    fname = f"{HISTORY_DIR}/audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)

def load_all_audits() -> list:
    audits = []
    for fname in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(HISTORY_DIR, fname)) as f:
                try: audits.append(json.load(f))
                except Exception: pass
    return audits

for key in ["audit_done", "bias_result", "severity", "recommendations",
            "inspection", "insights", "df", "mit_result",
            "model_audit_result", "pdf_path"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "audit_done" not in st.session_state:
    st.session_state.audit_done = False

# ── 5. Sidebar Navigation ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-container">
        <div class="logo-badge">AI</div>
        <div class="logo-text">Unbiased</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "MODULES",
        ["Dashboard Overview", "Historical Data Audit", "Predictive Model Audit", "Compliance Logs"],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="sidebar-footer">Unbiased AI Governance v1.2<br>© 2026 Enterprise Trust</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Dashboard Overview":
    ui_page_header("Platform Overview", "Enterprise AI fairness monitoring and regulatory governance command center.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: ui_card("Supported Metrics", "4", "📐", "EEOC & Parity Standard")
    with col2: ui_card("Mitigation Ops", "Active", "🔧", "Advanced Reweighing")
    with col3: ui_card("Global Frameworks", "3+", "🌍", "EU AI Act, ISO 42001")
    with col4: ui_card("Core Intelligence", "Gemini", "⚡", "Live Root Cause Analysis")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Operational Workflows")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        <div class="feature-block">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🔍</div>
            <h3 style="margin-top:0; color:#0F172A; font-weight:700;">Historical Data Audit</h3>
            <p style="color:#64748B; font-size:0.875rem; line-height:1.6;">Ingest legacy decision datasets to detect historical patterns of bias. Map outcomes across gender, race, and other protected vectors.</p>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class="feature-block">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🤖</div>
            <h3 style="margin-top:0; color:#0F172A; font-weight:700;">Predictive Model Audit</h3>
            <p style="color:#64748B; font-size:0.875rem; line-height:1.6;">Evaluate trained classification models before deployment. Detect proxy behavioral shifts and ensure fair inference across subsets.</p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORICAL DATA AUDIT
# ════════════════════════════════════════════════════════════════════════════
elif page == "Historical Data Audit":
    ui_page_header("Dataset Bias Scanner", "Analyze tabular datasets to identify systemic outcome disparities.")
    
    st.markdown("## 1. Data Ingestion")
    with st.container():
        uploaded = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        inspection = inspect_dataset(df)
        st.session_state.inspection = inspection
        
        ui_alert(f"File ingestion successful: {len(df):,} records identified in memory.", "info")
        
        st.markdown("## 2. Configuration Parameters")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            sensitive_col = st.selectbox(
                "Protected Vector", 
                df.columns, 
                index=list(df.columns).index(inspection["likely_sensitive"][0]) if inspection["likely_sensitive"] else 0
            )
        with c2:
            label_col = st.selectbox("Target Outcome", df.columns, index=len(df.columns)-1)
        with c3:
            unique_labels = df[label_col].dropna().unique().tolist()
            positive_label = st.selectbox("Favorable Outcome Mapping", unique_labels)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Initialize Fairness Audit", type="primary"):
            with st.spinner("Processing fairness vectors..."):
                bias_result     = compute_metrics(df, label_col, sensitive_col, positive_label)
                severity        = compute_severity(bias_result)
                recommendations = generate_recommendations(bias_result, sensitive_col)
                insights        = get_gemini_insights(inspection, bias_result, severity, sensitive_col, label_col)

            st.session_state.bias_result     = bias_result
            st.session_state.severity        = severity
            st.session_state.recommendations = recommendations
            st.session_state.insights        = insights
            st.session_state.audit_done      = True
            st.session_state.mit_result      = None
            st.session_state.pdf_path        = None

            save_audit_local({
                "sensitive_col":  str(sensitive_col), "label_col": str(label_col),
                "positive_label": str(positive_label), "severity": severity["severity"],
                "score": severity["score"], "rows": inspection["total_rows"],
                "health_score": inspection["health_score"],
                "metrics": {k: {"value": v["value"], "pass": v["pass"]} for k, v in bias_result["metrics"].items()},
                "recommendations": recommendations, "created_at": datetime.now().isoformat()
            })

    # Output State
    if st.session_state.audit_done and st.session_state.bias_result:
        bias_result = st.session_state.bias_result
        severity    = st.session_state.severity
        insights    = st.session_state.insights
        
        st.markdown("<hr style='margin: 3rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown("## Diagnostic Intelligence")
        
        # Severity KPI
        sev_status_map = {"Low": "success", "Medium": "warning", "High": "error", "Critical": "error"}
        sev_class_map = {"Low": "sev-low", "Medium": "sev-medium", "High": "sev-high", "Critical": "sev-critical"}
        
        s_class = sev_class_map.get(severity["severity"], "sev-medium")
        st.markdown(f'<div class="severity-strip {s_class}">BIAS SEVERITY: {severity["severity"].upper()} (SCORE: {severity["score"]}/100)</div>', unsafe_allow_html=True)
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1: ui_card("System Integrity", f"{100-severity['score']}%", "🛡️", "Calculated Fairness Index")
        with kpi2: ui_card("Advantaged Subset", bias_result.get("advantaged_group", "N/A"), "📈", "Highest Outcome Rate")
        with kpi3: ui_card("Disadvantaged Subset", bias_result.get("disadvantaged_group", "N/A"), "📉", "Lowest Outcome Rate")

        # Insights & Breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns([1.5, 1])
        
        with c_left:
            st.markdown("## Executive AI Summary")
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown('<div class="insight-header">Automated Assessment</div>', unsafe_allow_html=True)
            if insights and "plain_summary" in insights:
                st.markdown(f"{insights['plain_summary']}")
                if "human_impact" in insights:
                    st.markdown(f"<br><strong style='color:#B91C1C'>Social Impact:</strong> {insights['human_impact']}", unsafe_allow_html=True)
                if "compliance_risk" in insights:
                    st.markdown(f"<br>{ui_badge('Compliance Warning', 'error' if severity['score'] > 50 else 'warning')} {insights['compliance_risk']}", unsafe_allow_html=True)
            else:
                st.write("Metric calculation complete. AI Narrative generator is offline.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown("## Approval Variance")
            
            # Plotly Chart
            df_plot = pd.DataFrame({
                "Group": list(bias_result["group_rates"].keys()),
                "Rate": [v for v in bias_result["group_rates"].values()]
            })
            fig = px.bar(
                df_plot, x="Group", y="Rate", text_auto='.1%', color="Group",
                color_discrete_sequence=["#2563EB", "#94A3B8", "#64748B"]
            )
            fig.update_layout(yaxis=dict(tickformat=".0%"))
            st.plotly_chart(style_fig(fig), use_container_width=True, config={'displayModeBar': False})

        # Detailed Metrics Table
        st.markdown("## Core Fairness Telemetry")
        
        table_html = '<table class="audit-table"><thead><tr><th>Audit Vector</th><th>Value</th><th>Threshold</th><th>Status</th></tr></thead><tbody>'
        for name, data in bias_result["metrics"].items():
            b_class = "success" if data["pass"] else "error"
            b_text = "PASSED" if data["pass"] else "VIOLATION"
            explain_html = f'<div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">{data.get("explain", "")}</div>'
            table_html += f"""
<tr>
    <td>
        <div style="font-weight: 600; color:#0F172A;">{name.replace('_', ' ').title()}</div>
        {explain_html}
    </td>
    <td>{data['value']}</td>
    <td>{data['range']}</td>
    <td>{ui_badge(b_text, b_class)}</td>
</tr>
"""
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # ── Mitigation Engine ──
        st.markdown("<hr style='margin: 3rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown("## 🛠️ Remediation Protocol")
        st.markdown('<div style="font-size:0.875rem; color:#64748B; margin-bottom:1rem;">Execute algorithmic weights adjustment to counteract identified disparate impact.</div>', unsafe_allow_html=True)
        
        if st.button("Execute Training Weights Normalization", type="primary"):
            with st.spinner("Processing remediations..."):
                mit_result = apply_reweighing(st.session_state.df, label_col, sensitive_col, positive_label)
            st.session_state.mit_result = mit_result

        if st.session_state.mit_result:
            mit = st.session_state.mit_result
            if "error" in mit:
                ui_alert(mit["error"], "error")
            else:
                ui_alert("Reweighing sequence complete. Feature subset successfully normalized.", "info")
                
                col_b, col_a = st.columns(2)
                with col_b:
                    st.markdown("**Pre-normalization Variance:**")
                    fig_b = px.bar(
                        pd.DataFrame({"Group": list(bias_result["group_rates"].keys()), "Rate": list(bias_result["group_rates"].values())}),
                        x="Group", y="Rate"
                    )
                    fig_b.update_traces(marker_color="#94A3B8")
                    st.plotly_chart(style_fig(fig_b), use_container_width=True)
                with col_a:
                    st.markdown("**Normalized Target Variance:**")
                    new_rates = mit["new_metrics"]["group_rates"]
                    fig_a = px.bar(
                        pd.DataFrame({"Group": list(new_rates.keys()), "Rate": list(new_rates.values())}),
                        x="Group", y="Rate"
                    )
                    fig_a.update_traces(marker_color="#059669")
                    st.plotly_chart(style_fig(fig_a), use_container_width=True)

        # PDF Report
        st.markdown("<br>", unsafe_allow_html=True)
        col_pdf, _ = st.columns([2, 1])
        with col_pdf:
            st.markdown("## Regulatory Documentation")
            st.markdown('<p style="color:#64748B; font-size:0.875rem;">Compile a formal audit dossier mapping these findings to EU AI Act and ISO 42001 governance standards.</p>', unsafe_allow_html=True)
            
            if st.button("Generate Formal Governance Report", type="secondary"):
                with st.spinner("Compiling dossier..."):
                    pdf_path = generate_pdf(
                        st.session_state.inspection, st.session_state.bias_result,
                        st.session_state.severity, st.session_state.recommendations,
                        sensitive_col, label_col, insights=st.session_state.insights
                    )
                st.session_state.pdf_path = pdf_path

            if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
                with open(st.session_state.pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Governance Report",
                        data=f, file_name=f"Compliance_Dossier_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICTIVE MODEL AUDIT
# ════════════════════════════════════════════════════════════════════════════
elif page == "Predictive Model Audit":
    ui_page_header("Predictive System Evaluation", "Audit the behavioral fairness of trained Machine Learning models.")

    uploaded_m = st.file_uploader("Ingest Training Architecture Data (CSV)", type=["csv"], key="model_upload")
    
    if uploaded_m:
        df_m = pd.read_csv(uploaded_m)
        cA, cB, cC, cD = st.columns(4)
        sens_m  = cA.selectbox("Protected Input Vector", df_m.columns, key="m_sens")
        lab_m   = cB.selectbox("Prediction Target", df_m.columns, key="m_label")
        pos_m   = cC.selectbox("Favorable Outcome Mapping", df_m[lab_m].dropna().unique(), key="m_pos")
        model_m = cD.selectbox("Classifier Architecture", list(MODEL_OPTIONS.keys()), key="m_model")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Execute Inference Audit", type="primary"):
            with st.spinner(f"Compiling {model_m} graph..."):
                res = audit_model(df_m, lab_m, sens_m, pos_m, model_m)
            st.session_state.model_audit_result = res

    if st.session_state.model_audit_result:
        res = st.session_state.model_audit_result
        if "error" in res:
            ui_alert(res["error"], "error")
        else:
            fm = res.get("fairness_metrics", {})
            st.markdown("<hr style='margin: 2.5rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
            
            kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
            with kpi_c1: ui_card("General Accuracy", f"{res['accuracy']*100:.1f}%", "🎯", f"Samples: {res['test_size']:,}")
            
            if "error" not in fm:
                sev2 = compute_severity(fm)
                with kpi_c2: ui_card("Model Bias Score", f"{sev2['score']}/100", "⚖️", f"Classification: {sev2['severity']}")
            
            if "error" not in fm:
                with kpi_c3:
                    st.markdown('<div class="metric-title">Predicted Subset Differential</div>', unsafe_allow_html=True)
                    df_pred = pd.DataFrame({"Group": list(fm["group_rates"].keys()), "Rate": list(fm["group_rates"].values())})
                    fig_p = px.bar(df_pred, x="Rate", y="Group", orientation='h', color="Group", color_discrete_sequence=["#2563EB", "#94A3B8"])
                    fig_p.update_layout(height=100)
                    st.plotly_chart(style_fig(fig_p), use_container_width=True, config={'displayModeBar': False})

            st.markdown("<br>", unsafe_allow_html=True)
            row_l, row_r = st.columns([1.5, 1])
            
            with row_l:
                st.markdown("## Decision Vector Importance")
                if res["feature_importances"]:
                    df_feat = pd.DataFrame(list(res["feature_importances"].items())[:8], columns=["Feature", "Score"])
                    fig_f = px.bar(df_feat, x="Score", y="Feature", orientation='h')
                    fig_f.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
                    fig_f.update_traces(marker_color="#2563EB")
                    st.plotly_chart(style_fig(fig_f), use_container_width=True)

            with row_r:
                st.markdown("## Behavioral Proxy Risks")
                if res["proxy_warnings"]:
                    st.markdown('<p style="color:#64748B; font-size:0.85rem;">Detected correlations between non-protected inputs and sensitive vectors.</p>', unsafe_allow_html=True)
                    for w in res["proxy_warnings"]:
                        w_class = "error" if "High" in w["risk"] else "warning"
                        st.markdown(f'''
                        <div style="margin-bottom:12px; padding:12px; border-radius:8px; border:1px solid #E2E8F0; background:white;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                <strong style="color:#0F172A">{w["feature"]}</strong>
                                {ui_badge(w["risk"], w_class)}
                            </div>
                            <div style="color:#64748B; font-size:0.75rem;">Correlation: {w["correlation"]}</div>
                        </div>''', unsafe_allow_html=True)
                else:
                    ui_alert("No critical proxy vectors detected in schema.", "info")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: COMPLIANCE LOGS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Compliance Logs":
    ui_page_header("Governance Audit Trace", "Immutable logs and historical trending of system fairness validation.")
    
    audits = load_all_audits()
    if not audits:
        st.markdown('<div style="text-align:center; padding:5rem;"><div style="font-size:3rem; color:#D1D5DB; margin-bottom:1rem;">🗄️</div><h3 style="color:#64748B;">No validation logs available in the local database.</h3></div>', unsafe_allow_html=True)
    else:
        if len(audits) > 1:
            st.markdown("## Bias Level Aggregation (Historical Trend)")
            df_trend = pd.DataFrame([{"Date": a.get("created_at", "")[:10], "Score": a.get("score", 0)} for a in audits]).set_index("Date")
            fig_t = px.line(df_trend, y="Score", markers=True)
            fig_t.update_layout(height=240, yaxis=dict(range=[0, 100]))
            fig_t.update_traces(line_color="#2563EB", marker_size=8)
            st.plotly_chart(style_fig(fig_t), use_container_width=True)
        
        log_rows = ""
        for a in audits[:15]:
            s_map = {"Low": "success", "Medium": "warning", "High": "error", "Critical": "error"}
            s_class = s_map.get(a.get("severity", "Medium"), "neutral")
            log_rows += f"""
<tr>
    <td>{a.get('created_at', '')[:16].replace('T', ' ')}</td>
    <td style="font-weight:600; color:#0F172A;">{a.get('sensitive_col')}</td>
    <td>{a.get('label_col')}</td>
    <td>{ui_badge(a.get('severity', 'Unknown'), s_class)}</td>
    <td style="font-weight: 700;">{a.get('score', 0)}</td>
</tr>
"""
        
        st.markdown(f"""
<table class="audit-table">
    <thead>
        <tr>
            <th>Timestamp</th>
            <th>Vector</th>
            <th>Target</th>
            <th>Severity</th>
            <th>Score</th>
        </tr>
    </thead>
    <tbody>
        {log_rows}
    </tbody>
</table>
""", unsafe_allow_html=True)