import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

from src.data.preprocess import clean_data
from src.utils.dtype_enforcer import enforce_training_dtypes
from src.utils.schema_handler import align_schema
from src.utils.segment_roi import segment_roi_analysis
from src.utils.executive_summary import executive_summary
from src.utils.roi_table import segment_roi_table
from src.utils.ppt_export import export_ppt
from src.utils.pdf_export import export_pdf
from src.utils.report_bundle import build_report_bundle
from src.utils.live_budget_chart import live_budget_vs_recoverable
from src.utils.export_bundle import export_bundle

os.makedirs("reports/figures", exist_ok=True)

st.set_page_config(layout="wide", page_title="ChurnGuard AI")
st.title("📉 ChurnGuard AI Dashboard")

# -----------------------------
# Load trained pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("models/churn_model.pkl")

pipeline = load_pipeline()

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    raw = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Raw Data")
    st.dataframe(raw.head())

    # -----------------------------
    # DROP TARGET IF PRESENT
    # -----------------------------
    if "Churn" in raw.columns:
        raw = raw.drop(columns=["Churn"])

# -----------------------------
# ONLY CLEAN SAME AS TRAINING
# -----------------------------

# -------------------------------------
# STEP 1 : BASIC CLEANING
# -------------------------------------
    data = clean_data(raw)

# -------------------------------------
# STEP 2 : ALIGN TO TRAINING SCHEMA
# -------------------------------------
    data = align_schema(data)

# -------------------------------------
# STEP 3 : RE-CREATE TRAINED FEATURES
# (ALIGNMENT REMOVED THEM)
# -------------------------------------
    if "tenure" in data.columns:

        data["tenure"] = pd.to_numeric(
            data["tenure"],
            errors="coerce"
        )

        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[0,12,24,48,60,100],
            labels=[
                "0-12",
                "12-24",
                "24-48",
                "48-60",
                "60+"
            ]
        ).astype(str)

# -------------------------------------
# STEP 4 : FINAL TYPE ENFORCEMENT
# -------------------------------------
    trained = pd.read_csv(
        "data/processed/processed.csv"
    )

    trained_features = trained.drop(
        columns=["Churn"],
        errors="ignore"
    )

    data = enforce_training_dtypes(
        data,
        trained_features
    )

    data = data.fillna(0)

    threshold = st.slider(
        "Churn Probability Threshold",
        0.1, 0.9, 0.35, 0.05
    )

    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------
    probs = pipeline.predict_proba(data)[:,1]
    preds = (probs >= threshold).astype(int)

    data["Churn_Prediction"] = preds
    data["Churn_Probability"] = probs

    st.subheader("🔮 Predictions")
    st.dataframe(data)

# -------------------------
# SIDEBAR FILTERS
# -------------------------
    st.sidebar.title("Filters")

    gender = st.sidebar.selectbox(
        "Gender",
        ["All"] + list(data["gender"].unique())
    )

    contract = st.sidebar.selectbox(
        "Contract",
        ["All"] + list(data["Contract"].unique())
    )

    filtered_data = data.copy()

    if gender != "All":
        filtered_data = filtered_data[
            filtered_data["gender"] == gender
        ]

    if contract != "All":
        filtered_data = filtered_data[
            filtered_data["Contract"] == contract
        ]

    filtered_data = filtered_data.reset_index(drop=True)
    
# -------------------------
# CFO INPUTS
# -------------------------
    st.subheader("Retention Campaign Assumptions")

    colA,colB,colC = st.columns(3)

    cost = colA.number_input(
        "Retention Cost (₹)",
        100,5000,500
    )

    months = colB.slider(
        "Months Lost",
        3,24,12
    )

    success = colC.slider(
        "Success Rate",
        0.1,0.8,0.3
    )

    total_budget = st.number_input(
        "Retention Budget (₹)",
        50000,500000,100000
    )
    
    seg_df = segment_roi_analysis(
        filtered_data,
        cost,
        months,
        success
    )

# -------------------------
# KPI ROW
# -------------------------
    k1,k2,k3,k4 = st.columns(4)

    k1.metric(
        "Revenue at Risk",
        f"₹{seg_df['Revenue_at_Risk'].sum():,.0f}"
    )

    k2.metric(
        "Recoverable Revenue",
        f"₹{seg_df['Recoverable_Revenue'].sum():,.0f}"
    )

    k3.metric(
        "Net Value Created",
        f"₹{seg_df['Net_Value_Created'].sum():,.0f}"
    )

    k4.metric(
        "Budget Utilisation",
        f"{seg_df['Retention_Investment'].sum()/total_budget:.0%}"
    )

# -------------------------
# GENDER CHURN RATE
# -------------------------
    gender_rate = (
        filtered_data
        .groupby("gender")["Churn_Prediction"]
        .mean()
        .reset_index()
    )

    gender_rate["Churn_Rate_%"] = (
        gender_rate["Churn_Prediction"] * 100
    )

    fig = px.bar(
        gender_rate,
        x="gender",
        y="Churn_Rate_%",
        text="Churn_Rate_%",
        color="gender"
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )

    st.plotly_chart(fig, width="stretch")

# -------------------------
# DONUT CHART
# -------------------------
    counts = filtered_data[
        "Churn_Prediction"
    ].value_counts()

    donut_df = pd.DataFrame({
        "Status": ["Retained", "Churned"],
        "Count": [
            counts.get(0,0),
            counts.get(1,0)
        ]
    })

    fig = px.pie(
        donut_df,
        names="Status",
        values="Count",
        hole=0.6
    )

    st.plotly_chart(fig, width="stretch")

# -------------------------
# SEGMENT ROI
# -------------------------

    summary = executive_summary(seg_df)
    roi_tbl = segment_roi_table(seg_df)

    st.dataframe(roi_tbl, width="stretch")
    
    fig = px.bar(
        seg_df,
        x="Risk_Tier",
        y="Net_Value_Created",
        color="Recommendation",
        text="Net_Value_Created",
        color_discrete_map={
            "INVEST":"#22C55E",
            "AVOID":"#EF4444"
        }
    )

    fig.update_traces(
        texttemplate='₹%{text:,.0f}',
        textposition='outside'
    )

    st.plotly_chart(fig, width="stretch")

# -------------------------
# LIVE BUDGET SIMULATION
# -------------------------
    st.subheader("Live Budget vs Recoverable Revenue")

    live_budget_vs_recoverable(
        filtered_data,
        cost,
        months,
        success
    )

# -------------------------
# BUILD REPORT
# -------------------------
    bundle = build_report_bundle(
        seg_df,
        summary,
        cost,
        months,
        success,
        total_budget
    )
    
    fig_rev = px.bar(
        seg_df,
        x="Risk_Tier",
        y="Revenue_at_Risk",
        title="Revenue Exposure by Segment"
    )

    fig_rev.write_image(
        "reports/figures/revenue_by_segment.png",
        scale=2
    )
    
    fig_roi = px.bar(
        seg_df,
        x="Risk_Tier",
        y="Net_Value_Created",
        title="Net Value Created"
    )

    fig_roi.write_image(
        "reports/figures/value_created.png",
        scale=2
    )
    
    fig_target = px.bar(
        seg_df,
        x="Risk_Tier",
        y="Retention_Investment",
        title="Retention Investment by Segment"
    )

    fig_target.write_image(
        "reports/figures/retention_investment.png",
        scale=2
    )

    export_pdf(bundle)
    export_ppt(bundle)

# -------------------------
# CHECK REPORT GENERATION
# -------------------------

pdf_ready = os.path.exists(
    "reports/churnguard_report.pdf"
)

ppt_ready = os.path.exists(
    "reports/churnguard_report.pptx"
)

if pdf_ready and ppt_ready:
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        if st.button(
            "📦 Download Retention Strategy Pack",
            use_container_width=True
        ):

            zip_file = export_bundle()

            with open(zip_file,"rb") as f:

                st.download_button(
                    label="⬇ Executive Bundle Ready",
                    data=f,
                    file_name="Retention_Strategy_Pack.zip",
                    mime="application/zip",
                    use_container_width=True
                )