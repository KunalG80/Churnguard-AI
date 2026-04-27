"""
ChurnGuard AI — Streamlit Dashboard
All fixes applied — see CHANGELOG in Readme.md.
"""
import logging
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    MODEL_PATH, PROCESSED_CSV, FIGURES_DIR, REPORTS_DIR,
    TENURE_BINS, TENURE_LABELS, THRESHOLD,
)
from src.data.preprocess import clean_data
from src.utils.dtype_enforcer import enforce_training_dtypes
from src.utils.schema_handler import align_schema
from src.utils.segment_roi import segment_roi_analysis, budget_constrained_targets
from src.utils.executive_summary import executive_summary
from src.utils.roi_table import segment_roi_table
from src.utils.roi_analysis import roi_vs_threshold
from src.utils.ppt_export import export_ppt
from src.utils.pdf_export import export_pdf
from src.utils.report_bundle import build_report_bundle
from src.utils.live_budget_chart import live_budget_vs_recoverable
from src.utils.export_bundle import export_bundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide", page_title="ChurnGuard AI")
st.title("📉 ChurnGuard AI Dashboard")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model not found at `{MODEL_PATH}`. Run `python main.py` first.")
        st.stop()

pipeline = load_pipeline()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV to begin. Required columns: tenure, MonthlyCharges, Contract, gender …")
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not parse CSV: {e}"); st.stop()

st.subheader("📄 Raw data preview")
st.dataframe(raw.head())

if "Churn" in raw.columns:
    raw = raw.drop(columns=["Churn"])

# ── Preprocessing ─────────────────────────────────────────────────────────────
try:
    data = clean_data(raw)
    data = align_schema(data)

    if "tenure" in data.columns:
        data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
        data["tenure_group"] = pd.cut(
            data["tenure"], bins=TENURE_BINS,
            labels=TENURE_LABELS, include_lowest=True,
        ).astype(str)

    trained_features = pd.read_csv(PROCESSED_CSV).drop(columns=["Churn"], errors="ignore")
    data = enforce_training_dtypes(data, trained_features)

    num_cols = data.select_dtypes(include="number").columns
    cat_cols = data.select_dtypes(include=["object", "string"]).columns
    data[num_cols] = data[num_cols].fillna(0)
    data[cat_cols] = data[cat_cols].fillna("Unknown")

except Exception as e:
    st.error(f"Preprocessing error: {e}")
    logger.exception("Preprocessing failed"); st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
threshold = st.sidebar.slider("Churn probability threshold", 0.1, 0.9,
                               float(THRESHOLD), 0.05)

gender_opts   = ["All"] + sorted(data["gender"].dropna().unique().tolist())   if "gender"   in data.columns else ["All"]
contract_opts = ["All"] + sorted(data["Contract"].dropna().unique().tolist()) if "Contract" in data.columns else ["All"]
gender   = st.sidebar.selectbox("Gender",   gender_opts)
contract = st.sidebar.selectbox("Contract", contract_opts)

# ── Predict ───────────────────────────────────────────────────────────────────
probs = pipeline.predict_proba(data)[:, 1]
preds = (probs >= threshold).astype(int)
data["Churn_Prediction"]  = preds
data["Churn_Probability"] = probs

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = data.copy()
if gender   != "All" and "gender"   in filtered.columns:
    filtered = filtered[filtered["gender"]   == gender]
if contract != "All" and "Contract" in filtered.columns:
    filtered = filtered[filtered["Contract"] == contract]
filtered = filtered.reset_index(drop=True)

st.subheader("🔮 Predictions")
preview_cols = ["Churn_Prediction", "Churn_Probability"] + [
    c for c in filtered.columns if c not in ["Churn_Prediction", "Churn_Probability"]
]
st.dataframe(filtered[preview_cols].head(200), use_container_width=True)

# ── Campaign assumptions ───────────────────────────────────────────────────────
st.subheader("Retention campaign assumptions")
colA, colB, colC = st.columns(3)
cost         = colA.number_input("Cost per customer (₹)", 100, 5000, 500)
months       = colB.slider("Months of revenue at risk", 3, 24, 12)
success      = colC.slider("Campaign success rate", 0.1, 0.8, 0.3)
total_budget = st.number_input("Total retention budget (₹)", 50_000, 500_000, 100_000)

seg_df  = segment_roi_analysis(filtered, cost, months, success)
summary = executive_summary(seg_df)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue at Risk",           f"₹{summary['Revenue at Risk']:,.0f}")
k2.metric("Total Recoverable Revenue", f"₹{summary['Total Recoverable Revenue']:,.0f}")
k3.metric("Net Value Created",         f"₹{summary['Net Value Created']:,.0f}")
budget_used = seg_df["Retention_Investment"].sum() if not seg_df.empty else 0
k4.metric("Budget utilisation",
          f"{budget_used/total_budget:.0%}" if total_budget else "—")

# ── Charts row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    if "gender" in filtered.columns:
        gr = filtered.groupby("gender")["Churn_Prediction"].mean().reset_index()
        gr["Churn_Rate_%"] = gr["Churn_Prediction"] * 100
        fig = px.bar(gr, x="gender", y="Churn_Rate_%", text="Churn_Rate_%",
                     color="gender", title="Churn rate by gender")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    counts   = filtered["Churn_Prediction"].value_counts()
    donut_df = pd.DataFrame({"Status": ["Retained", "Churned"],
                              "Count": [counts.get(0, 0), counts.get(1, 0)]})
    fig = px.pie(donut_df, names="Status", values="Count",
                 hole=0.6, title="Churn split")
    st.plotly_chart(fig, use_container_width=True)

# ── Segment ROI ───────────────────────────────────────────────────────────────
st.subheader("Segment ROI analysis")
roi_tbl = segment_roi_table(seg_df)
st.dataframe(roi_tbl, use_container_width=True)

if not seg_df.empty:
    fig = px.bar(seg_df, x="Risk_Tier", y="Net_Value_Created",
                 color="Recommendation", text="Net_Value_Created",
                 color_discrete_map={"INVEST": "#22C55E", "AVOID": "#EF4444"},
                 title="Net value created by risk tier")
    fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ── ROI vs threshold (previously orphaned — now wired in) ─────────────────────
st.subheader("ROI vs decision threshold")
st.caption("Sweep thresholds to find the optimal targeting cutoff for your budget.")
roi_sweep = roi_vs_threshold(filtered, cost, success, months)
if not roi_sweep.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roi_sweep["Threshold"], y=roi_sweep["Net_ROI"],
                             name="Net ROI (₹)", line=dict(color="#185FA5", width=2)))
    fig.add_trace(go.Scatter(x=roi_sweep["Threshold"], y=roi_sweep["Retention_Spend"],
                             name="Retention Spend (₹)",
                             line=dict(color="#E24B4A", width=1.5, dash="dash")))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(xaxis_title="Decision Threshold",
                      yaxis_title="₹", height=320,
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

# ── Budget-constrained target list ────────────────────────────────────────────
st.subheader("Budget-constrained target list")
st.caption("Predicted churners ranked by net value — greedy budget allocation.")
target_df = budget_constrained_targets(filtered, cost, months, success, total_budget)
n_churners = int((filtered["Churn_Prediction"] == 1).sum())
if not target_df.empty:
    st.success(
        f"₹{total_budget:,.0f} budget reaches **{len(target_df)}** of "
        f"{n_churners} predicted churners."
    )
    show_cols = [c for c in ["Churn_Probability", "MonthlyCharges",
                              "Net_Value", "Cumulative_Spend",
                              "Contract", "tenure"] if c in target_df.columns]
    st.dataframe(target_df[show_cols].head(100), use_container_width=True)
else:
    st.info("No predicted churners in current filter selection.")

# ── Live budget simulation ────────────────────────────────────────────────────
st.subheader("Live budget vs recoverable revenue")
live_budget_vs_recoverable(filtered, cost, months, success)

# ── Report generation (behind button) ─────────────────────────────────────────
st.subheader("Export reports")
if st.button("📊 Generate PDF + PPTX report", use_container_width=True):
    with st.spinner("Generating reports …"):
        try:
            if not seg_df.empty:
                for fname, ycol, title in [
                    ("revenue_by_segment.png",   "Revenue_at_Risk",      "Revenue Exposure by Segment"),
                    ("value_created.png",         "Net_Value_Created",    "Net Value Created"),
                    ("retention_investment.png",  "Retention_Investment", "Retention Investment by Segment"),
                ]:
                    fig = px.bar(seg_df, x="Risk_Tier", y=ycol, title=title)
                    fig.write_image(str(FIGURES_DIR / fname), scale=2)

            bundle = build_report_bundle(seg_df, summary, cost, months, success, total_budget)
            export_pdf(bundle)
            export_ppt(bundle)
            st.success("Reports ready!")
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            logger.exception("Report error")

if (REPORTS_DIR / "churnguard_report.pdf").exists() and \
   (REPORTS_DIR / "churnguard_report.pptx").exists():
    zip_bytes = export_bundle()
    st.download_button(
        label="⬇ Download retention strategy pack (.zip)",
        data=zip_bytes,
        file_name="Retention_Strategy_Pack.zip",
        mime="application/zip",
        use_container_width=True,
    )
