"""
BUG FIX: This file was imported in app.py but did not exist → ModuleNotFoundError crash.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go


def live_budget_vs_recoverable(df, retention_cost: float, months_lost: int, success_rate: float):
    """
    Interactive slider showing recoverable revenue vs budget at different spend levels.
    Renders directly into the Streamlit page.
    """
    churners = df[df["Churn_Prediction"] == 1].copy()
    if churners.empty:
        st.info("No predicted churners in current filter selection.")
        return

    churners = churners.sort_values("Churn_Probability", ascending=False).reset_index(drop=True)
    n_total  = len(churners)

    # Build cumulative curves across all possible budget points
    cumulative_invest  = np.arange(1, n_total + 1) * retention_cost
    cumulative_revenue = (
        churners["Churn_Probability"].cumsum()
        * churners["MonthlyCharges"].expanding().mean()
        * months_lost
        * success_rate
    )
    cumulative_net = cumulative_revenue - cumulative_invest

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_invest, y=cumulative_revenue,
        name="Recoverable Revenue", line=dict(color="#1D9E75", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=cumulative_invest, y=cumulative_invest,
        name="Retention Spend", line=dict(color="#E24B4A", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=cumulative_invest, y=cumulative_net,
        name="Net Value", line=dict(color="#185FA5", width=2),
        fill="tozeroy", fillcolor="rgba(24,95,165,0.08)",
    ))

    # Mark break-even
    break_even_idx = np.searchsorted(cumulative_net.values, 0)
    if break_even_idx < n_total:
        fig.add_vline(
            x=float(cumulative_invest.iloc[break_even_idx]),
            line_dash="dot", line_color="#EF9F27",
            annotation_text="Break-even",
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis_title="Retention Investment (₹)",
        yaxis_title="Revenue (₹)",
        legend=dict(orientation="h", y=1.05),
        height=380,
        margin=dict(t=40, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)
