from datetime import datetime


def generate_churn_report(summary: dict, seg_df, total_budget: float, retention_cost: float) -> str:
    return f"""
ChurnGuard AI – CFO Retention Strategy Memo
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Total Retention Budget: ₹{total_budget:,}
Retention Cost / Customer: ₹{retention_cost:,}

Revenue at Risk:    ₹{summary.get('Revenue at Risk', 0):,.0f}
Medium Risk ROI:    ₹{summary.get('Medium Risk ROI', 0):,.2f}
High Risk ROI:      ₹{summary.get('High Risk ROI', 0):,.2f}

Segment Strategy:
{seg_df.to_string() if seg_df is not None else 'N/A'}
""".strip()
