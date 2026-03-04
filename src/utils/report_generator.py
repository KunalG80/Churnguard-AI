from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(
    TTFont(
        "DejaVu",
        "assets/fonts/DejaVuSans.ttf"
    )
)

def generate_churn_report(
    summary,
    seg_df,
    total_budget,
    retention_cost
):

    report = f"""
    ChurnGuard AI – CFO Retention Strategy Memo

    Total Retention Budget: ₹{total_budget}
    Retention Cost per Customer: ₹{retention_cost}

    Revenue at Risk: ₹{summary['Revenue at Risk']:,.0f}
    Medium Risk ROI: ₹{summary['Medium Risk ROI']:,.0f}
    High Risk ROI: ₹{summary['High Risk ROI']:,.0f}

    Segment Strategy:
    {seg_df.to_string()}
    """

    return report