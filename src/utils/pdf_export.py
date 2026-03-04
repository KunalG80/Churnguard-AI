from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from src.utils.report_generator import generate_churn_report
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(
    TTFont(
        "DejaVu",
        "assets/fonts/DejaVuSans.ttf"
    )
)

def export_pdf(bundle):

    summary = bundle["summary"]
    cost = bundle["retention_cost"]
    months = bundle["months_lost"]
    success = bundle["success_rate"]
    budget = bundle["budget"]
    segmentation = bundle["segmentation"]
    
    report_text = generate_churn_report(
        bundle["summary"],
        bundle["segmentation"],
        bundle["budget"],
        bundle["retention_cost"]
    )

    doc = SimpleDocTemplate(
        "reports/churnguard_report.pdf",
        pagesize=A4
    )

    styles = getSampleStyleSheet()
    story = []

# TITLE
    story.append(
        Paragraph(
            "<b>ChurnGuard AI Financial Memo</b>",
            styles["Title"]
        )
    )

    story.append(Spacer(1,20))

# ASSUMPTIONS
    story.append(
        Paragraph(
            f"Retention Cost: ₹{cost}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Success Rate: {success*100:.0f}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Campaign Duration: {months} months",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Budget: ₹{budget}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1,20))

# KPI
    story.append(
        Paragraph(
            f"Revenue at Risk: ₹{summary['Revenue at Risk']:,.0f}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Medium Risk ROI: ₹{summary['Medium Risk ROI']:,.0f}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"High Risk ROI: ₹{summary['High Risk ROI']:,.0f}",
            styles["Normal"]
        )
    )

    doc.build(story)