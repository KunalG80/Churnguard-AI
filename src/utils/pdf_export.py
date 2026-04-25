import logging
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from src.config import REPORTS_DIR, FONT_PATH
from src.utils.report_generator import generate_churn_report

logger = logging.getLogger(__name__)

_font_registered = False

def _register_font():
    global _font_registered
    if _font_registered:
        return
    if FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_PATH)))
    _font_registered = True


def export_pdf(bundle: dict):
    _register_font()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary     = bundle["summary"]
    cost        = bundle["retention_cost"]
    months      = bundle["months_lost"]
    success     = bundle["success_rate"]
    budget      = bundle["budget"]
    seg_df      = bundle["segmentation"]

    out_path = REPORTS_DIR / "churnguard_report.pdf"
    doc    = SimpleDocTemplate(str(out_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("<b>ChurnGuard AI — CFO Retention Memo</b>", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Retention cost per customer: ₹{cost}", styles["Normal"]))
    story.append(Paragraph(f"Campaign success rate: {success*100:.0f}%", styles["Normal"]))
    story.append(Paragraph(f"Months at risk: {months}", styles["Normal"]))
    story.append(Paragraph(f"Total budget: ₹{budget:,}", styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Revenue at Risk: ₹{summary.get('Revenue at Risk', 0):,.0f}", styles["Normal"]))

    doc.build(story)
    logger.info("PDF saved → %s", out_path)
