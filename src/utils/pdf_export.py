"""
PDF export — CFO financial memo.
Fixed: single font registration via font_setup.py (was registered twice).
Enhanced: includes segment table, not just 3 lines of text.
"""
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)
from src.config import PDF_PATH, REPORTS_DIR
from src.utils.font_setup import setup_fonts

logger = logging.getLogger(__name__)


def export_pdf(bundle: dict):
    """Generate a structured CFO memo PDF with assumptions, KPIs, and segment table."""
    setup_fonts()   # idempotent — safe to call multiple times
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = bundle["summary"]
    seg_df  = bundle["segmentation"]
    cost    = bundle["retention_cost"]
    months  = bundle["months_lost"]
    success = bundle["success_rate"]
    budget  = bundle["budget"]

    doc    = SimpleDocTemplate(str(PDF_PATH), pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    H1     = styles["Title"]
    H2     = ParagraphStyle("h2", parent=styles["Heading2"],
                            spaceBefore=12, spaceAfter=4)
    BODY   = styles["Normal"]
    story  = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story += [
        Paragraph("ChurnGuard AI — CFO Retention Memo", H1),
        Spacer(1, 16),
    ]

    # ── Campaign assumptions ───────────────────────────────────────────────────
    story += [
        Paragraph("Campaign Assumptions", H2),
        Paragraph(f"Retention cost / customer: ₹{cost:,}", BODY),
        Paragraph(f"Campaign success rate:      {success*100:.0f}%", BODY),
        Paragraph(f"Months of revenue at risk:  {months}", BODY),
        Paragraph(f"Total retention budget:     ₹{budget:,}", BODY),
        Spacer(1, 12),
    ]

    # ── KPI summary ───────────────────────────────────────────────────────────
    story += [Paragraph("Headline KPIs", H2)]
    kpi_rows = [
        ["Metric", "Value"],
        ["Revenue at Risk",           f"₹{summary.get('Revenue at Risk', 0):,.0f}"],
        ["Total Recoverable Revenue", f"₹{summary.get('Total Recoverable Revenue', 0):,.0f}"],
        ["Net Value Created",         f"₹{summary.get('Net Value Created', 0):,.0f}"],
        ["Overall ROI",               f"{summary.get('Overall ROI', 0)*100:.1f}%"],
    ]
    kpi_table = Table(kpi_rows, colWidths=[10*cm, 5*cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F1F5F9"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))
    story += [kpi_table, Spacer(1, 16)]

    # ── Segment strategy table ────────────────────────────────────────────────
    if not seg_df.empty:
        story += [Paragraph("Segment ROI Strategy", H2)]
        display_cols = ["Risk_Tier", "Customers", "Revenue_at_Risk",
                        "Retention_Investment", "Net_Value_Created", "Recommendation"]
        avail = [c for c in display_cols if c in seg_df.columns]
        headers = [c.replace("_", " ") for c in avail]

        seg_rows = [headers]
        for _, row in seg_df[avail].iterrows():
            seg_rows.append([
                str(row[c]) if c in ["Risk_Tier", "Recommendation"]
                else f"₹{row[c]:,.0f}" if isinstance(row[c], float)
                else str(row[c])
                for c in avail
            ])

        col_w = [16*cm / len(avail)] * len(avail)
        seg_table = Table(seg_rows, colWidths=col_w)
        seg_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#F1F5F9"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ]))
        story.append(seg_table)

    try:
        doc.build(story)
        logger.info("PDF saved → %s", PDF_PATH)
    except Exception as e:
        logger.error("PDF generation failed: %s", e)
        raise
