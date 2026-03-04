from pptx import Presentation
from pptx.util import Inches

def export_ppt(bundle):

    summary = bundle["summary"]

    prs = Presentation()

    slide = prs.slides.add_slide(
        prs.slide_layouts[5]
    )

    box = slide.shapes.add_textbox(
        Inches(1),
        Inches(1),
        Inches(8),
        Inches(3)
    )

    tf = box.text_frame
    tf.text = (
        f"Revenue at Risk: ₹{summary['Revenue at Risk']:,.0f}"
    )

    slide = prs.slides.add_slide(
        prs.slide_layouts[5]
    )

    slide.shapes.add_picture(
        "reports/figures/revenue_by_segment.png",
        Inches(1),
        Inches(1),
        width=Inches(6)
    )

    prs.save(
        "reports/churnguard_report.pptx"
    )