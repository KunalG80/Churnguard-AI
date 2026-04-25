"""
BUG FIX: This file was imported in app.py but did not exist → ModuleNotFoundError crash.
"""
import io
import zipfile
import logging
from src.config import REPORTS_DIR, FIGURES_DIR

logger = logging.getLogger(__name__)


def export_bundle() -> bytes:
    """
    Bundle the PDF, PPTX, and key chart PNGs into a zip and
    return the raw bytes for Streamlit's st.download_button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["churnguard_report.pdf", "churnguard_report.pptx"]:
            fpath = REPORTS_DIR / fname
            if fpath.exists():
                zf.write(fpath, fname)
            else:
                logger.warning("Report file missing, skipping: %s", fpath)

        if FIGURES_DIR.exists():
            for png in FIGURES_DIR.glob("*.png"):
                zf.write(png, f"figures/{png.name}")

    buf.seek(0)
    return buf.read()
