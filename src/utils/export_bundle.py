"""
Bundle PDF + PPTX + charts into a single zip.
Fixed: now returns bytes (was returning str path — incompatible with
st.download_button(data=...) which expects bytes).
"""
import io
import logging
import zipfile
from src.config import REPORTS_DIR, FIGURES_DIR, PDF_PATH, PPT_PATH

logger = logging.getLogger(__name__)


def export_bundle() -> bytes:
    """
    Returns raw zip bytes for Streamlit's st.download_button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in [PDF_PATH, PPT_PATH]:
            if path.exists():
                zf.write(path, path.name)
                logger.info("Bundled: %s", path.name)
            else:
                logger.warning("Missing, skipped: %s", path.name)

        if FIGURES_DIR.exists():
            for png in sorted(FIGURES_DIR.glob("*.png")):
                zf.write(png, f"figures/{png.name}")

    buf.seek(0)
    return buf.read()
