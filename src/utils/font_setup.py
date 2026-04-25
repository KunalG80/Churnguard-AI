"""
Single point of font registration for ReportLab.
Import and call setup_fonts() once before any PDF generation.
"""

import logging
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from src.config import FONT_PATH

logger = logging.getLogger(__name__)
_fonts_registered = False


def setup_fonts():
    """Register DejaVu font once per process; safe to call multiple times."""
    global _fonts_registered
    if _fonts_registered:
        return
    if Path(FONT_PATH).exists():
        pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_PATH)))
        logger.debug("DejaVu font registered from %s", FONT_PATH)
    else:
        logger.warning(
            "Font file not found at %s — falling back to Helvetica.", FONT_PATH
        )
    _fonts_registered = True
