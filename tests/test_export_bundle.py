import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.export_bundle import export_bundle

def test_returns_bytes():
    """FIX: was returning str path — st.download_button needs bytes."""
    result = export_bundle()
    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"

def test_returns_valid_zip():
    import zipfile, io
    result = export_bundle()
    assert zipfile.is_zipfile(io.BytesIO(result))
