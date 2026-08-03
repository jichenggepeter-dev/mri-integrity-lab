from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_an_uploaded_image() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "MRI Integrity Lab"
    assert "Upload one brain image" in app.info[0].value
