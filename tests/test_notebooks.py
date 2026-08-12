import json
from pathlib import Path


def test_notebooks_are_valid_and_clear_of_saved_errors():
    root = Path(__file__).resolve().parents[1]
    notebooks = [root / "notebooks" / "01_structure_model.ipynb"]
    assert notebooks[0].is_file()
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
