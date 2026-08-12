import ast
import json
from pathlib import Path


def test_notebooks_are_valid_and_clear_of_saved_errors():
    root = Path(__file__).resolve().parents[1]
    notebooks = [
        root / "notebooks" / name
        for name in (
            "01_structure_model.ipynb",
            "02_synthetic_avo_generation.ipynb",
            "03_ml_dataset_construction.ipynb",
            "04_sage_avo_training.ipynb",
            "05_evaluation_and_field_application.ipynb",
        )
    ]
    assert all(path.is_file() for path in notebooks)
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=str(path))
