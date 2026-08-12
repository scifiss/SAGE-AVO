"""Regression tests for optional dependency boundaries."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_core_imports_do_not_require_optional_dependencies() -> None:
    """Core modules must import with only the dependencies installed by CI."""
    script = """
        import importlib.abc
        import sys

        class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
            blocked = {"joblib", "lasio", "pyseistr", "segyio", "sklearn", "torch", "torch_geometric"}

            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in self.blocked:
                    raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
                return None

        sys.meta_path.insert(0, BlockOptionalDependencies())

        from sage_avo.data.prior import PriorDefinition
        from sage_avo.experiments.dataset import prepare_controlled_dataset
        from sage_avo.geology.conventions import delta_from_sand_probability

        assert PriorDefinition
        assert prepare_controlled_dataset
        assert delta_from_sand_probability
    """
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
