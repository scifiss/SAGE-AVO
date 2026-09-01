import json
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_matched_training_v00332e import (  # noqa: E402
    _archive_incomplete_run,
    _last_completed_epoch,
)


def test_last_completed_epoch_defaults_to_zero(tmp_path):
    run = tmp_path / "full"
    run.mkdir()
    assert _last_completed_epoch(run) == 0
    (run / "manifest.json").write_text(
        json.dumps({"last_completed_epoch": 3}), encoding="utf-8"
    )
    assert _last_completed_epoch(run) == 3


def test_archive_incomplete_run_preserves_files(tmp_path):
    run = tmp_path / "full"
    run.mkdir()
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    archived = _archive_incomplete_run(run)
    assert not run.exists()
    assert archived.exists()
    assert (archived / "manifest.json").exists()


def test_archive_refuses_completed_run(tmp_path):
    run = tmp_path / "full"
    run.mkdir()
    (run / "last.pt").write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="without a completed epoch"):
        _archive_incomplete_run(run)
