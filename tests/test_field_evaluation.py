from pathlib import Path

import numpy as np
import pandas as pd

from sage_avo.evaluation import field_well_consistency


def test_field_well_consistency_samples_prediction_in_time_and_space(tmp_path: Path):
    time_ms = np.arange(10, dtype=float) * 4.0 + 2000.0
    line_xy = np.column_stack((np.arange(4, dtype=float) * 100.0, np.zeros(4)))
    prediction = np.empty((3, 10, 4), dtype=np.float32)
    for channel, base in enumerate((2500.0, 1400.0, 2.2)):
        prediction[channel] = base + channel + np.arange(10)[:, None]
    wells = tmp_path / "wells"
    wells.mkdir()
    frame = pd.DataFrame(
        {
            "WELL": ["A"] * 10,
            "X": [202.0] * 10,
            "Y": [1.0] * 10,
            "TWT_MS": time_ms,
            "VP": prediction[0, :, 2],
            "VS": prediction[1, :, 2],
            "RHOB": prediction[2, :, 2],
        }
    )
    frame.to_csv(wells / "A.csv", index=False)
    metrics, overlays = field_well_consistency(
        prediction, time_ms=time_ms, line_xy=line_xy, wells_directory=wells
    )
    assert len(metrics) == 3
    assert set(metrics["trace_index"]) == {2}
    assert np.allclose(metrics["rmse"], 0.0, atol=1e-6)
    assert len(overlays) == 3
