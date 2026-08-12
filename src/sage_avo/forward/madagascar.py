"""Optional Madagascar implementation of the historical exact forward path."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import numpy as np

from .pipeline import ForwardConfig, ForwardResult
from .stacks import apply_front_mute, stack_bands


@dataclass(frozen=True)
class MadagascarAvailability:
    available: bool
    missing_commands: tuple[str, ...]


def madagascar_availability() -> MadagascarAvailability:
    """Report whether every command required by the production pipeline exists."""
    commands = ("sfbin2rsf", "sfzoeppritz2", "sftransp", "sfricker1", "sfrsf2bin")
    missing = tuple(command for command in commands if shutil.which(command) is None)
    return MadagascarAvailability(not missing, missing)


def _quote(path: Path) -> str:
    return shlex.quote(str(path))


def forward_avo_madagascar(
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    config: ForwardConfig = ForwardConfig(),
    *,
    time_origin_seconds: float = 0.0,
    trace_origin: float = 0.0,
) -> ForwardResult:
    """Run ``sfzoeppritz2 -> sfricker1`` and return the common forward contract.

    This preserves the historical production route for local cross-checking.
    The dependency-light NumPy solver remains an exact Zoeppritz implementation,
    not an Aki--Richards or Shuey substitute.
    """
    availability = madagascar_availability()
    if not availability.available:
        raise RuntimeError(f"Madagascar commands are missing: {availability.missing_commands}")
    arrays = [np.asarray(value, dtype=np.float32) for value in (vp, vs, density)]
    if any(value.ndim != 2 for value in arrays) or len({value.shape for value in arrays}) != 1:
        raise ValueError("vp, vs, and density must be matching [time, trace] arrays")
    angles = np.asarray(config.angles_degrees, dtype=float)
    if angles.size < 2 or not np.allclose(np.diff(angles), np.diff(angles)[0]):
        raise ValueError("Madagascar sfzoeppritz2 requires a regular angle axis")
    height, width = arrays[0].shape
    with tempfile.TemporaryDirectory(prefix="sage_avo_rsf_") as directory:
        work = Path(directory)
        environment = {**os.environ, "DATAPATH": f"{work}/"}
        rsf_paths = []
        for name, array in zip(("vp", "vs", "rho"), arrays):
            binary = work / f"{name}.bin"
            array.flatten(order="F").tofile(binary)
            rsf = work / f"{name}.rsf"
            command = (
                f"sfbin2rsf bfile={_quote(binary)} n1={height} n2={width} "
                f"d1={config.dt_seconds} o1={time_origin_seconds} d2=1 o2={trace_origin} "
                f"> {_quote(rsf)}"
            )
            subprocess.run(
                ["bash", "-lc", command],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            rsf_paths.append(rsf)
        dense_rsf = work / "dense.rsf"
        dense_binary = work / "dense.bin"
        command = (
            f"sfzoeppritz2 < {_quote(rsf_paths[0])} vs={_quote(rsf_paths[1])} "
            f"rho={_quote(rsf_paths[2])} a0={angles[0]} da={angles[1] - angles[0]} na={angles.size} "
            f"| sftransp | sfricker1 frequency={config.wavelet_hz} | sftransp > {_quote(dense_rsf)}"
        )
        subprocess.run(
            ["bash", "-lc", command],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        subprocess.run(
            ["sfrsf2bin", f"bfile={dense_binary}"],
            stdin=dense_rsf.open("rb"),
            check=True,
            capture_output=True,
            env=environment,
        )
        data = np.fromfile(dense_binary, dtype=np.float32)
        expected = angles.size * height * width
        if data.size != expected:
            raise RuntimeError(f"Madagascar returned {data.size} samples; expected {expected}")
        seismic = data.reshape((width, height, angles.size)).transpose(2, 1, 0)
    if config.apply_mute:
        seismic = apply_front_mute(seismic, angles, config.dt_seconds)
    stacks = stack_bands(seismic, angles, config.bands)
    return ForwardResult(
        reflectivity=np.full_like(seismic, np.nan),
        seismic=seismic.astype(np.float32),
        stacks=stacks.astype(np.float32),
        angles_degrees=angles.astype(np.float32),
        band_names=tuple(item.name for item in config.bands),
    )
