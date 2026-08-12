"""Exact P-P Zoeppritz reflectivity for isotropic elastic interfaces."""

from __future__ import annotations

import numpy as np


def zoeppritz_pp(
    vp1: np.ndarray | float,
    vs1: np.ndarray | float,
    rho1: np.ndarray | float,
    vp2: np.ndarray | float,
    vs2: np.ndarray | float,
    rho2: np.ndarray | float,
    angle_degrees: float,
) -> np.ndarray:
    """Return exact pre-critical P-P reflection coefficients.

    Inputs broadcast following NumPy rules. Post-critical arguments are handled
    with complex arithmetic; the real component is returned for the real-valued
    synthetic workflow.
    """
    values = np.broadcast_arrays(vp1, vs1, rho1, vp2, vs2, rho2)
    upper_vp, upper_vs, upper_rho, lower_vp, lower_vs, lower_rho = [
        np.asarray(value, dtype=np.complex128) for value in values
    ]
    theta1 = np.deg2rad(angle_degrees)
    ray_parameter = np.sin(theta1) / upper_vp
    theta2 = np.arcsin(ray_parameter * lower_vp)
    phi1 = np.arcsin(ray_parameter * upper_vs)
    phi2 = np.arcsin(ray_parameter * lower_vs)

    matrix = np.empty(upper_vp.shape + (4, 4), dtype=np.complex128)
    matrix[..., 0, 0] = -np.sin(theta1)
    matrix[..., 0, 1] = -np.cos(phi1)
    matrix[..., 0, 2] = np.sin(theta2)
    matrix[..., 0, 3] = np.cos(phi2)
    matrix[..., 1, 0] = np.cos(theta1)
    matrix[..., 1, 1] = -np.sin(phi1)
    matrix[..., 1, 2] = np.cos(theta2)
    matrix[..., 1, 3] = -np.sin(phi2)
    matrix[..., 2, 0] = 2 * upper_rho * upper_vs * np.sin(phi1) * np.cos(theta1)
    matrix[..., 2, 1] = upper_rho * upper_vs * (1 - 2 * np.sin(phi1) ** 2)
    matrix[..., 2, 2] = 2 * lower_rho * lower_vs * np.sin(phi2) * np.cos(theta2)
    matrix[..., 2, 3] = lower_rho * lower_vs * (1 - 2 * np.sin(phi2) ** 2)
    matrix[..., 3, 0] = -upper_rho * upper_vp * (1 - 2 * np.sin(phi1) ** 2)
    matrix[..., 3, 1] = upper_rho * upper_vs * np.sin(2 * phi1)
    matrix[..., 3, 2] = lower_rho * lower_vp * (1 - 2 * np.sin(phi2) ** 2)
    matrix[..., 3, 3] = -lower_rho * lower_vs * np.sin(2 * phi2)

    rhs = np.empty(upper_vp.shape + (4,), dtype=np.complex128)
    rhs[..., 0] = np.sin(theta1)
    rhs[..., 1] = np.cos(theta1)
    rhs[..., 2] = 2 * upper_rho * upper_vs * np.sin(phi1) * np.cos(theta1)
    rhs[..., 3] = upper_rho * upper_vp * (1 - 2 * np.sin(phi1) ** 2)

    solution = np.linalg.solve(matrix, rhs)
    return np.real_if_close(solution[..., 0], tol=1000).real


def reflectivity_gather(
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    angles_degrees: np.ndarray,
) -> np.ndarray:
    """Build ``[angle, time, trace]`` exact P-P reflectivity."""
    properties = [np.asarray(value, dtype=float) for value in (vp, vs, density)]
    if any(value.ndim != 2 for value in properties) or len({value.shape for value in properties}) != 1:
        raise ValueError("vp, vs, and density must be matching 2-D arrays")
    height, width = properties[0].shape
    result = np.zeros((len(angles_degrees), height, width), dtype=np.float64)
    for index, angle in enumerate(np.asarray(angles_degrees, dtype=float)):
        result[index, 1:] = zoeppritz_pp(
            properties[0][:-1],
            properties[1][:-1],
            properties[2][:-1],
            properties[0][1:],
            properties[1][1:],
            properties[2][1:],
            float(angle),
        )
    return result
