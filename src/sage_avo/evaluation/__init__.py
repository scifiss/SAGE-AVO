"""Elastic, segmentation, forward-consistency, and ablation evaluation."""

from .metrics import elastic_metrics, elastic_metrics_with_ssim, segmentation_metrics, ssim_2d
from .field import field_well_consistency
from .field_calibration import (
    FieldDiagnosticThresholds,
    FieldTransferSpecification,
    apply_field_transfer,
    field_domain_diagnostics,
    load_passing_field_calibration,
    prepare_calibrated_field_input,
    prepare_calibrated_field_observation,
    save_field_calibration_manifest,
)

__all__ = [
    "elastic_metrics",
    "elastic_metrics_with_ssim",
    "field_well_consistency",
    "FieldDiagnosticThresholds",
    "FieldTransferSpecification",
    "apply_field_transfer",
    "field_domain_diagnostics",
    "load_passing_field_calibration",
    "prepare_calibrated_field_input",
    "prepare_calibrated_field_observation",
    "save_field_calibration_manifest",
    "segmentation_metrics",
    "ssim_2d",
]
