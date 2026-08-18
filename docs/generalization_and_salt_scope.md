# Generalization and salt/subsalt scope

The current SAGE-AVO corpus is a target-field-conditioned experiment. Its
structural coordinate, reservoir geometry, elastic background, and empirical
property relationships are derived from the configured S01 `ss.segy` line,
interpreted horizons, and available wells. Synthetic members perturb that
conditioned model; they do not establish universal geological coverage.

The Stage-02 observation model is an exact local PP Zoeppritz reflectivity
operator followed by configured wavelet convolution, angle-domain muting,
three-band stacking, and post-forward observational perturbations. This is an
appropriate controlled AVA operator for the present experiment, but it does
not model illumination, multipathing, transmission loss, shadow zones, or
uncertainty in subsalt reflection angles.

For that reason, an arbitrary salt body is not inserted into the current
Zoeppritz-convolution family. A scientifically defensible salt/subsalt study
would be a separate domain with its own velocity and density family, survey
geometry, and wave-equation modeling. Its gathers should be produced and
quality-controlled with a suitable finite-difference or spectral wave solver
and an RTM/FWI-style imaging workflow, including illumination and angle-gather
uncertainty. Transfer from the present model to that domain would then require
an explicit generalization experiment rather than a visual salt perturbation.

Accordingly, current claims are limited to the configured field-conditioned
family and its declared perturbation ranges. Generalization to other basins,
acquisition systems, or salt/subsalt settings remains future controlled work.
