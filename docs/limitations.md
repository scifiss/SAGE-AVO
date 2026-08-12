# Scientific limitations

1. **Conditional prior refinement.** Synthetic evaluation supplies
   truth-derived low-frequency elastic priors. Results measure residual-detail
   recovery, not AVO-only absolute inversion.

2. **Field wells are not blind.** Available wells contribute upstream to
   horizon calibration, facies modeling, rock-physics learning, or the field
   prior. Their final comparison is quality control, not independent validation.

3. **Field domain gap.** Each field deployment requires versioned near/mid/far
   forward-consistency QC. Per-band scale, correlation, and normalized RMSE
   quantify synthetic-to-field mismatch.

4. **Synthetic family coverage.** Realizations perturb one field-conditioned
   structural and rock-physics family. Realization-level splitting prevents
   patch leakage but does not create independent regional geology.

5. **Forward-model inverse crime.** Synthetic data and physics loss use closely
   related forward assumptions. Wavelet, phase, Q, noise, scaling, and processing
   variation require broader augmentation.

6. **Dataset-specific SEG-Y semantics.** The local S01 configuration interprets
   a configured gather-coordinate header as incidence-angle bins. This requires
   support from acquisition/export metadata; matching numerical bins alone does
   not establish semantic meaning. Other datasets require independent metadata
   verification or offset-to-angle conversion.

7. **Approximate depth assumptions.** Fluid-substitution demonstrations map
   time samples to nominal depth increments. Production rock physics requires a
   calibrated depth/time model.

8. **Multiscale resizing.** Different raw physical windows are resized to one
   tensor shape. Scale metadata is retained, but the current network does not
   yet condition explicitly on scale.

9. **Deterministic flow.** The current straight-path flow has no stochastic
   latent state and does not represent a calibrated posterior.

10. **Sensitivity scope.** Checkpoint/prior-cutoff ensembles characterize
    model and prior sensitivity. Posterior uncertainty requires a calibrated
    probabilistic formulation.
