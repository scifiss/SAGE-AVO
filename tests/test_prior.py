import numpy as np

from sage_avo.data.prior import (
    PriorDefinition,
    make_low_frequency_prior,
    make_truth_derived_prior,
)


def test_synthetic_and_field_prior_use_the_same_configured_filter():
    rng = np.random.default_rng(7)
    elastic = rng.normal(size=(3, 64, 41))
    synthetic_definition = PriorDefinition(truth_derived=True)
    field_definition = PriorDefinition(
        source="field_conditioned_elastic_background",
        truth_derived=False,
    )
    synthetic = make_truth_derived_prior(elastic, synthetic_definition)
    field = make_low_frequency_prior(elastic, field_definition)
    np.testing.assert_array_equal(synthetic, field)
