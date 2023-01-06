"""Tests that count measurement adds noise from the correct distribution."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-member, no-self-use

from typing import Dict, Union

from nose.plugins.attrib import attr

from tmlt.core.measurements.aggregations import NoiseMechanism, create_count_measurement
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.utils.testing import (
    ChiSquaredTestCase,
    FixedGroupDataSet,
    KSTestCase,
    PySparkTest,
    get_noise_scales,
    get_prob_functions,
    get_sampler,
    run_test_using_chi_squared_test,
    run_test_using_ks_test,
)

from . import NOISE_SCALE_FUDGE_FACTOR, P_THRESHOLD, SAMPLE_SIZE


def _get_count_test_cases(noise_mechanism: NoiseMechanism):
    """Returns count test cases.

    This returns a list of 4 test instances specifying the sampler (that produces
    a count sample), expected count location, expected noise scale and corresponding
    cdf (if noise mechanism is Laplace) or cmf and pmf (if noise mechanism is not
    Laplace).

    Each of the 4 samplers produces a sample of size SAMPLE_SIZE.
      * 2 samplers that compute noisy groupby-count once on a DataFrame with
         # groups = SAMPLE_SIZE. These two samplers have different true counts
         and different noise scales.
      * 2 samplers that compute noisy groupby-count 200 times on a DataFrame with
         # groups = SAMPLE_SIZE/200. These two samplers have different true counts
         and different noise scales.
    """
    test_cases = []
    count_locations = [10, 45]
    privacy_budgets = [1, "0.4"]
    for count_loc, budget in zip(count_locations, privacy_budgets):
        dataset = FixedGroupDataSet(
            group_vals=list(range(count_loc)), num_groups=SAMPLE_SIZE
        )
        true_answers: Dict[str, Union[float, int]] = {"count": len(dataset.group_vals)}
        measurement = create_count_measurement(
            input_domain=dataset.domain,
            input_metric=SymmetricDifference(),
            output_measure=PureDP()
            if noise_mechanism
            not in (NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN)
            else RhoZCDP(),
            d_out=budget,
            noise_mechanism=noise_mechanism,
            groupby_transformation=dataset.groupby(noise_mechanism),
            count_column="count",
        )
        sampler = get_sampler(measurement, dataset, lambda df: df.select("count"))
        noise_scales = get_noise_scales(
            agg="count", budget=budget, dataset=dataset, noise_mechanism=noise_mechanism
        )
        prob_functions = get_prob_functions(noise_mechanism, true_answers)
        test_cases.append(
            {
                "sampler": sampler,
                "locations": true_answers,
                "scales": noise_scales,
                **prob_functions,
            }
        )
    return test_cases


class TestCountNoiseDistributions(PySparkTest):
    """Distribution tests for create_count_measurement."""

    @attr("slow")
    def test_count_with_geometric_noise(self):
        """`create_count_measurement` has expected geometric distribution."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in _get_count_test_cases(NoiseMechanism.GEOMETRIC)
        ]
        for case in cases:
            run_test_using_chi_squared_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_count_with_discrete_gaussian_noise(self):
        """`create_count_measurement` has expected discrete Gaussian distribution."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in _get_count_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN)
        ]
        for case in cases:
            run_test_using_chi_squared_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_count_with_laplace_noise(self):
        """`create_count_measurement` has expected Laplace distribution."""
        cases = [
            KSTestCase.from_dict(e)
            for e in _get_count_test_cases(NoiseMechanism.LAPLACE)
        ]
        for case in cases:
            run_test_using_ks_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_count_with_gaussian_noise(self):
        """`create_count_measurement` has expected Gaussian distribution."""
        cases = [
            KSTestCase.from_dict(e)
            for e in _get_count_test_cases(NoiseMechanism.GAUSSIAN)
        ]
        for case in cases:
            run_test_using_ks_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)
