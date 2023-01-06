"""Tests `create_average_measurement` noise distributions are as expected."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-member, no-self-use

from typing import Dict, List, Union

from nose.plugins.attrib import attr
from pyspark.sql import functions as sf

from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_average_measurement,
    get_midpoint,
)
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
    get_values_summing_to_loc,
    run_test_using_chi_squared_test,
    run_test_using_ks_test,
)

from . import NOISE_SCALE_FUDGE_FACTOR, P_THRESHOLD, SAMPLE_SIZE


def _get_average_test_cases(noise_mechanism: NoiseMechanism) -> List[Dict]:
    """Returns average test cases.

    This returns a list of test instances specifying the sampler (that produces
    a count sample and a sum sample used to compute the average), expected locations
    for count and sum, expected noise scales and corresponding cdfs (if noise mechanism
    is Laplace) or cmfs and pmfs (if noise mechanism is not Laplace).
    """
    test_cases = []
    sum_locations: Union[List[float], List[int]]
    supports_continuous = noise_mechanism in (
        NoiseMechanism.LAPLACE,
        NoiseMechanism.GAUSSIAN,
    )
    if not supports_continuous:
        sum_locations = [100, 14]  # Must be integers
    else:
        sum_locations = [99.78, 13.63]
    count_locations = [8, 5]
    privacy_budgets = ["0.8", "0.3"]
    for sum_loc, count_loc, budget in zip(
        sum_locations, count_locations, privacy_budgets
    ):
        group_values = get_values_summing_to_loc(sum_loc, n=count_loc)
        dataset = FixedGroupDataSet(
            group_vals=group_values,
            num_groups=SAMPLE_SIZE,
            float_measure_column=supports_continuous,
        )
        measurement = create_average_measurement(
            input_domain=dataset.domain,
            input_metric=SymmetricDifference(),
            output_measure=PureDP()
            if noise_mechanism
            not in (NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN)
            else RhoZCDP(),
            measure_column="B",
            lower=dataset.lower,
            upper=dataset.upper,
            noise_mechanism=noise_mechanism,
            d_out=budget,
            groupby_transformation=dataset.groupby(noise_mechanism),
            keep_intermediates=True,
            count_column="count",
            sum_column="sod",
        )

        true_answers: Dict[str, Union[float, int]] = {
            "count": len(dataset.group_vals),
            "sum": sum(dataset.group_vals),
        }
        midpoint, _ = get_midpoint(
            dataset.lower,
            dataset.upper,
            integer_midpoint=not dataset.float_measure_column,
        )
        postprocessor = lambda df, count=count_loc, midpoint=midpoint: df.withColumn(
            "sum", sf.col("sod") + sf.lit(count) * sf.lit(midpoint)
        ).select("count", "sum")
        sampler = get_sampler(measurement, dataset, postprocessor)
        noise_scales = get_noise_scales(
            agg="average",
            budget=budget,
            dataset=dataset,
            noise_mechanism=noise_mechanism,
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


class TestAverageNoiseDistributions(PySparkTest):
    """Distribution tests for create_average_measurement."""

    @attr("slow")
    def test_average_with_laplace_noise(self):
        """Average adds noise from expected Laplace distribution."""
        cases = [
            KSTestCase.from_dict(e)
            for e in _get_average_test_cases(NoiseMechanism.LAPLACE)
        ]
        for case in cases:
            run_test_using_ks_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_average_with_geometric_noise(self):
        """Average adds noise from expected geometric distribution."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in _get_average_test_cases(NoiseMechanism.GEOMETRIC)
        ]
        for case in cases:
            run_test_using_chi_squared_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_average_with_discrete_gaussian_noise(self):
        """Average adds noise from expected discrete Gaussian distribution."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in _get_average_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN)
        ]
        for case in cases:
            run_test_using_chi_squared_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_average_with_gaussian_noise(self):
        """Average adds noise from expected Gaussian distribution."""
        cases = [
            KSTestCase.from_dict(e)
            for e in _get_average_test_cases(NoiseMechanism.GAUSSIAN)
        ]
        for case in cases:
            run_test_using_ks_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)
