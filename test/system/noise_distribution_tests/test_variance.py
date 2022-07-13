"""Tests `create_variance_measurement` noise distributions are as expected."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-member, no-self-use

from typing import Dict, List, Union

from nose.plugins.attrib import attr
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_standard_deviation_measurement,
    create_variance_measurement,
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


def _get_var_stddev_test_cases(
    noise_mechanism: NoiseMechanism, stddev: bool
) -> List[Dict]:
    """Returns variance or stddev test cases.

    This returns a list of test instances specifying the sampler that produces samples
    for count, sum and sum of squares, the expected locations and noise scales for each
    and corresponding cdfs (if noise_mechanism is Laplace) or cmfs and pmfs (otherwise)
    """
    test_cases = []
    sum_locations: Union[List[float], List[int]]
    if noise_mechanism != NoiseMechanism.LAPLACE:
        sum_locations = [100, 14]
    else:
        sum_locations = [99.78, 13.63]
    count_locations = [8, 5]
    privacy_budgets = ["3.4", "1.1"]
    for sum_loc, count_loc, budget in zip(
        sum_locations, count_locations, privacy_budgets
    ):
        group_values = get_values_summing_to_loc(sum_loc, n=count_loc)
        dataset = FixedGroupDataSet(
            group_vals=group_values,
            num_groups=SAMPLE_SIZE,
            float_measure_column=noise_mechanism == NoiseMechanism.LAPLACE,
        )
        create_measurement = (
            create_standard_deviation_measurement
            if stddev
            else create_variance_measurement
        )
        measurement = create_measurement(
            input_domain=dataset.domain,
            input_metric=SymmetricDifference(),
            output_measure=PureDP()
            if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
            else RhoZCDP(),
            measure_column="B",
            lower=dataset.lower,
            upper=dataset.upper,
            noise_mechanism=noise_mechanism,
            d_out=budget,
            groupby_transformation=dataset.groupby(noise_mechanism),
            keep_intermediates=True,
            sum_of_deviations_column="sod",
            sum_of_squared_deviations_column="sos",
            count_column="count",
        )

        true_answers: Dict[str, Union[float, int]] = {
            "count": len(dataset.group_vals),
            "sum": sum(dataset.group_vals),
            "sum_of_squares": sum(val ** 2 for val in dataset.group_vals),
        }
        midpoint_sod, _ = get_midpoint(
            dataset.lower,
            dataset.upper,
            integer_midpoint=not dataset.float_measure_column,
        )
        midpoint_sos, _ = get_midpoint(
            0 if dataset.lower <= 0 <= dataset.upper else dataset.lower ** 2,
            dataset.upper ** 2,
            integer_midpoint=not dataset.float_measure_column,
        )

        def postprocessor(
            df: DataFrame,
            count: int = count_loc,
            midpoint_sod: float = midpoint_sod,
            midpoint_sos: float = midpoint_sos,
        ):
            """Postprocess the output to pull out the original measurements."""
            return (
                df.withColumn(
                    "sum", sf.col("sod") + (sf.lit(count) * sf.lit(midpoint_sod))
                )
                .withColumn(
                    "sum_of_squares",
                    sf.col("sos") + (sf.lit(count) * sf.lit(midpoint_sos)),
                )
                .select("count", "sum", "sum_of_squares")
            )

        sampler = get_sampler(measurement, dataset, postprocessor)
        noise_scales = get_noise_scales(
            agg="standard deviation" if stddev else "variance",
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


class TestVarianceNoiseDistributions(PySparkTest):
    """Noise distributions test for `create_variance_measurement`."""

    @attr("slow")
    def test_variance_with_laplace_noise(self):
        """`create_variance_measurement` adds appropriate Laplace noise."""
        cases = [
            KSTestCase.from_dict(e)
            for e in _get_var_stddev_test_cases(NoiseMechanism.LAPLACE, stddev=False)
        ]
        for case in cases:
            run_test_using_ks_test(
                case,
                p_threshold=P_THRESHOLD,
                noise_scale_fudge_factor=NOISE_SCALE_FUDGE_FACTOR,
            )

    @attr("slow")
    def test_variance_with_geometric_noise(self):
        """`create_variance_measurement` adds appropriate geometric noise."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in _get_var_stddev_test_cases(NoiseMechanism.GEOMETRIC, stddev=False)
        ]
        for case in cases:
            run_test_using_chi_squared_test(
                case,
                p_threshold=P_THRESHOLD,
                noise_scale_fudge_factor=NOISE_SCALE_FUDGE_FACTOR,
            )

    @attr("slow")
    def test_variance_with_discrete_gaussian_noise(self):
        """`create_variance_measurement` adds appropriate discrete Gaussian noise."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in _get_var_stddev_test_cases(
                NoiseMechanism.DISCRETE_GAUSSIAN, stddev=False
            )
        ]
        for case in cases:
            run_test_using_chi_squared_test(
                case,
                p_threshold=P_THRESHOLD,
                noise_scale_fudge_factor=NOISE_SCALE_FUDGE_FACTOR,
            )
