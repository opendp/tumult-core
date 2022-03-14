"""Tests that measurements which add noise sample from the correct distributions."""

# <placeholder: boilerplate>

# pylint: disable=no-member

import unittest
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union, overload

import numpy as np
import pandas as pd
import sympy as sp
from nose.plugins.attrib import attr
from parameterized import parameterized
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from scipy.stats import chisquare, kstest, laplace

from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_average_measurement,
    create_count_distinct_measurement,
    create_count_measurement,
    create_standard_deviation_measurement,
    create_sum_measurement,
    create_variance_measurement,
    get_midpoint,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise as AddDiscreteGaussianNoiseToNumber,
)
from tmlt.core.measurements.noise_mechanisms import (
    AddGeometricNoise as AddGeometricNoiseToNumber,
)
from tmlt.core.measurements.noise_mechanisms import (
    AddLaplaceNoise as AddLaplaceNoiseToNumber,
)
from tmlt.core.measurements.pandas_measurements.series import (
    AddDiscreteGaussianNoise as AddDiscreteGaussianNoiseToSeries,
)
from tmlt.core.measurements.pandas_measurements.series import (
    AddGeometricNoise as AddGeometricNoiseToSeries,
)
from tmlt.core.measurements.pandas_measurements.series import (
    AddLaplaceNoise as AddLaplaceNoiseToSeries,
)
from tmlt.core.measurements.pandas_measurements.series import (
    NoisyQuantile,
    _get_quantile_probabilities,
)
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.utils.distributions import (
    discrete_gaussian_cmf,
    discrete_gaussian_pmf,
    double_sided_geometric_cmf,
    double_sided_geometric_pmf,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import PySparkTest

P_THRESHOLD = 0.01
"""The alpha threshold to use for the statistical tests."""

NOISE_SCALE_FUDGE_FACTOR = 0.3
"""The amount to perturb the noise for the statistical tests to reject.

We want to get a p value above :data:`P_THRESHOLD` for the actual noise
scale we are using for the mechanism, but we want to get a p value below
:data:`P_THRESHOLD` for the noise scale * (1 +/- :data:`NOISE_SCALE_FUDGE_FACTOR`).
"""

SAMPLE_SIZE = 100000
"""The number of samples to use in the statistical tests."""


@dataclass
class FixedGroupDataSet:
    """Encapsulates a Spark DataFrame with specified number of identical groups."""

    group_vals: Union[List[float], List[int]]
    """Values for each group."""

    num_groups: int
    """Number of identical groups."""

    float_measure_column: bool = False
    """If True, measure column has floating point values."""

    def __post_init__(self):
        """Create groupby transformations and dataframe."""
        spark = SparkSession.builder.getOrCreate()
        group_keys = spark.createDataFrame(
            [(i,) for i in range(self.num_groups)], schema=["A"]
        )
        self._groupby_transformations = {
            "SumOf": GroupBy(
                input_domain=self.domain,
                input_metric=SymmetricDifference(),
                output_metric=SumOf(SymmetricDifference()),
                group_keys=group_keys,
            ),
            "RootSumOfSquared": GroupBy(
                input_domain=self.domain,
                input_metric=SymmetricDifference(),
                output_metric=RootSumOfSquared(SymmetricDifference()),
                group_keys=group_keys,
            ),
        }
        self._dataframe = spark.createDataFrame(
            [(x, val) for x in range(self.num_groups) for val in self.group_vals],
            schema=["A", "B"],
        )

    def groupby(self, noise_mechanism: NoiseMechanism) -> GroupBy:
        """Returns groupby_domains."""
        return self._groupby_transformations[
            "SumOf"
            if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
            else "RootSumOfSquared"
        ]

    @property
    def domain(self) -> SparkDataFrameDomain:
        """Return dataframe domain."""
        return SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkFloatColumnDescriptor()
                if self.float_measure_column
                else SparkIntegerColumnDescriptor(),
            }
        )

    @property
    def lower(self) -> ExactNumber:
        """Returns a lower bound on the values in B."""
        return ExactNumber.from_float(min(self.group_vals), round_up=False)

    @property
    def upper(self) -> ExactNumber:
        """Returns an upper bound on the values in B."""
        return ExactNumber.from_float(max(self.group_vals), round_up=True)

    def get_dataframe(self) -> DataFrame:
        """Returns dataframe."""
        return self._dataframe


@overload
def _get_values_summing_to_loc(loc: int, n: int) -> List[int]:
    ...


@overload
def _get_values_summing_to_loc(loc: float, n: int) -> List[float]:
    ...


def _get_values_summing_to_loc(
    loc, n
):  # pylint: disable=missing-type-doc, missing-return-type-doc
    """Returns a list of n values that sum to loc.

    Args:
        loc: Value to which the return list adds up to. If this is a float,
            a list of floats will be returned, otherwise this must be an int,
            and a list of ints will be returned.
        n: Desired list size.
    """
    assert n > 0
    if n % 2 == 0:
        shifts = [sign * shift for sign in [-1, 1] for shift in range(1, n // 2 + 1)]
    else:
        shifts = list(range(-(n // 2), n // 2 + 1))
    if isinstance(loc, float):
        return [loc / n + shift for shift in shifts]
    assert isinstance(loc, int)
    int_values = [loc // n + shift for shift in shifts]
    int_values[-1] += loc % n
    return int_values


def _create_laplace_cdf(loc: float):
    return lambda value, noise_scale: laplace.cdf(value, loc=loc, scale=noise_scale)


def _create_two_sided_geometric_cmf(loc: int):
    return lambda k, noise_scale: double_sided_geometric_cmf(k - loc, noise_scale)


def _create_two_sided_geometric_pmf(loc: int):
    return lambda k, noise_scale: double_sided_geometric_pmf(k - loc, noise_scale)


def _create_discrete_gaussian_cmf(loc: int):
    return lambda k, noise_scale: discrete_gaussian_cmf(k - loc, noise_scale)


def _create_discrete_gaussian_pmf(loc: int):
    return lambda k, noise_scale: discrete_gaussian_pmf(k - loc, noise_scale)


def _get_sampler(
    measurement: Measurement,
    dataset: FixedGroupDataSet,
    post_processor: Callable[[DataFrame], DataFrame],
    iterations: int = 1,
) -> Callable[[], Dict[str, np.ndarray]]:
    """Returns a sampler function.

    A sampler function takes 0 arguments and produces a numpy array containing samples
    obtaining by performing groupby-agg on the given dataset.

    Args:
        measurement: Measurement to sample from.
        dataset: FixedGroupDataSet object containing DataFrame to perform measurement
            on.
        post_processor: Function to process measurement's output DataFrame and select
            relevant columns.
        iterations: Number of iterations of groupby-agg.
    """

    def sampler() -> Dict[str, np.ndarray]:
        samples = []
        df = dataset.get_dataframe().repartition(200)
        # This is to make sure we catch any correlations across
        # chunks when spark.applyInPandas is called.
        for _ in range(iterations):
            raw_output_df = measurement(df)
            processed_df = post_processor(
                raw_output_df
            ).toPandas()  # Produce columns to be sampled.
            samples.append(
                {col: processed_df[col].values for col in processed_df.columns}
            )
        cols = samples[0].keys()
        return {
            col: np.concatenate([sample_dict[col] for sample_dict in samples])
            for col in cols
        }

    return sampler


def _get_noise_scales(
    agg: str,
    budget: ExactNumberInput,
    dataset: FixedGroupDataSet,
    noise_mechanism: NoiseMechanism,
) -> Dict[str, ExactNumber]:
    """Get noise scale per output column for an aggregation."""
    budget = ExactNumber(budget)
    assert budget > 0
    second_if_dgauss = (
        lambda s1, s2: s1 if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN else s2
    )
    if agg == "count":
        scale = second_if_dgauss(1 / budget, 1 / (2 * budget))
        return {"count": scale}
    if agg == "sum":
        scale = second_if_dgauss(
            dataset.upper / budget, dataset.upper ** 2 / (2 * budget)
        )
        return {"sum": scale}
    if agg == "average":
        sod_sensitivity = (dataset.upper - dataset.lower) / 2
        budget_per_subagg = budget / 2
        sod_scale = second_if_dgauss(
            sod_sensitivity / budget_per_subagg,
            sod_sensitivity ** 2 / (2 * budget_per_subagg),
        )
        count_scale = second_if_dgauss(
            1 / budget_per_subagg, 1 / (2 * budget_per_subagg)
        )
        return {"sum": sod_scale, "count": count_scale}
    if agg in ("standard deviation", "variance"):
        sod_sensitivity = (dataset.upper - dataset.lower) / 2
        sos_sensitivity = (dataset.upper ** 2 - dataset.lower ** 2) / 2
        budget_per_subagg = budget / 3
        sod_scale = second_if_dgauss(
            sod_sensitivity / budget_per_subagg,
            sod_sensitivity ** 2 / (2 * budget_per_subagg),
        )
        sos_scale = second_if_dgauss(
            sos_sensitivity / budget_per_subagg,
            sos_sensitivity ** 2 / (2 * budget_per_subagg),
        )
        count_scale = second_if_dgauss(
            1 / budget_per_subagg, 1 / (2 * budget_per_subagg)
        )
        return {"sum": sod_scale, "count": count_scale, "sum_of_squares": sos_scale}
    raise ValueError(agg)


def _get_output_metric(
    noise_mechanism: NoiseMechanism,
) -> Union[SumOf, RootSumOfSquared]:
    """Returns output metric for given noise mechanism."""
    return (
        SumOf(AbsoluteDifference())
        if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
        else RootSumOfSquared(AbsoluteDifference())
    )


def _get_prob_functions(
    noise_mechanism: NoiseMechanism, locations: Dict[str, Union[float, int]]
) -> Dict[str, Dict[str, Callable]]:
    if noise_mechanism == NoiseMechanism.LAPLACE:
        return {
            "cdfs": {col: _create_laplace_cdf(loc) for col, loc in locations.items()}
        }
    if noise_mechanism == NoiseMechanism.GEOMETRIC:
        assert all(isinstance(val, int) for val in locations.values())
        return {
            "pmfs": {
                col: _create_two_sided_geometric_pmf(int(loc))
                for col, loc in locations.items()
            },
            "cmfs": {
                col: _create_two_sided_geometric_cmf(int(loc))
                for col, loc in locations.items()
            },
        }
    if noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
        assert all(isinstance(val, int) for val in locations.values())
        return {
            "pmfs": {
                col: _create_discrete_gaussian_pmf(int(loc))
                for col, loc in locations.items()
            },
            "cmfs": {
                col: _create_discrete_gaussian_cmf(int(loc))
                for col, loc in locations.items()
            },
        }
    raise ValueError("This should be unreachable.")


# Base Mechanisms Test Instances
def _create_base_laplace_sampler(
    loc: float, noise_scale: ExactNumberInput, sample_size: int
):
    return lambda: {
        "noisy_vals": np.array(
            list(
                map(
                    AddLaplaceNoiseToNumber(
                        scale=noise_scale, input_domain=NumpyFloatDomain()
                    ),
                    [loc] * sample_size,
                )
            )
        )
    }


def _create_vector_laplace_sampler(
    loc: float, noise_scale: ExactNumberInput, sample_size: int
):
    return lambda: {
        "noisy_vals": AddLaplaceNoiseToSeries(
            scale=noise_scale, input_domain=PandasSeriesDomain(NumpyFloatDomain())
        )(pd.Series([loc] * sample_size)).to_numpy()
    }


BASE_LAPLACE_TEST_INSTANCES = [
    {
        "sampler": staticmethod(sampler),
        "locations": {"noisy_vals": loc},
        "scales": {"noisy_vals": noise_scale},
        "cdfs": {"noisy_vals": _create_laplace_cdf(loc)},
    }
    for loc, noise_scale in [(3.5, 0.3), (111.3, 10.123)]
    for sampler in (
        _create_base_laplace_sampler(loc, noise_scale, SAMPLE_SIZE),
        _create_vector_laplace_sampler(loc, noise_scale, SAMPLE_SIZE),
    )
]


def _create_base_geometric_sampler(
    loc: int, noise_scale: ExactNumberInput, sample_size: int
):
    return lambda: {
        "noisy_vals": np.array(
            list(map(AddGeometricNoiseToNumber(alpha=noise_scale), [loc] * sample_size))
        )
    }


def _create_vector_geometric_sampler(
    loc: int, noise_scale: ExactNumberInput, sample_size: int
):
    return lambda: {
        "noisy_vals": AddGeometricNoiseToSeries(
            alpha=noise_scale, input_domain=PandasSeriesDomain(NumpyIntegerDomain())
        )(pd.Series([loc] * sample_size)).to_numpy()
    }


def _create_base_discrete_gaussian_sampler(
    loc: int, noise_scale: ExactNumberInput, sample_size: int
):
    return lambda: {
        "noisy_vals": np.array(
            list(
                map(
                    AddDiscreteGaussianNoiseToNumber(sigma_squared=noise_scale),
                    [loc] * sample_size,
                )
            )
        )
    }


def _create_vector_discrete_gaussian_sampler(
    loc: int, noise_scale: ExactNumberInput, sample_size: int
):
    return lambda: {
        "noisy_vals": AddDiscreteGaussianNoiseToSeries(
            sigma_squared=noise_scale,
            input_domain=PandasSeriesDomain(NumpyIntegerDomain()),
        )(pd.Series([loc] * sample_size)).to_numpy()
    }


BASE_GEOMETRIC_TEST_INSTANCES = [
    {
        "sampler": staticmethod(sampler),
        "locations": {"noisy_vals": loc},
        "scales": {"noisy_vals": noise_scale},
        "cmfs": {"noisy_vals": _create_two_sided_geometric_cmf(loc)},
        "pmfs": {"noisy_vals": _create_two_sided_geometric_pmf(loc)},
    }
    for loc, noise_scale in [(3, 0.3), (111, 10.123)]
    for sampler in [
        _create_base_geometric_sampler(loc, noise_scale, SAMPLE_SIZE),
        _create_vector_geometric_sampler(loc, noise_scale, SAMPLE_SIZE),
    ]
]

BASE_DISCRETE_GAUSSIAN_TEST_INSTANCES = [
    {
        "sampler": staticmethod(sampler),
        "locations": {"noisy_vals": loc},
        "scales": {"noisy_vals": noise_scale},
        "cmfs": {"noisy_vals": _create_discrete_gaussian_cmf(loc)},
        "pmfs": {"noisy_vals": _create_discrete_gaussian_pmf(loc)},
    }
    for loc, noise_scale in [(3, 0.3), (111, 10.123)]
    for sampler in [
        _create_base_discrete_gaussian_sampler(loc, noise_scale, SAMPLE_SIZE),
        _create_vector_discrete_gaussian_sampler(loc, noise_scale, SAMPLE_SIZE),
    ]
]

# Noisy Aggregations test instances


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
    for iterations in [1, 200]:
        for count_loc, budget in zip(count_locations, privacy_budgets):
            dataset = FixedGroupDataSet(
                group_vals=list(range(count_loc)), num_groups=SAMPLE_SIZE // iterations
            )
            true_answers: Dict[str, Union[float, int]] = {
                "count": len(dataset.group_vals)
            }
            measurement = create_count_measurement(
                input_domain=dataset.domain,
                input_metric=SymmetricDifference(),
                output_measure=PureDP()
                if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
                else RhoZCDP(),
                d_out=budget,
                noise_mechanism=noise_mechanism,
                groupby_transformation=dataset.groupby(noise_mechanism),
                count_column="count",
            )
            sampler = _get_sampler(
                measurement,
                dataset,
                lambda df: df.select("count"),
                iterations=iterations,
            )
            noise_scales = _get_noise_scales(
                agg="count",
                budget=budget,
                dataset=dataset,
                noise_mechanism=noise_mechanism,
            )
            prob_functions = _get_prob_functions(noise_mechanism, true_answers)
            test_cases.append(
                {
                    "sampler": staticmethod(sampler),
                    "locations": true_answers,
                    "scales": noise_scales,
                    **prob_functions,
                }
            )
    return test_cases


def _get_count_distinct_test_cases(noise_mechanism: NoiseMechanism):
    """Returns count test cases.

    This returns a list of 4 test instances specifying the sampler (that produces
    a count sample), expected count location, expected noise scale, and corresponding
    cdf (if noise mechanism is Laplace) or cmf and pmf (if noise mechanism is not
    Laplace).

    Each of the 4 samplers produces a sample of size SAMPLE_SIZE.
    * 2 samplers that compute noisy groupby-count_distinct once on a DataFrame with
        # groups = SAMPLE_SIZE. These two samplers have different true counts
        and different noise scales.
    * 2 samplers that compute noisy groupby-count_distinct 200 times on a DataFrame
        with # groups = SAMPLE_SIZE/200. These two samplers have different true
        counts and different noise scales.
    """
    test_cases = []
    count_locations = [10, 45]
    privacy_budgets = [1, "0.4"]
    for iterations in [1, 200]:
        for count_loc, budget in zip(count_locations, privacy_budgets):
            dataset = FixedGroupDataSet(
                group_vals=list(range(count_loc)), num_groups=SAMPLE_SIZE // iterations
            )
            true_answers: Dict[str, Union[float, int]] = {
                "count": len(dataset.group_vals)
            }
            measurement = create_count_distinct_measurement(
                input_domain=dataset.domain,
                input_metric=SymmetricDifference(),
                output_measure=PureDP()
                if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
                else RhoZCDP(),
                d_out=budget,
                noise_mechanism=noise_mechanism,
                groupby_transformation=dataset.groupby(noise_mechanism),
                count_column="count",
            )
            sampler = _get_sampler(
                measurement,
                dataset,
                lambda df: df.select("count"),
                iterations=iterations,
            )
            noise_scales = _get_noise_scales(
                agg="count",
                budget=budget,
                dataset=dataset,
                noise_mechanism=noise_mechanism,
            )
            prob_functions = _get_prob_functions(noise_mechanism, true_answers)
            test_cases.append(
                {
                    "sampler": staticmethod(sampler),
                    "locations": true_answers,
                    "scales": noise_scales,
                    **prob_functions,
                }
            )
    return test_cases


def _get_sum_test_cases(noise_mechanism: NoiseMechanism):
    """Returns sum test cases.

    This returns a list of 4 test cases specifying the sampler (that produces
    a sum sample), expected sum location, expected noise scale and corresponding
    cdf (if noise mechanism is Laplace) or cmf and pmf (if noise mechanism is not
    Laplace).

    Each of the 4 samplers produces a sample of size SAMPLE_SIZE.
      * 2 samplers that compute noisy groupby-sum once on a DataFrame with
         # groups = SAMPLE_SIZE. These two samplers have different true sums
         and different noise scales.

      * 2 samplers that compute noisy groupby-sum 200 times on a DataFrame with
         # groups = SAMPLE_SIZE/200. These two samplers have different true sums
         and different noise scales.

    """
    test_cases = []
    sum_locations = (
        [3.5, 111.3] if noise_mechanism == NoiseMechanism.LAPLACE else [3, 111]
    )
    privacy_budgets = ["3.3", "0.11"]
    for iterations in [1, 200]:
        for sum_loc, budget in zip(sum_locations, privacy_budgets):
            group_values = _get_values_summing_to_loc(
                sum_loc, n=3
            )  # Fixed group size of 3
            dataset = FixedGroupDataSet(
                group_vals=group_values,
                num_groups=SAMPLE_SIZE // iterations,
                float_measure_column=noise_mechanism == NoiseMechanism.LAPLACE,
            )

            true_answers: Dict[str, Union[float, int]] = {
                "sum": sum(dataset.group_vals)
            }
            measurement = create_sum_measurement(
                input_domain=dataset.domain,
                input_metric=SymmetricDifference(),
                measure_column="B",
                output_measure=PureDP()
                if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
                else RhoZCDP(),
                lower=ExactNumber.from_float(min(group_values), round_up=False),
                upper=ExactNumber.from_float(max(group_values), round_up=True),
                noise_mechanism=noise_mechanism,
                d_out=budget,
                groupby_transformation=dataset.groupby(noise_mechanism),
                sum_column="sum",
            )
            sampler = _get_sampler(
                measurement, dataset, lambda df: df.select("sum"), iterations=iterations
            )
            noise_scales = _get_noise_scales(
                agg="sum",
                budget=budget,
                dataset=dataset,
                noise_mechanism=noise_mechanism,
            )
            prob_functions = _get_prob_functions(noise_mechanism, true_answers)
            test_cases.append(
                {
                    "sampler": staticmethod(sampler),
                    "locations": true_answers,
                    "scales": noise_scales,
                    **prob_functions,
                }
            )
    return test_cases


def _get_average_test_cases(noise_mechanism: NoiseMechanism) -> List[Dict]:
    """Returns average test cases.

    This returns a list of test instances specifying the sampler (that produces
    a count sample and a sum sample used to compute the average), expected locations
    for count and sum, expected noise scales and corresponding cdfs (if noise mechanism
    is Laplace) or cmfs and pmfs (if noise mechanism is not Laplace).
    """
    test_cases = []
    sum_locations: Union[List[float], List[int]]
    if noise_mechanism != NoiseMechanism.LAPLACE:
        sum_locations = [100, 14]  # Must be integers
    else:
        sum_locations = [99.78, 13.63]
    count_locations = [8, 5]
    privacy_budgets = ["0.8", "0.3"]
    for sum_loc, count_loc, budget in zip(
        sum_locations, count_locations, privacy_budgets
    ):
        group_values = _get_values_summing_to_loc(sum_loc, n=count_loc)
        dataset = FixedGroupDataSet(
            group_vals=group_values,
            num_groups=SAMPLE_SIZE,
            float_measure_column=noise_mechanism == NoiseMechanism.LAPLACE,
        )
        measurement = create_average_measurement(
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
        sampler = _get_sampler(measurement, dataset, postprocessor)
        noise_scales = _get_noise_scales(
            agg="average",
            budget=budget,
            dataset=dataset,
            noise_mechanism=noise_mechanism,
        )
        prob_functions = _get_prob_functions(noise_mechanism, true_answers)
        test_cases.append(
            {
                "sampler": staticmethod(sampler),
                "locations": true_answers,
                "scales": noise_scales,
                **prob_functions,
            }
        )
    return test_cases


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
        group_values = _get_values_summing_to_loc(sum_loc, n=count_loc)
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

        sampler = _get_sampler(measurement, dataset, postprocessor)
        noise_scales = _get_noise_scales(
            agg="standard deviation" if stddev else "variance",
            budget=budget,
            dataset=dataset,
            noise_mechanism=noise_mechanism,
        )
        prob_functions = _get_prob_functions(noise_mechanism, true_answers)
        test_cases.append(
            {
                "sampler": staticmethod(sampler),
                "locations": true_answers,
                "scales": noise_scales,
                **prob_functions,
            }
        )
    return test_cases


def _get_quantile_samples(
    quantile: float, lower: int, upper: int, epsilon: ExactNumberInput, data: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns samples using epsilon, low epsilon and high epsilon.

    Low and high epsilon are calculated using :data:`NOISE_SCALE_FUDGE_FACTOR`.
    """
    epsilon = ExactNumber(epsilon)

    def get_quantile_measurements(epsilons: List[ExactNumberInput]):
        """Returns a NoisyQuantile for each epsilon."""
        return [
            NoisyQuantile(
                input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
                output_measure=PureDP(),
                quantile=quantile,
                lower=lower,
                upper=upper,
                epsilon=sp.Rational(eps),
            )
            for eps in epsilons
        ]

    good_quantile, less_eps_quantile, more_eps_quantile = get_quantile_measurements(
        [
            epsilon,
            epsilon
            * ExactNumber.from_float((1 - NOISE_SCALE_FUDGE_FACTOR), round_up=True),
            epsilon
            * ExactNumber.from_float((1 + NOISE_SCALE_FUDGE_FACTOR), round_up=True),
        ]
    )

    good_samples = np.array([good_quantile(data) for _ in range(SAMPLE_SIZE)])
    less_eps_samples = np.array([less_eps_quantile(data) for _ in range(SAMPLE_SIZE)])
    more_eps_samples = np.array([more_eps_quantile(data) for _ in range(SAMPLE_SIZE)])

    return good_samples, less_eps_samples, more_eps_samples


class TestQuantileNoiseDistribution(PySparkTest):
    """Tests that NoisyQuantile has expected output distribution."""

    @parameterized.expand([(2, 0.5), ("4.5", 0.9), ("0.5", 0.25)])
    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime
    def test_quantile_noise(self, epsilon: ExactNumberInput, quantile: float):
        """Tests NoisyQuantile adds correct noise for given epsilon.

        This test samples given quantile of the dataset [2, 4, 6] with lo=0 and hi=8
        with 3 different epsilon values -> epsilon, epsilon * 0.9, epsilon * 1.1,
        samples are binned into 8 bins (each of length 1). Expected counts for each bin
        is computed by computing the probabilities and multiplying by SAMPLE_SIZE. For
        each of the 3 samples, p-values are obtained from a chi-square test.
        """
        epsilon = ExactNumber(epsilon)
        print(f"Testing: quantile={quantile} with epsilon={epsilon}")
        lo, hi = 0, 8
        test_list = [2, 4, 6]
        test_data = pd.Series(test_list)
        bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        probs = _get_quantile_probabilities(
            quantile=quantile,
            data=test_list,
            lower=lo,
            upper=hi,
            epsilon=epsilon.to_float(round_up=False),
        )
        assert np.allclose(sum(probs), 1)
        # Split default (data-based) bins to test each interval is uniformly sampled
        bin_probs = [p for prob in probs for p in [prob / 2, prob / 2]]
        expected_counts = [prob * SAMPLE_SIZE for prob in bin_probs]
        good_samples, less_eps_samples, more_eps_samples = _get_quantile_samples(
            quantile=quantile, lower=lo, upper=hi, epsilon=epsilon, data=test_data
        )
        _, good_p = chisquare(np.histogram(good_samples, bins=bins)[0], expected_counts)
        _, less_eps_counts = chisquare(
            np.histogram(less_eps_samples, bins=bins)[0], expected_counts
        )
        _, more_eps_counts = chisquare(
            np.histogram(more_eps_samples, bins=bins)[0], expected_counts
        )
        self.assertGreater(good_p, P_THRESHOLD)
        self.assertLess(less_eps_counts, P_THRESHOLD)
        self.assertLess(more_eps_counts, P_THRESHOLD)


def _run_ks_tests(
    sample: np.ndarray, conjectured_scale: float, cdf: Callable
) -> Tuple[float, float, float]:
    """Run KS test on the sample."""
    (_, good_p) = kstest(sample, cdf=lambda value: cdf(value, conjectured_scale))
    (_, less_noise_p) = kstest(
        sample,
        cdf=lambda value: cdf(
            value, conjectured_scale * (1 - NOISE_SCALE_FUDGE_FACTOR)
        ),
    )
    (_, more_noise_p) = kstest(
        sample,
        cdf=lambda value: cdf(
            value, conjectured_scale * (1 - NOISE_SCALE_FUDGE_FACTOR)
        ),
    )
    return (good_p, less_noise_p, more_noise_p)


def _run_chi_squared_tests(
    sample: np.ndarray, loc: int, conjectured_scale: float, cmf: Callable, pmf: Callable
) -> Tuple[float, float, float]:
    """Performs a Chi-squared test on the sample.

    Since chi2 tests don't work well for infinite bins, this test groups all values
    of k with an expected number of samples less than 5 into one of two bins:
    one bin is for small k values, and the other is for large k values.
    """
    sample_size = len(sample)
    # Find the minimum/maximum k values where the expected number of counts is > 5.
    max_k = loc
    while sample_size * pmf(max_k, conjectured_scale) >= 5:
        max_k += 1
    min_k = loc - (max_k - loc)

    # Calculate the actual and expected counts for all bins
    less_noise_noise_scale = conjectured_scale * (1 - NOISE_SCALE_FUDGE_FACTOR)
    more_noise_noise_scale = conjectured_scale * (1 + NOISE_SCALE_FUDGE_FACTOR)

    actual_counts = []
    good_expected_counts = []
    less_noise_expected_counts = []
    more_noise_expected_counts = []

    # Less than or equal to min_k
    actual_counts.append(np.sum(sample <= min_k))
    good_expected_counts.append(sample_size * cmf(min_k, conjectured_scale))
    less_noise_expected_counts.append(sample_size * cmf(min_k, less_noise_noise_scale))
    more_noise_expected_counts.append(sample_size * cmf(min_k, more_noise_noise_scale))

    # Each k between min_k, max_k (exclusive)
    for k in range(min_k + 1, max_k):
        actual_counts.append(np.sum(sample == k))
        good_expected_counts.append(sample_size * pmf(k, conjectured_scale))
        less_noise_expected_counts.append(sample_size * pmf(k, less_noise_noise_scale))
        more_noise_expected_counts.append(sample_size * pmf(k, more_noise_noise_scale))

    # Greater than or equal to max_k
    actual_counts.append(np.sum(sample >= max_k))
    good_expected_counts.append(sample_size * (1 - cmf(max_k - 1, conjectured_scale)))
    less_noise_expected_counts.append(
        sample_size * (1 - cmf(max_k - 1, less_noise_noise_scale))
    )
    more_noise_expected_counts.append(
        sample_size * (1 - cmf(max_k - 1, more_noise_noise_scale))
    )

    # Sanity check for actual/expected counts
    assert sum(actual_counts) == sample_size
    assert np.allclose(sum(good_expected_counts), sample_size)
    assert np.allclose(sum(less_noise_expected_counts), sample_size)
    assert np.allclose(sum(more_noise_expected_counts), sample_size)

    # Calculate and check p values
    (_, good_p) = chisquare(actual_counts, good_expected_counts)
    (_, less_noise_p) = chisquare(actual_counts, less_noise_expected_counts)
    (_, more_noise_p) = chisquare(actual_counts, more_noise_expected_counts)
    return good_p, less_noise_p, more_noise_p


class _ksTestCase:
    """A test case for TestUsingKSTest."""

    sampler: Callable[[], Dict[str, np.ndarray]]
    locations: Dict[str, Union[str, float]]
    scales: Dict[str, ExactNumberInput]
    cdfs: Dict[str, Callable]

    def __init__(self, sampler=None, locations=None, scales=None, cdfs=None):
        self.sampler = sampler
        self.locations = locations
        self.scales = scales
        self.cdfs = cdfs

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_ksTestCase":
        """Transform a dictionary into an _ksTestCase."""
        return cls(
            sampler=d["sampler"],
            locations=d["locations"],
            scales=d["scales"],
            cdfs=d["cdfs"],
        )


# This class used to be parameterized and it took more than 60 minutes
# for Python to generate all the tests - even when those tests weren't running!
# If you re-parameterize this class, tests will become MUCH slower.
# TODO(#1336): Unskip this KSTest.
@unittest.skip("Some combination of the test is causing OOM error")
@attr("slow")
class TestUsingKSTest(PySparkTest):
    """KS Tests for continuous noise mechanisms.

    These tests are parametrized by the following:
        - sampler : Function (with 0 parameters) that returns one or more samples as a
            dictionary from sample name (str) to a sample (numpy array).
        - locations : Expected locations for each sample.
        - scales : Expected noise scales for each sample.
        - cdf : Expected cdf for each sample.
    """

    cases: List[_ksTestCase]

    def setUp(self):
        """Setup test cases."""
        self.cases = [
            _ksTestCase.from_dict(e)
            for e in BASE_LAPLACE_TEST_INSTANCES
            + _get_count_test_cases(NoiseMechanism.LAPLACE)
            + _get_count_distinct_test_cases(NoiseMechanism.LAPLACE)
            + _get_sum_test_cases(NoiseMechanism.LAPLACE)
            + _get_average_test_cases(NoiseMechanism.LAPLACE)
            + _get_var_stddev_test_cases(NoiseMechanism.LAPLACE, stddev=False)
            + _get_var_stddev_test_cases(NoiseMechanism.LAPLACE, stddev=True)
        ]

    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime
    def test_using_ks_test(self):
        """Performs a KS test."""
        for case in self.cases:
            samples = case.sampler()
            for sample_name, sample in samples.items():
                good_p, less_noise_p, more_noise_p = _run_ks_tests(
                    sample=sample,
                    conjectured_scale=case.scales[sample_name],
                    cdf=case.cdfs[sample_name],
                )
                self.assertGreater(good_p, P_THRESHOLD)
                self.assertLess(less_noise_p, P_THRESHOLD)
                self.assertLess(more_noise_p, P_THRESHOLD)


class _chiSquaredTestCase:
    """Class representing a single test case for ChiSquaredTest."""

    sampler: Callable[[], Dict[str, np.ndarray]]
    locations: Dict[str, int]
    scales: Dict[str, ExactNumberInput]
    cmfs: Dict[str, Callable]
    pmfs: Dict[str, Callable]

    def __init__(self, sampler=None, locations=None, scales=None, cmfs=None, pmfs=None):
        self.sampler = sampler
        self.locations = locations
        self.scales = scales
        self.cmfs = cmfs
        self.pmfs = pmfs

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_chiSquaredTestCase":
        """Turns a dictionary into a _chiSquaredTestCase."""
        return cls(
            sampler=d["sampler"],
            locations=d["locations"],
            scales=d["scales"],
            cmfs=d["cmfs"],
            pmfs=d["pmfs"],
        )


# TODO(#1336): Unskip this ChiSquaredTest.
# As previously - do not re-parameterize this test.
@unittest.skip("Some combination of the test is causing OOM error")
@attr("slow")
class TestUsingChiSquaredTest(PySparkTest):
    """Chi-Squared Tests for discrete noise mechanisms.

    Each test is parametrized by the following:
        - sampler : Function (with 0 parameters) that returns one or more samples as a
            dictionary from sample name (str) to a sample (numpy array).
        - locations : Expected locations for each sample.
        - scales : Expected noise scales for each sample.
        - cmf : Expected cmf for each sample.
        - pmf : Expected pmf for each sample.
    """

    cases: List[_chiSquaredTestCase]

    def setUp(self):
        """Generate test cases."""
        self.cases = [
            _chiSquaredTestCase.from_dict(e)
            for e in BASE_GEOMETRIC_TEST_INSTANCES
            + BASE_DISCRETE_GAUSSIAN_TEST_INSTANCES
            + _get_count_test_cases(NoiseMechanism.GEOMETRIC)
            + _get_count_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN)
            + _get_count_distinct_test_cases(NoiseMechanism.GEOMETRIC)
            + _get_count_distinct_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN)
            + _get_sum_test_cases(NoiseMechanism.GEOMETRIC)
            + _get_sum_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN)
            + _get_average_test_cases(NoiseMechanism.GEOMETRIC)
            + _get_average_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN)
            + _get_var_stddev_test_cases(NoiseMechanism.GEOMETRIC, stddev=False)
            + _get_var_stddev_test_cases(NoiseMechanism.GEOMETRIC, stddev=True)
            + _get_var_stddev_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN, stddev=False)
            + _get_var_stddev_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN, stddev=True)
        ]

    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime
    def test_using_chi_squared_test(self):
        """Performs a chi-squared test."""
        for case in self.cases:

            samples = case.sampler()
            for sample_name, sample in samples.items():
                good_p, less_noise_p, more_noise_p = _run_chi_squared_tests(
                    sample=sample,
                    loc=case.locations[sample_name],
                    conjectured_scale=case.scales[sample_name],
                    cmf=case.cmfs[sample_name],
                    pmf=case.pmfs[sample_name],
                )
                self.assertGreater(good_p, P_THRESHOLD)
                self.assertLess(less_noise_p, P_THRESHOLD)
                self.assertLess(more_noise_p, P_THRESHOLD)


class TestCorrelationDetection(PySparkTest):
    """Tests that samples with duplicates are rejected.

    These tests verify that statistical tests do fail when samples drawn are
    correlated. In particular, these tests verify that if sampled noise recurs 200
    times across a sample of size SAMPLE_SIZE then the samples are rejected.
    """

    @parameterized.expand([(0.2,), (0.8,), (1.2,), (2.2,), (4.2,), (10.2,)])
    def test_ks_test_rejects_samples_with_duplicates(self, scale: float):
        """Tests that KS test rejects sample with duplicates."""
        noise = np.random.laplace(np.zeros(SAMPLE_SIZE // 200), scale=scale)
        replicated_noise = np.concatenate([noise for _ in range(200)])
        cdf = _create_laplace_cdf(0)
        (_, p_value) = kstest(replicated_noise, cdf=lambda value: cdf(value, scale))
        self.assertLess(p_value, P_THRESHOLD)

    # Note: This test doesn't use 0.2, as small noise scales make the test flaky
    @parameterized.expand([(0.8,), (1.2,), (2.2,), (4.2,), (10.2,)])
    def test_chi_squared_rejects_samples_with_duplicates(self, scale: float):
        """Tests that Chi-squared test rejects samples with duplicates."""
        p = 1 - np.exp(-1 / scale)
        noise = np.random.geometric(p, size=SAMPLE_SIZE // 200) - np.random.geometric(
            p, size=SAMPLE_SIZE // 200
        )
        sample = np.concatenate([noise for _ in range(200)])
        (p_value, _, _) = _run_chi_squared_tests(
            sample,
            loc=0,
            conjectured_scale=scale,
            cmf=_create_two_sided_geometric_cmf(0),
            pmf=_create_two_sided_geometric_pmf(0),
        )
        self.assertLess(p_value, P_THRESHOLD)
