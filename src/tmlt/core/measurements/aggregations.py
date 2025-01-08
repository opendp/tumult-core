"""Derived measurements for computing noisy aggregates on spark DataFrames."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from enum import Enum
from math import ceil, log2
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType, StructField, StructType
from typeguard import typechecked

from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    convert_spark_schema,
)
from tmlt.core.exceptions import (
    DomainMismatchError,
    MetricMismatchError,
    UnsupportedCombinationError,
    UnsupportedDomainError,
    UnsupportedMeasureError,
    UnsupportedMetricError,
    UnsupportedNoiseMechanismError,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.composition import Composition
from tmlt.core.measurements.converters import PureDPToApproxDP, PureDPToRhoZCDP
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.dataframe import AggregateByColumn
from tmlt.core.measurements.pandas_measurements.series import (
    AddNoiseToSeries,
    NoisyQuantile,
)
from tmlt.core.measurements.postprocess import PostProcess
from tmlt.core.measurements.spark_measurements import (
    AddNoiseToColumn,
    ApplyInPandas,
    GeometricPartitionSelection,
    SparseVectorPrefixSums,
)
from tmlt.core.measures import (
    ApproxDP,
    ApproxDPBudget,
    PrivacyBudget,
    PrivacyBudgetInput,
    PureDP,
    RhoZCDP,
    RhoZCDPBudget,
)
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.converters import UnwrapIfGroupedBy
from tmlt.core.transformations.identity import Identity
from tmlt.core.transformations.spark_transformations.agg import (
    CountGrouped,
    create_count_aggregation,
    create_count_distinct_aggregation,
    create_sum_aggregation,
)
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.map import (
    Map,
    RowToRowTransformation,
)
from tmlt.core.utils.distributions import double_sided_geometric_inverse_cmf_exact
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.join import join
from tmlt.core.utils.misc import get_nonconflicting_string
from tmlt.core.utils.parameters import calculate_noise_scale


class NoiseMechanism(Enum):
    """Enumerating noise mechanisms."""

    LAPLACE = 1
    GEOMETRIC = 2
    DISCRETE_GAUSSIAN = 3
    GAUSSIAN = 4

    def check_output_measure(self, output_measure: Union[PureDP, RhoZCDP]) -> None:
        """Checks if the specified output measure is supported."""
        if output_measure not in self.supported_output_measure():
            str_output_measures = ", ".join(
                [repr(measure) for measure in self.supported_output_measure()]
            )
            raise UnsupportedMeasureError(
                output_measure,
                (
                    f"Output measure {output_measure}"
                    f" is not supported by noise mechanism {self}."
                    f" Supported output measures are {str_output_measures}."
                ),
            )

    def supported_output_measure(self) -> List[Union[PureDP, RhoZCDP]]:
        """Returns a list of output measures supported by this noise mechanism."""
        if self == NoiseMechanism.LAPLACE:
            return [PureDP(), RhoZCDP()]
        if self == NoiseMechanism.GEOMETRIC:
            return [PureDP(), RhoZCDP()]
        if self == NoiseMechanism.DISCRETE_GAUSSIAN:
            return [RhoZCDP()]
        if self == NoiseMechanism.GAUSSIAN:
            return [RhoZCDP()]
        raise UnsupportedNoiseMechanismError(self, f"Unknown noise mechanism {self}")

    def __str__(self) -> str:
        """Returns a string representation of the noise mechanism."""
        return self.name.lower().capitalize().replace("_", " ")


@typechecked
def create_count_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    d_out: PrivacyBudgetInput,
    noise_mechanism: NoiseMechanism,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    count_column: Optional[str] = None,
) -> Measurement:
    """Returns a noisy count measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are ``d_in``-close under the ``input_metric``,
    M(x) and M(x') are sampled from distributions that are ``d_out`` apart under the
    ``output_measure``. Noise scale is computed appropriately for the specified
    ``noise_mechanism`` such that the stated privacy property is guaranteed.

    Note:
        ``d_out`` is interpreted as the "epsilon" parameter if ``output_measure`` is
        :class:`~.PureDP`, the "rho" parameter if ``output_measure`` is
        :class:`~.RhoZCDP`, and ("epsilon", "delta") if ``output_measure`` is
        :class:`~.ApproxDP`.

    Note:
        :class:`~.ApproxDP` budgets with delta>0 are not yet supported.


    Args:
        input_domain: Domain of input spark DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP`,
            :class:`~.RhoZCDP`, or :class:`~.ApproxDP`).
        d_out: Desired distance between output distributions w.r.t. ``d_in``. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP`, "rho" if it
            is :class:`~.RhoZCDP`, and ("epsilon", "delta") if it is
            :class:`~.ApproxDP`.
        noise_mechanism: Noise mechanism to apply to count(s).
        d_in: Distance between inputs under the ``input_metric``. The returned
            measurement is guaranteed to have output distributions that are ``d_out``
            apart for inputs that are ``d_in`` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy counts for each group obtained by applying the groupby transformation
            . Otherwise, this measurement outputs a single number - the noisy count.
        count_column: If a ``groupby_transformation`` is provided, this is the column
            name to be used for counts in the dataframe output by the measurement. If
            None, this column will be named "count".
    """
    if isinstance(output_measure, ApproxDP):
        epsilon, delta = ApproxDPBudget(d_out).value
        if noise_mechanism in (NoiseMechanism.LAPLACE, NoiseMechanism.GEOMETRIC):
            if delta > 0:
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Cannot spend an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism}. Use ApproxDP with delta = 0 or PureDP."
                    ),
                )
            return PureDPToApproxDP(
                create_count_measurement(
                    input_domain=input_domain,
                    input_metric=input_metric,
                    output_measure=PureDP(),
                    d_out=epsilon,
                    noise_mechanism=noise_mechanism,
                    d_in=d_in,
                    groupby_transformation=groupby_transformation,
                    count_column=count_column,
                )
            )
        elif noise_mechanism in (
            NoiseMechanism.GAUSSIAN,
            NoiseMechanism.DISCRETE_GAUSSIAN,
        ):
            if delta > 0:
                # Once supported, we will compute the corresponding zCDP budget and set
                # the ouptut measure to zCDP.
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Spending an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism} is not yet supported. Use either"
                        f" {NoiseMechanism.LAPLACE} or {NoiseMechanism.GEOMETRIC}."
                    ),
                )
            raise UnsupportedCombinationError(
                (noise_mechanism, output_measure, d_out),
                (
                    f"Cannot spend a budget with delta = 0 using {noise_mechanism}. Set"
                    f" delta > 0 or use either {NoiseMechanism.LAPLACE} or"
                    f" {NoiseMechanism.GEOMETRIC}."
                ),
            )
        else:
            assert False
    elif isinstance(output_measure, (RhoZCDP, PureDP)):
        d_out = PrivacyBudget.cast(output_measure, d_out).value
    else:
        assert False
    d_in = ExactNumber(d_in)
    # help mypy
    assert isinstance(output_measure, (PureDP, RhoZCDP))
    noise_mechanism.check_output_measure(output_measure)
    count_aggregation: Transformation
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise TypeError(
                input_metric,
                (
                    "Cannot use IfGroupedBy input metric if no groupby_transformation"
                    " is provided"
                ),
            )
        count_aggregation = create_count_aggregation(
            input_domain=input_domain,
            input_metric=input_metric,
            count_column=count_column,
        )
        d_mid = count_aggregation.stability_function(d_in)
        noise_scale = calculate_noise_scale(
            d_in=d_mid, d_out=d_out, output_measure=output_measure
        )
        add_noise_to_number: Measurement
        if noise_mechanism == NoiseMechanism.LAPLACE:
            add_noise_to_number = AddLaplaceNoise(
                scale=noise_scale, input_domain=NumpyIntegerDomain()
            )
        elif noise_mechanism == NoiseMechanism.GEOMETRIC:
            add_noise_to_number = AddGeometricNoise(alpha=noise_scale)
        elif noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
            add_noise_to_number = AddDiscreteGaussianNoise(
                sigma_squared=noise_scale**2
            )
        elif noise_mechanism == NoiseMechanism.GAUSSIAN:
            add_noise_to_number = AddGaussianNoise(
                sigma_squared=noise_scale**2, input_domain=NumpyIntegerDomain()
            )
        else:
            raise UnsupportedNoiseMechanismError(
                noise_mechanism,
                (
                    f"Unrecognized noise mechanism {noise_mechanism}. "
                    "Supported noise mechanisms are LAPLACE, "
                    "GEOMETRIC, GAUSSIAN, and DISCRETE_GAUSSIAN."
                ),
            )
        count_measurement: Measurement = count_aggregation | add_noise_to_number
        if (
            output_measure == RhoZCDP()
            and PureDP() in noise_mechanism.supported_output_measure()
        ):
            # count_measurement has output_measure PureDP and needs to be wrapped in a
            # converter.
            count_measurement = PureDPToRhoZCDP(count_measurement)
        assert count_measurement.privacy_function(d_in) == d_out
        return count_measurement
    assert isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared))
    assert isinstance(groupby_transformation.output_domain, SparkGroupedDataFrameDomain)

    if isinstance(input_metric, IfGroupedBy):
        if input_metric.column not in groupby_transformation.group_keys.columns:
            raise ValueError(
                "The input_metric column must be in the groupby_transformation "
                f"group_keys columns. Got input_metric column = {input_metric.column},"
                f" group_keys columns = {groupby_transformation.group_keys.columns}",
            )

    count_aggregation = create_count_aggregation(
        input_domain=groupby_transformation.output_domain,
        input_metric=groupby_transformation.output_metric,
        count_column=count_column,
    )
    groupby_count = groupby_transformation | count_aggregation
    d_mid = groupby_count.stability_function(d_in)
    noise_scale = calculate_noise_scale(
        d_in=d_mid, d_out=d_out, output_measure=output_measure
    )
    add_noise_to_series: AddNoiseToSeries
    if noise_mechanism == NoiseMechanism.LAPLACE:
        add_noise_to_series = AddNoiseToSeries(
            AddLaplaceNoise(scale=noise_scale, input_domain=NumpyIntegerDomain())
        )
    elif noise_mechanism == NoiseMechanism.GEOMETRIC:
        add_noise_to_series = AddNoiseToSeries(AddGeometricNoise(alpha=noise_scale))
    elif noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
        add_noise_to_series = AddNoiseToSeries(
            AddDiscreteGaussianNoise(sigma_squared=noise_scale**2)
        )
    elif noise_mechanism == NoiseMechanism.GAUSSIAN:
        add_noise_to_series = AddNoiseToSeries(
            AddGaussianNoise(
                sigma_squared=noise_scale**2, input_domain=NumpyIntegerDomain()
            )
        )

    else:
        raise UnsupportedNoiseMechanismError(
            noise_mechanism,
            (
                f"Unrecognized noise mechanism {noise_mechanism}. "
                "Supported noise mechanisms are LAPLACE, "
                "GEOMETRIC, GAUSSIAN, and DISCRETE_GAUSSIAN."
            ),
        )

    assert isinstance(groupby_count.output_domain, SparkDataFrameDomain)
    add_noise_to_column = AddNoiseToColumn(
        input_domain=groupby_count.output_domain,
        measure_column=count_aggregation.count_column,
        measurement=add_noise_to_series,
    )
    count_measurement = groupby_count | add_noise_to_column
    if (
        output_measure == RhoZCDP()
        and PureDP() in noise_mechanism.supported_output_measure()
    ):
        # count_measurement has output_measure PureDP and needs to be wrapped in a
        # converter.
        count_measurement = PureDPToRhoZCDP(count_measurement)
    assert count_measurement.privacy_function(d_in) == d_out
    return count_measurement


@typechecked
def create_count_distinct_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    d_out: PrivacyBudgetInput,
    noise_mechanism: NoiseMechanism,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    count_column: Optional[str] = None,
) -> Measurement:
    """Returns a noisy count_distinct measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are ``d_in``-close under the ``input_metric``,
    M(x) and M(x') are sampled from distributions that are ``d_out`` apart
    under the ``output_measure``. Noise scale is computed appropriately for the
    specified ``noise_mechanism`` such that the stated privacy property
    is guaranteed.

    Note:
        ``d_out`` is interpreted as the "epsilon" parameter if ``output_measure`` is
        :class:`~.PureDP`, the "rho" parameter if ``output_measure`` is
        :class:`~.RhoZCDP`, and ("epsilon", "delta") if ``output_measure`` is
        :class:`~.ApproxDP`.

    Note:
        :class:`~.ApproxDP` budgets with delta>0 are not yet supported.

    Args:
        input_domain: Domain of input spark DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP`,
            :class:`~.RhoZCDP`, or :class:`~.ApproxDP`).
        d_out: Desired distance between output distributions w.r.t. ``d_in``. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP`, "rho" if it
            is :class:`~.RhoZCDP`, and ("epsilon", "delta") if it is
            :class:`~.ApproxDP`.
        noise_mechanism: Noise mechanism to apply to count(s).
        d_in: Distance between inputs under the ``input_metric``. The returned
            measurement is guaranteed to have output distributions that are
            ``d_out`` apart for inputs that are ``d_in`` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame
            with noisy counts for each group obtained by applying the groupby
            transformation. Otherwise, this measurement outputs a single number -
            the noisy count of distinct items.
        count_column: If a ``groupby_transformation`` is provided, this is the
            column name to be used for counts in the dataframe output by the
            measurement. If None, this column will be named "count".
    """
    if isinstance(output_measure, ApproxDP):
        epsilon, delta = ApproxDPBudget(d_out).value
        if noise_mechanism in (NoiseMechanism.LAPLACE, NoiseMechanism.GEOMETRIC):
            if delta > 0:
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Cannot spend an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism}. Use ApproxDP with delta = 0 or PureDP."
                    ),
                )
            return PureDPToApproxDP(
                create_count_distinct_measurement(
                    input_domain=input_domain,
                    input_metric=input_metric,
                    output_measure=PureDP(),
                    d_out=epsilon,
                    noise_mechanism=noise_mechanism,
                    d_in=d_in,
                    groupby_transformation=groupby_transformation,
                    count_column=count_column,
                )
            )
        elif noise_mechanism in (
            NoiseMechanism.GAUSSIAN,
            NoiseMechanism.DISCRETE_GAUSSIAN,
        ):
            if delta > 0:
                # Once supported, we will compute the corresponding zCDP budget and set
                # the ouptut measure to zCDP.
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Spending an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism} is not yet supported. Use either"
                        f" {NoiseMechanism.LAPLACE} or {NoiseMechanism.GEOMETRIC}."
                    ),
                )
            raise UnsupportedCombinationError(
                (noise_mechanism, output_measure, d_out),
                (
                    f"Cannot spend a budget with delta = 0 using {noise_mechanism}. Set"
                    f" delta > 0 or use either {NoiseMechanism.LAPLACE} or"
                    f" {NoiseMechanism.GEOMETRIC}."
                ),
            )
        else:
            assert False
    elif isinstance(output_measure, (RhoZCDP, PureDP)):
        d_out = PrivacyBudget.cast(output_measure, d_out).value
    else:
        assert False
    d_in = ExactNumber(d_in)
    # help mypy
    assert isinstance(output_measure, (PureDP, RhoZCDP))
    noise_mechanism.check_output_measure(output_measure)
    count_distinct_aggregation: Transformation
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise UnsupportedMetricError(
                input_metric,
                (
                    "Cannot use IfGroupedBy input metric if no"
                    "groupby_transformation is provided."
                ),
            )
        count_distinct_aggregation = create_count_distinct_aggregation(
            input_domain=input_domain,
            input_metric=input_metric,
            count_column=count_column,
        )
        d_mid = count_distinct_aggregation.stability_function(d_in)
        noise_scale = calculate_noise_scale(
            d_in=d_mid, d_out=d_out, output_measure=output_measure
        )
        add_noise_to_number: Measurement
        if noise_mechanism == NoiseMechanism.LAPLACE:
            add_noise_to_number = AddLaplaceNoise(
                scale=noise_scale, input_domain=NumpyIntegerDomain()
            )
        elif noise_mechanism == NoiseMechanism.GEOMETRIC:
            add_noise_to_number = AddGeometricNoise(alpha=noise_scale)
        elif noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
            add_noise_to_number = AddDiscreteGaussianNoise(
                sigma_squared=noise_scale**2
            )
        elif noise_mechanism == NoiseMechanism.GAUSSIAN:
            add_noise_to_number = AddGaussianNoise(
                sigma_squared=noise_scale**2, input_domain=NumpyIntegerDomain()
            )
        else:
            raise UnsupportedNoiseMechanismError(
                noise_mechanism,
                (
                    f"Unrecognized noise mechanism {noise_mechanism}. "
                    "Supported noise mechanisms are LAPLACE, "
                    "GEOMETRIC, GAUSSIAN, and DISCRETE_GAUSSIAN."
                ),
            )

        count_distinct_measurement: Measurement = (
            count_distinct_aggregation | add_noise_to_number
        )
        if (
            output_measure == RhoZCDP()
            and PureDP() in noise_mechanism.supported_output_measure()
        ):
            # the measurement created above has output_measure PureDP,
            # so it needs to be converted
            count_distinct_measurement = PureDPToRhoZCDP(count_distinct_measurement)
        assert count_distinct_measurement.privacy_function(d_in) == d_out
        return count_distinct_measurement
    if not isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared)):
        raise UnsupportedMetricError(
            groupby_transformation.output_metric,
            (
                "A groupby_transformation for count_distinct_measurement must have an "
                "output metric of either SumOf or RootSumOfSquared."
            ),
        )
    if not isinstance(
        groupby_transformation.output_domain, SparkGroupedDataFrameDomain
    ):
        raise UnsupportedDomainError(
            groupby_transformation.output_domain,
            (
                "A groupby_transformation for count_distinct_measurement must have an "
                "output domain of SparkGroupedDataFrameDomain."
            ),
        )
    count_distinct_aggregation = create_count_distinct_aggregation(
        input_domain=groupby_transformation.output_domain,
        input_metric=groupby_transformation.output_metric,
        count_column=count_column,
    )
    groupby_count_distinct = groupby_transformation | count_distinct_aggregation
    d_mid = groupby_count_distinct.stability_function(d_in)
    noise_scale = calculate_noise_scale(
        d_in=d_mid, d_out=d_out, output_measure=output_measure
    )
    add_noise_to_series: AddNoiseToSeries
    if noise_mechanism == NoiseMechanism.LAPLACE:
        add_noise_to_series = AddNoiseToSeries(
            AddLaplaceNoise(scale=noise_scale, input_domain=NumpyIntegerDomain())
        )
    elif noise_mechanism == NoiseMechanism.GEOMETRIC:
        add_noise_to_series = AddNoiseToSeries(AddGeometricNoise(alpha=noise_scale))
    elif noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
        add_noise_to_series = AddNoiseToSeries(
            AddDiscreteGaussianNoise(sigma_squared=noise_scale**2)
        )
    elif noise_mechanism == NoiseMechanism.GAUSSIAN:
        add_noise_to_series = AddNoiseToSeries(
            AddGaussianNoise(
                sigma_squared=noise_scale**2, input_domain=NumpyIntegerDomain()
            )
        )
    else:
        raise UnsupportedNoiseMechanismError(
            noise_mechanism,
            (
                f"Unrecognized noise mechanism {noise_mechanism}. "
                "Supported noise mechanisms are LAPLACE, "
                "GEOMETRIC, GAUSSIAN, and DISCRETE_GAUSSIAN."
            ),
        )
    assert isinstance(groupby_count_distinct.output_domain, SparkDataFrameDomain)
    add_noise_to_column = AddNoiseToColumn(
        input_domain=groupby_count_distinct.output_domain,
        measure_column=count_distinct_aggregation.count_column,
        measurement=add_noise_to_series,
    )
    count_distinct_measurement = groupby_count_distinct | add_noise_to_column
    if (
        output_measure == RhoZCDP()
        and PureDP() in noise_mechanism.supported_output_measure()
    ):
        # the count_distinct_measurement generated above has the
        # output_measure PureDP, and needs to be converted
        count_distinct_measurement = PureDPToRhoZCDP(count_distinct_measurement)
    assert count_distinct_measurement.privacy_function(d_in) == d_out
    return count_distinct_measurement


@typechecked
def create_sum_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    d_out: PrivacyBudgetInput,
    noise_mechanism: NoiseMechanism,
    measure_column: str,
    lower: ExactNumberInput,
    upper: ExactNumberInput,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    sum_column: Optional[str] = None,
) -> Measurement:
    """Returns a noisy sum measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are ``d_in``-close under the ``input_metric``,
    M(x) and M(x') are sampled from distributions that are ``d_out`` apart under the
    ``output_measure``. Noise scale is computed appropriately for the specified
    ``noise_mechanism`` such that the stated privacy property is guaranteed.

    Note:
        ``d_out`` is interpreted as the "epsilon" parameter if ``output_measure`` is
        :class:`~.PureDP`, the "rho" parameter if ``output_measure`` is
        :class:`~.RhoZCDP`, and ("epsilon", "delta") if ``output_measure`` is
        :class:`~.ApproxDP`.

    Note:
        :class:`~.ApproxDP` budgets with delta>0 are not yet supported.

    Args:
        input_domain: Domain of input spark DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP`,
            :class:`~.RhoZCDP`, or :class:`~.ApproxDP`).
        d_out: Desired distance between output distributions w.r.t. ``d_in``. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP`, "rho" if it
            is :class:`~.RhoZCDP`, and ("epsilon", "delta") if it is
            :class:`~.ApproxDP`.
        noise_mechanism: Noise mechanism to be applied to the sum(s).
        measure_column: Column to be summed.
        lower: Lower clipping bound on ``measure_column``.
        upper: Upper clipping bound on ``measure_column``.
        d_in: Distance between inputs under the ``input_metric``. The returned
            measurement is guaranteed to have output distributions that are ``d_out``
            apart for inputs that are ``d_in`` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy sums for each group obtained by applying the groupby transformation.
            If None, this measurement outputs a single number - the noisy sum.
        sum_column: If a ``groupby_transformation`` is supplied, this is the column
            name to be used for sums in the DataFrame output by the measurement. If
            None, this column will be named "sum(<measure_column>)".
    """
    if isinstance(output_measure, ApproxDP):
        epsilon, delta = ApproxDPBudget(d_out).value
        if noise_mechanism in (NoiseMechanism.LAPLACE, NoiseMechanism.GEOMETRIC):
            if delta > 0:
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Cannot spend an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism}. Use ApproxDP with delta = 0 or PureDP."
                    ),
                )
            return PureDPToApproxDP(
                create_sum_measurement(
                    input_domain=input_domain,
                    input_metric=input_metric,
                    output_measure=PureDP(),
                    d_out=epsilon,
                    noise_mechanism=noise_mechanism,
                    measure_column=measure_column,
                    lower=lower,
                    upper=upper,
                    d_in=d_in,
                    groupby_transformation=groupby_transformation,
                    sum_column=sum_column,
                )
            )
        elif noise_mechanism in (
            NoiseMechanism.GAUSSIAN,
            NoiseMechanism.DISCRETE_GAUSSIAN,
        ):
            if delta > 0:
                # Once supported, we will compute the corresponding zCDP budget and set
                # the ouptut measure to zCDP.
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Spending an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism} is not yet supported. Use either"
                        f" {NoiseMechanism.LAPLACE} or {NoiseMechanism.GEOMETRIC}."
                    ),
                )
            raise UnsupportedCombinationError(
                (noise_mechanism, output_measure, d_out),
                (
                    f"Cannot spend a budget with delta = 0 using {noise_mechanism}. Set"
                    f" delta > 0 or use either {NoiseMechanism.LAPLACE} or"
                    f" {NoiseMechanism.GEOMETRIC}."
                ),
            )
        else:
            assert False
    elif isinstance(output_measure, (RhoZCDP, PureDP)):
        d_out = PrivacyBudget.cast(output_measure, d_out).value
    else:
        assert False
    d_in = ExactNumber(d_in)
    lower = ExactNumber(lower)
    upper = ExactNumber(upper)
    # help mypy
    assert isinstance(output_measure, (PureDP, RhoZCDP))
    noise_mechanism.check_output_measure(output_measure)
    sum_aggregation: Transformation
    measure_column_domain = input_domain[measure_column].to_numpy_domain()
    if not isinstance(measure_column_domain, (NumpyIntegerDomain, NumpyFloatDomain)):
        raise ValueError(f"Measure column must be numeric, not {measure_column_domain}")
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise UnsupportedMetricError(
                input_metric,
                (
                    "IfGroupedBy must be accompanied by an appropriate groupby "
                    "transformation."
                ),
            )
        sum_aggregation = create_sum_aggregation(
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column=measure_column,
            upper=upper,
            lower=lower,
            sum_column=sum_column,
        )
        d_mid = sum_aggregation.stability_function(d_in)
        noise_scale = calculate_noise_scale(
            d_in=d_mid, d_out=d_out, output_measure=output_measure
        )
        add_noise_to_number: Measurement
        if noise_mechanism == NoiseMechanism.LAPLACE:
            add_noise_to_number = AddLaplaceNoise(
                scale=noise_scale, input_domain=measure_column_domain
            )
        elif noise_mechanism == NoiseMechanism.GEOMETRIC:
            add_noise_to_number = AddGeometricNoise(alpha=noise_scale)

        elif noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
            add_noise_to_number = AddDiscreteGaussianNoise(
                sigma_squared=noise_scale**2
            )
        elif noise_mechanism == NoiseMechanism.GAUSSIAN:
            add_noise_to_number = AddGaussianNoise(
                sigma_squared=noise_scale**2, input_domain=measure_column_domain
            )
        else:
            raise UnsupportedNoiseMechanismError(
                noise_mechanism,
                (
                    f"Unrecognized noise mechanism {noise_mechanism}. "
                    "Supported noise mechanisms are LAPLACE, "
                    "GEOMETRIC, GAUSSIAN, and DISCRETE_GAUSSIAN."
                ),
            )
        sum_measurement: Measurement = sum_aggregation | add_noise_to_number
        if (
            output_measure == RhoZCDP()
            and PureDP() in noise_mechanism.supported_output_measure()
        ):
            # sum_measurement has output_measure PureDP and needs to be wrapped in a
            # converter.
            sum_measurement = PureDPToRhoZCDP(sum_measurement)
        assert sum_measurement.privacy_function(d_in) == d_out
        return sum_measurement
    add_noise_to_series: AddNoiseToSeries
    assert isinstance(groupby_transformation.output_domain, SparkGroupedDataFrameDomain)
    assert isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared))
    sum_aggregation = create_sum_aggregation(
        input_domain=groupby_transformation.output_domain,
        input_metric=groupby_transformation.output_metric,
        measure_column=measure_column,
        lower=lower,
        upper=upper,
        sum_column=sum_column,
    )
    groupby_sum = groupby_transformation | sum_aggregation
    d_mid = groupby_sum.stability_function(d_in)
    noise_scale = calculate_noise_scale(
        d_in=d_mid, d_out=d_out, output_measure=output_measure
    )
    if noise_mechanism == NoiseMechanism.LAPLACE:
        add_noise_to_series = AddNoiseToSeries(
            AddLaplaceNoise(scale=noise_scale, input_domain=measure_column_domain)
        )
    elif noise_mechanism == NoiseMechanism.GEOMETRIC:
        add_noise_to_series = AddNoiseToSeries(AddGeometricNoise(alpha=noise_scale))
    elif noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
        add_noise_to_series = AddNoiseToSeries(
            AddDiscreteGaussianNoise(sigma_squared=noise_scale**2)
        )
    elif noise_mechanism == NoiseMechanism.GAUSSIAN:
        add_noise_to_series = AddNoiseToSeries(
            AddGaussianNoise(
                sigma_squared=noise_scale**2, input_domain=measure_column_domain
            )
        )
    else:
        raise UnsupportedNoiseMechanismError(
            noise_mechanism,
            (
                f"Unrecognized noise mechanism {noise_mechanism}. "
                "Supported noise mechanisms are LAPLACE, "
                "GEOMETRIC, GAUSSIAN, and DISCRETE_GAUSSIAN."
            ),
        )
    assert isinstance(sum_aggregation.output_domain, SparkDataFrameDomain)
    add_noise_to_column = AddNoiseToColumn(
        input_domain=sum_aggregation.output_domain,
        measure_column=sum_aggregation.sum_column,
        measurement=add_noise_to_series,
    )
    sum_measurement = groupby_sum | add_noise_to_column
    if (
        output_measure == RhoZCDP()
        and PureDP() in noise_mechanism.supported_output_measure()
    ):
        # sum_measurement has output_measure PureDP and needs to be wrapped in a
        # converter.
        sum_measurement = PureDPToRhoZCDP(sum_measurement)
    assert sum_measurement.privacy_function(d_in) == d_out
    return sum_measurement


@typechecked
def create_average_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    d_out: PrivacyBudgetInput,
    noise_mechanism: NoiseMechanism,
    measure_column: str,
    lower: ExactNumberInput,
    upper: ExactNumberInput,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    average_column: Optional[str] = None,
    keep_intermediates: bool = False,
    sum_column: Optional[str] = None,
    count_column: Optional[str] = None,
) -> Union[PostProcess, PureDPToApproxDP]:
    """Returns a noisy average measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are ``d_in``-close under the ``input_metric``,
    M(x) and M(x') are sampled from distributions that are ``d_out`` apart under the
    ``output_measure``. Noise scale is computed appropriately for the specified
    ``noise_mechanism`` such that the stated privacy property is guaranteed.

    Note:
        ``d_out`` is interpreted as the "epsilon" parameter if ``output_measure`` is
        :class:`~.PureDP`, the "rho" parameter if ``output_measure`` is
        :class:`~.RhoZCDP`, and ("epsilon", "delta") if ``output_measure`` is
        :class:`~.ApproxDP`.

    Note:
        :class:`~.ApproxDP` budgets with delta>0 are not yet supported.

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP`,
            :class:`~.RhoZCDP`, or :class:`~.ApproxDP`).
        d_out: Desired distance between output distributions w.r.t. ``d_in``. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP`, "rho" if it
            is :class:`~.RhoZCDP`, and ("epsilon", "delta") if it is
            :class:`~.ApproxDP`.
        noise_mechanism: Noise mechanism to apply.
        measure_column: Name to column to compute average of.
        lower: Lower clipping bound for ``measure_column``.
        upper: Upper clipping bound for ``measure_column``.
        d_in: Distance between inputs under the ``input_metric``. The returned
            measurement is guaranteed to have output distributions that are ``d_out``
            apart for inputs that are ``d_in`` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy averages for each group obtained from the groupby transformation.
            If None, this measurement outputs a single number - the noisy average.
        average_column: If a ``groupby_transformation`` is supplied, this is the column
            name to be used for noisy average in the DataFrame output by the
            measurement. If None, this column will be named "avg(<measure_column>)".
        keep_intermediates: If True, intermediates (noisy sum of deviations and noisy
            count) will also be output in addition to the noisy average.
        sum_column: If a ``groupby_transformation`` is supplied and
            ``keep_intermediates`` is True, this is the column name to be used
            for intermediate sums in the DataFrame output by the measurement. If
            None, this column will be named "sum(<measure_column>)".
        count_column: If a ``groupby_transformation`` is supplied and
            ``keep_intermediates`` is True, this is the column name to be used
            for intermediate counts in the DataFrame output by the
            measurement. If None, this column will be named "count".
    """
    if isinstance(output_measure, ApproxDP):
        epsilon, delta = ApproxDPBudget(d_out).value
        if noise_mechanism in (NoiseMechanism.LAPLACE, NoiseMechanism.GEOMETRIC):
            if delta > 0:
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Cannot spend an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism}. Use ApproxDP with delta = 0 or PureDP."
                    ),
                )
            return PureDPToApproxDP(
                create_average_measurement(
                    input_domain=input_domain,
                    input_metric=input_metric,
                    output_measure=PureDP(),
                    d_out=epsilon,
                    noise_mechanism=noise_mechanism,
                    measure_column=measure_column,
                    lower=lower,
                    upper=upper,
                    d_in=d_in,
                    groupby_transformation=groupby_transformation,
                    average_column=average_column,
                    keep_intermediates=keep_intermediates,
                    sum_column=sum_column,
                    count_column=count_column,
                )
            )
        elif noise_mechanism in (
            NoiseMechanism.GAUSSIAN,
            NoiseMechanism.DISCRETE_GAUSSIAN,
        ):
            if delta > 0:
                # Once supported, we will compute the corresponding zCDP budget and set
                # the ouptut measure to zCDP.
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Spending an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism} is not yet supported. Use either"
                        f" {NoiseMechanism.LAPLACE} or {NoiseMechanism.GEOMETRIC}."
                    ),
                )
            raise UnsupportedCombinationError(
                (noise_mechanism, output_measure, d_out),
                (
                    f"Cannot spend a budget with delta = 0 using {noise_mechanism}. Set"
                    f" delta > 0 or use either {NoiseMechanism.LAPLACE} or"
                    f" {NoiseMechanism.GEOMETRIC}."
                ),
            )
        else:
            assert False
    elif isinstance(output_measure, (RhoZCDP, PureDP)):
        d_out = PrivacyBudget.cast(output_measure, d_out).value
    else:
        assert False
    d_in = ExactNumber(d_in)
    lower = ExactNumber(lower)
    upper = ExactNumber(upper)
    # help mypy
    assert isinstance(output_measure, (PureDP, RhoZCDP))
    if not average_column:
        average_column = f"avg({measure_column})"
    if not sum_column:
        sum_column = f"avg({sum_column})"
    if not count_column:
        count_column = "count"
    midpoint_of_measure_column, exact_midpoint_of_measure_column = get_midpoint(
        lower=lower,
        upper=upper,
        integer_midpoint=isinstance(
            input_domain[measure_column], SparkIntegerColumnDescriptor
        ),
    )

    deviations_column = get_nonconflicting_string(list(input_domain.schema))
    deviations_map = Map(
        row_transformer=RowToRowTransformation(
            input_domain=SparkRowDomain(input_domain.schema),
            output_domain=SparkRowDomain(
                {**input_domain.schema, deviations_column: input_domain[measure_column]}
            ),
            trusted_f=lambda row: {
                deviations_column: row[measure_column] - midpoint_of_measure_column
            },
            augment=True,
        ),
        metric=input_metric,
    )
    assert isinstance(deviations_map.output_domain, SparkDataFrameDomain)
    assert isinstance(
        deviations_map.output_metric,
        (SymmetricDifference, HammingDistance, IfGroupedBy),
    )
    if groupby_transformation is None:
        sod_measurement = create_sum_measurement(
            input_domain=deviations_map.output_domain,
            input_metric=deviations_map.output_metric,
            measure_column=deviations_column,
            lower=lower - exact_midpoint_of_measure_column,
            upper=upper - exact_midpoint_of_measure_column,
            noise_mechanism=noise_mechanism,
            d_in=d_in,
            d_out=d_out / 2,
            groupby_transformation=None,
            sum_column=None,
            output_measure=output_measure,
        )
        count_measurement = create_count_measurement(
            input_domain=deviations_map.output_domain,
            input_metric=deviations_map.output_metric,
            noise_mechanism=noise_mechanism,
            d_in=d_in,
            d_out=d_out / 2,
            groupby_transformation=None,
            count_column=None,
            output_measure=output_measure,
        )
        sum_and_count = deviations_map | Composition(
            measurements=[sod_measurement, count_measurement]
        )

        def postprocess_sod_and_count(
            answers: List[Union[np.int64, np.float64]],
        ) -> Union[
            np.int64,
            np.float64,
            Dict[str, Union[Union[float, np.int64], Union[int, np.float64]]],
        ]:
            """Computes average from noisy count and sum of deviations."""
            sod, count = answers
            average = sod / max(1, count) + midpoint_of_measure_column
            if keep_intermediates:
                return {
                    "average": average,
                    "sum_of_deviations": sod,
                    "count": count,
                    "midpoint_of_deviations": midpoint_of_measure_column,
                }
            return average

        average_measurement = PostProcess(
            measurement=sum_and_count, f=postprocess_sod_and_count
        )
        assert average_measurement.privacy_function(d_in) == d_out
        return average_measurement
    assert isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared))
    groupby = GroupBy(
        input_domain=deviations_map.output_domain,
        input_metric=input_metric,
        use_l2=groupby_transformation.use_l2,
        group_keys=groupby_transformation.group_keys,
    )
    sod_measurement = create_sum_measurement(
        input_domain=deviations_map.output_domain,
        input_metric=deviations_map.output_metric,
        measure_column=deviations_column,
        lower=lower - exact_midpoint_of_measure_column,
        upper=upper - exact_midpoint_of_measure_column,
        noise_mechanism=noise_mechanism,
        d_in=d_in,
        d_out=d_out / 2,
        groupby_transformation=groupby,
        sum_column=sum_column,
        output_measure=output_measure,
    )
    count_measurement = create_count_measurement(
        input_domain=deviations_map.output_domain,
        input_metric=deviations_map.output_metric,
        noise_mechanism=noise_mechanism,
        d_in=d_in,
        d_out=d_out / 2,
        groupby_transformation=groupby,
        count_column=count_column,
        output_measure=output_measure,
    )
    sum_and_count = deviations_map | Composition(
        measurements=[sod_measurement, count_measurement]
    )

    def postprocess_sod_and_count_dfs(answers: List[DataFrame]) -> DataFrame:
        """Computes average from noisy count and sum of deviations."""
        # Give mypy some help -- none of these can be None, but it has trouble
        # figuring that out because of how closures are handled.
        assert (
            average_column is not None
            and sum_column is not None
            and count_column is not None
        )
        sod_df, count_df = answers
        if groupby.groupby_columns:
            df_with_sod_and_count = join(
                left=sod_df,
                right=count_df,
                on=groupby.groupby_columns,
                how="inner",  # same groups
                nulls_are_equal=True,
            )
        else:
            temp_column = get_nonconflicting_string(sod_df.columns + count_df.columns)
            df_with_sod_and_count = join(
                left=sod_df.withColumn(temp_column, sf.lit(1)),
                right=count_df.withColumn(temp_column, sf.lit(1)),
                on=[temp_column],
                how="inner",  # only one row
                nulls_are_equal=False,  # no nulls
            ).drop(temp_column)

        df_with_all_columns = df_with_sod_and_count.withColumn(
            average_column,
            (sf.col(sum_column) / sf.greatest(sf.lit(1), sf.col(count_column)))
            + sf.lit(midpoint_of_measure_column),
        )
        if keep_intermediates:
            return df_with_all_columns
        return df_with_all_columns.drop(sum_column, count_column)

    average_measurement = PostProcess(
        measurement=sum_and_count, f=postprocess_sod_and_count_dfs
    )
    assert average_measurement.privacy_function(d_in) == d_out
    return average_measurement


@typechecked
def create_variance_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    d_out: PrivacyBudgetInput,
    noise_mechanism: NoiseMechanism,
    measure_column: str,
    lower: ExactNumberInput,
    upper: ExactNumberInput,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    variance_column: Optional[str] = None,
    keep_intermediates: bool = False,
    sum_of_deviations_column: Optional[str] = None,
    sum_of_squared_deviations_column: Optional[str] = None,
    count_column: Optional[str] = None,
) -> Union[PostProcess, PureDPToApproxDP]:
    """Returns a noisy variance measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are ``d_in``-close under the ``input_metric``,
    M(x) and M(x') are sampled from distributions that are ``d_out`` apart under the
    ``output_measure``. Noise scale is computed appropriately for the specified
    ``noise_mechanism`` such that the stated privacy property is guaranteed.

    Note:
        ``d_out`` is interpreted as the "epsilon" parameter if ``output_measure`` is
        :class:`~.PureDP`, the "rho" parameter if ``output_measure`` is
        :class:`~.RhoZCDP`, and ("epsilon", "delta") if ``output_measure`` is
        :class:`~.ApproxDP`.

    Note:
        :class:`~.ApproxDP` budgets with delta>0 are not yet supported.

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP`,
            :class:`~.RhoZCDP`, or :class:`~.ApproxDP`).
        d_out: Desired distance between output distributions w.r.t. ``d_in``. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP`, "rho" if it
            is :class:`~.RhoZCDP`, and ("epsilon", "delta") if it is
            :class:`~.ApproxDP`.
        noise_mechanism: Noise mechanism to apply.
        measure_column: Name to column to compute variance of.
        lower: Lower clipping bound for ``measure_column``.
        upper: Upper clipping bound for ``measure_column``.
        d_in: Distance between inputs under the ``input_metric``. The returned
            measurement is guaranteed to have output distributions that are ``d_out``
            apart for inputs that are ``d_in`` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            a noisy variance for each group obtained from the groupby transformation.
            If None, this measurement outputs a single number - the noisy variance.
        variance_column: If a ``groupby_transformation`` is supplied, this is
            the column name to be used for noisy variance in the DataFrame
            output by the measurement. If None, this column will be named
            "var(<measure_column>)".
        keep_intermediates: If True, intermediates (noisy sum of deviations, noisy sum
            of squared deviations and noisy count) will also be output in addition to
            the noisy variance.
        sum_of_deviations_column: If a ``groupby_transformation`` is supplied and
            ``keep_intermediates`` is True, this is the column name to be used for
            intermediate sums of deviations in the DataFrame output by the measurement.
            If None, this column will be named "sod(<measure_column>)".
        sum_of_squared_deviations_column: If a ``groupby_transformation`` is supplied
            and ``keep_intermediates`` is True, this is the column name to be used for
            intermediate sums of squared deviations in the DataFrame output by the
            measurement. If None, this column will be named "sos(<measure_column>)".
        count_column: If a ``groupby_transformation`` is supplied and
            ``keep_intermediates`` is True, this is the column name to be used
            for intermediate counts in the DataFrame output by the measurement.
            If None, this column will be named "count".
    """
    if isinstance(output_measure, ApproxDP):
        epsilon, delta = ApproxDPBudget(d_out).value
        if noise_mechanism in (NoiseMechanism.LAPLACE, NoiseMechanism.GEOMETRIC):
            if delta > 0:
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Cannot spend an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism}. Use ApproxDP with delta = 0 or PureDP."
                    ),
                )
            return PureDPToApproxDP(
                create_variance_measurement(
                    input_domain=input_domain,
                    input_metric=input_metric,
                    output_measure=PureDP(),
                    d_out=epsilon,
                    noise_mechanism=noise_mechanism,
                    measure_column=measure_column,
                    lower=lower,
                    upper=upper,
                    d_in=d_in,
                    groupby_transformation=groupby_transformation,
                    variance_column=variance_column,
                    keep_intermediates=keep_intermediates,
                    sum_of_deviations_column=sum_of_deviations_column,
                    sum_of_squared_deviations_column=sum_of_squared_deviations_column,
                    count_column=count_column,
                )
            )
        elif noise_mechanism in (
            NoiseMechanism.GAUSSIAN,
            NoiseMechanism.DISCRETE_GAUSSIAN,
        ):
            if delta > 0:
                # Once supported, we will compute the corresponding zCDP budget and set
                # the ouptut measure to zCDP.
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Spending an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism} is not yet supported. Use either"
                        f" {NoiseMechanism.LAPLACE} or {NoiseMechanism.GEOMETRIC}."
                    ),
                )
            raise UnsupportedCombinationError(
                (noise_mechanism, output_measure, d_out),
                (
                    f"Cannot spend a budget with delta = 0 using {noise_mechanism}. Set"
                    f" delta > 0 or use either {NoiseMechanism.LAPLACE} or"
                    f" {NoiseMechanism.GEOMETRIC}."
                ),
            )
        else:
            assert False
    elif isinstance(output_measure, (RhoZCDP, PureDP)):
        d_out = PrivacyBudget.cast(output_measure, d_out).value
    else:
        assert False
    # help mypy
    assert isinstance(output_measure, (PureDP, RhoZCDP))

    if sum_of_deviations_column is None:
        sum_of_deviations_column = f"sod({measure_column})"
    if sum_of_squared_deviations_column is None:
        sum_of_squared_deviations_column = f"sos({measure_column})"
    if count_column is None:
        count_column = "count"
    if variance_column is None:
        variance_column = f"var({measure_column})"

    lower = ExactNumber(lower)
    upper = ExactNumber(upper)
    d_in = ExactNumber(d_in)
    midpoint_of_measure_column, exact_midpoint_of_measure_column = get_midpoint(
        lower,
        upper,
        integer_midpoint=isinstance(
            input_domain[measure_column], SparkIntegerColumnDescriptor
        ),
    )

    lower_after_squaring: ExactNumber = (
        ExactNumber(0) if lower <= 0 <= upper else min(lower**2, upper**2)
    )
    upper_after_squaring: ExactNumber = max(lower**2, upper**2)
    (
        midpoint_of_squared_measure_column,
        exact_midpoint_of_squared_measure_column,
    ) = get_midpoint(
        lower_after_squaring,
        upper_after_squaring,
        integer_midpoint=isinstance(
            input_domain[measure_column], SparkIntegerColumnDescriptor
        ),
    )
    (
        deviations_map,
        deviations_column,
        squared_deviations_column,
    ) = _create_map_to_compute_deviations(
        input_domain=input_domain,
        input_metric=input_metric,
        measure_column=measure_column,
        lower=lower,
        upper=upper,
    )
    assert isinstance(deviations_map.output_domain, SparkDataFrameDomain)
    assert isinstance(
        deviations_map.output_metric,
        (SymmetricDifference, HammingDistance, IfGroupedBy),
    )
    if groupby_transformation is None:
        sod_measurement = create_sum_measurement(
            input_domain=deviations_map.output_domain,
            input_metric=deviations_map.output_metric,
            measure_column=deviations_column,
            lower=lower - exact_midpoint_of_measure_column,
            upper=upper - exact_midpoint_of_measure_column,
            noise_mechanism=noise_mechanism,
            d_in=d_in,
            d_out=d_out / 3,
            groupby_transformation=None,
            sum_column=None,
            output_measure=output_measure,
        )
        sos_measurement = create_sum_measurement(
            input_domain=deviations_map.output_domain,
            input_metric=deviations_map.output_metric,
            measure_column=squared_deviations_column,
            lower=lower_after_squaring - exact_midpoint_of_squared_measure_column,
            upper=upper_after_squaring - exact_midpoint_of_squared_measure_column,
            noise_mechanism=noise_mechanism,
            d_in=d_in,
            d_out=d_out / 3,
            groupby_transformation=None,
            sum_column=None,
            output_measure=output_measure,
        )
        count_measurement = create_count_measurement(
            input_domain=deviations_map.output_domain,
            input_metric=deviations_map.output_metric,
            noise_mechanism=noise_mechanism,
            d_in=d_in,
            d_out=d_out / 3,
            groupby_transformation=None,
            count_column=None,
            output_measure=output_measure,
        )
        sums_and_count = deviations_map | Composition(
            measurements=[sod_measurement, sos_measurement, count_measurement]
        )

        def postprocess_sums_and_count(
            answers: List[Union[np.int64, np.float64]],
        ) -> Union[
            np.int64,
            np.float64,
            Dict[str, Union[Union[float, np.int64], Union[int, np.float64]]],
        ]:
            """Computes variance from noisy count and sums of deviations."""
            sod, sos, count = answers
            variance: Any
            if count <= 1:
                variance = np.nan
            else:
                # The midpoint_of* variables are used to lower noise in the variance
                # calculation. By normalizing the values based on the top and bottom of
                # the measurement, the stability is reduced.

                # This calculates sample variance, which is why all values are
                # proportional to n-1 (ie. count - 1) instead of n.

                variance = (
                    sos / (count - 1)
                    + (count / (count - 1)) * midpoint_of_squared_measure_column
                ) - (sod / count + midpoint_of_measure_column) ** 2 * (
                    count / (count - 1)
                )
                if variance < 0:
                    variance = 0
                else:
                    variance = min(
                        variance,
                        (
                            ExactNumber(upper).to_float(round_up=False)
                            - ExactNumber(lower).to_float(round_up=True)
                        )
                        ** 2
                        / 4,
                    )
            if keep_intermediates:
                return {
                    "variance": variance,
                    "sum_of_deviations": sod,
                    "sum_of_squared_deviations": sos,
                    "count": count,
                    "midpoint_deviations": midpoint_of_measure_column,
                    "midpoint_squared_deviations": midpoint_of_squared_measure_column,
                }
            return variance

        variance_measurement = PostProcess(
            measurement=sums_and_count, f=postprocess_sums_and_count
        )
        assert variance_measurement.privacy_function(d_in) == d_out
        return variance_measurement
    assert isinstance(groupby_transformation, GroupBy)
    assert isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared))
    groupby = GroupBy(
        input_domain=deviations_map.output_domain,
        input_metric=deviations_map.output_metric,
        use_l2=groupby_transformation.use_l2,
        group_keys=groupby_transformation.group_keys,
    )
    sod_measurement = create_sum_measurement(
        input_domain=deviations_map.output_domain,
        input_metric=deviations_map.output_metric,
        measure_column=deviations_column,
        lower=lower - exact_midpoint_of_measure_column,
        upper=upper - exact_midpoint_of_measure_column,
        noise_mechanism=noise_mechanism,
        d_in=d_in,
        d_out=d_out / 3,
        groupby_transformation=groupby,
        sum_column=sum_of_deviations_column,
        output_measure=output_measure,
    )
    sos_measurement = create_sum_measurement(
        input_domain=deviations_map.output_domain,
        input_metric=deviations_map.output_metric,
        measure_column=squared_deviations_column,
        lower=lower_after_squaring - exact_midpoint_of_squared_measure_column,
        upper=upper_after_squaring - exact_midpoint_of_squared_measure_column,
        noise_mechanism=noise_mechanism,
        d_in=d_in,
        d_out=d_out / 3,
        groupby_transformation=groupby,
        sum_column=sum_of_squared_deviations_column,
        output_measure=output_measure,
    )
    count_measurement = create_count_measurement(
        input_domain=input_domain,
        input_metric=input_metric,
        noise_mechanism=noise_mechanism,
        d_in=d_in,
        d_out=d_out / 3,
        groupby_transformation=groupby_transformation,
        count_column=count_column,
        output_measure=output_measure,
    )
    sums_and_count = Composition(
        measurements=[
            deviations_map | sod_measurement,
            deviations_map | sos_measurement,
            count_measurement,
        ]
    )

    def postprocess_sums_and_count_dfs(answers: List[DataFrame]) -> DataFrame:
        """Computes variance from noisy counts and sums."""
        # Give mypy some help -- none of these can be None, but it has trouble
        # figuring that out because of how closures are handled.
        assert (
            variance_column is not None
            and sum_of_deviations_column is not None
            and sum_of_squared_deviations_column is not None
            and count_column is not None
        )
        sod_df, sos_df, count_df = answers
        assert groupby_transformation is not None
        if groupby.groupby_columns:
            df_with_sums_and_count = join(
                left=join(
                    left=sod_df,
                    right=sos_df,
                    on=groupby.groupby_columns,
                    how="inner",  # same groups
                    nulls_are_equal=True,
                ),
                right=count_df,
                on=groupby.groupby_columns,
                how="inner",
                nulls_are_equal=True,
            )
        else:
            temp_column = get_nonconflicting_string(sod_df.columns + sos_df.columns)
            df_with_sums_and_count = join(
                left=join(
                    left=sod_df.withColumn(temp_column, sf.lit(1)),
                    right=sos_df.withColumn(temp_column, sf.lit(1)),
                    on=[temp_column],
                    how="inner",  # only one row
                    nulls_are_equal=False,  # no nulls
                ),
                right=count_df.withColumn(temp_column, sf.lit(1)),
                on=[temp_column],
                how="inner",
                nulls_are_equal=False,
            ).drop(temp_column)

        df_with_all_columns = df_with_sums_and_count.withColumn(
            variance_column,
            sf.when(
                sf.col(count_column) <= 1,
                np.nan,
            ).otherwise(
                sf.lit(
                    (
                        (
                            sf.col(sum_of_squared_deviations_column)
                            / (sf.col(count_column) - 1)
                            + sf.lit(midpoint_of_squared_measure_column)
                            * (sf.col(count_column) / (sf.col(count_column) - 1))
                        )
                        - (
                            sf.col(sum_of_deviations_column) / sf.col(count_column)
                            + sf.lit(midpoint_of_measure_column)
                        )
                        ** 2
                        * (sf.col(count_column) / (sf.col(count_column) - 1))
                    )
                )
            ),
        ).withColumn(
            variance_column,
            sf.when(sf.col(variance_column) == np.nan, np.nan)
            .when(sf.col(variance_column) < 0.0, 0.0)
            .otherwise(
                sf.least(
                    sf.col(variance_column),
                    sf.lit(
                        (
                            ExactNumber(upper).to_float(round_up=False)
                            - ExactNumber(lower).to_float(round_up=True)
                        )
                        ** 2
                        / 4
                    ),
                )
            ),
        )

        if keep_intermediates:
            return df_with_all_columns
        return df_with_all_columns.drop(
            sum_of_deviations_column, sum_of_squared_deviations_column, count_column
        )

    variance_measurement = PostProcess(
        measurement=sums_and_count, f=postprocess_sums_and_count_dfs
    )
    assert variance_measurement.privacy_function(d_in) == d_out
    return variance_measurement


@typechecked
def create_standard_deviation_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    d_out: PrivacyBudgetInput,
    noise_mechanism: NoiseMechanism,
    measure_column: str,
    lower: ExactNumberInput,
    upper: ExactNumberInput,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    standard_deviation_column: Optional[str] = None,
    keep_intermediates: bool = False,
    sum_of_deviations_column: Optional[str] = None,
    sum_of_squared_deviations_column: Optional[str] = None,
    count_column: Optional[str] = None,
) -> Union[PostProcess, PureDPToApproxDP]:
    """Returns a noisy standard deviation measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are ``d_in``-close under the ``input_metric``,
    M(x) and M(x') are sampled from distributions that are ``d_out`` apart under the
    ``output_measure``. Noise scale is computed appropriately for the specified
    ``noise_mechanism`` such that the stated privacy property is guaranteed.

    Note:
        ``d_out`` is interpreted as the "epsilon" parameter if ``output_measure`` is
        :class:`~.PureDP`, the "rho" parameter if ``output_measure`` is
        :class:`~.RhoZCDP`, and ("epsilon", "delta") if ``output_measure`` is
        :class:`~.ApproxDP`.

    Note:
        :class:`~.ApproxDP` budgets with delta>0 are not yet supported.

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP`,
            :class:`~.RhoZCDP`, or :class:`~.ApproxDP`).
        d_out: Desired distance between output distributions w.r.t. ``d_in``. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP`, "rho" if it
            is :class:`~.RhoZCDP`, and ("epsilon", "delta") if it is
            :class:`~.ApproxDP`.
        noise_mechanism: Noise mechanism to apply.
        measure_column: Name to column to compute standard deviation of.
        lower: Lower clipping bound for ``measure_column``.
        upper: Upper clipping bound for ``measure_column``.
        d_in: Distance between inputs under the ``input_metric``. The returned
            measurement is guaranteed to have output distributions that are ``d_out``
            apart for inputs that are ``d_in`` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy standard deviations for each group obtained by applying the groupby
            transformation. If None, this measurement outputs a single number - the
            noisy standard deviation of ``measure_column``.
        standard_deviation_column: If a ``groupby_transformation`` is supplied, this is
            the column name to be used for noisy standard deviation in the DataFrame
            output by the measurement. If None, this column will be named
            "stddev(<measure_column>)".
        keep_intermediates: If True, intermediates (noisy sum of deviations, noisy sum
            of squared deviations noisy count) will also be output in addition to the
            noisy standard deviation.
        sum_of_deviations_column: If a ``groupby_transformation`` is supplied and
            ``keep_intermediates`` is True, this is the column name to be used for
            intermediate sums of deviations in the DataFrame output by the measurement.
            If None, this column will be named "sod(<measure_column>)".
        sum_of_squared_deviations_column: If a ``groupby_transformation`` is supplied
            and ``keep_intermediates`` is True, this is the column name to be used for
            intermediate sums of squared_deviations in the DataFrame output by the
            measurement. If None, this column will be named "sos(<measure_column>)".
        count_column: If a ``groupby_transformation`` is supplied and
            ``keep_intermediates`` is True, this is the column name to be used
            for intermediate counts in the DataFrame output by the measurement.
            If None, this column will be named "count".
    """
    if isinstance(output_measure, ApproxDP):
        epsilon, delta = ApproxDPBudget(d_out).value
        if noise_mechanism in (NoiseMechanism.LAPLACE, NoiseMechanism.GEOMETRIC):
            if delta > 0:
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Cannot spend an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism}. Use ApproxDP with delta = 0 or PureDP."
                    ),
                )
            return PureDPToApproxDP(
                create_standard_deviation_measurement(
                    input_domain=input_domain,
                    input_metric=input_metric,
                    output_measure=PureDP(),
                    d_out=epsilon,
                    noise_mechanism=noise_mechanism,
                    measure_column=measure_column,
                    lower=lower,
                    upper=upper,
                    d_in=d_in,
                    groupby_transformation=groupby_transformation,
                    standard_deviation_column=standard_deviation_column,
                    keep_intermediates=keep_intermediates,
                    sum_of_deviations_column=sum_of_deviations_column,
                    sum_of_squared_deviations_column=sum_of_squared_deviations_column,
                    count_column=count_column,
                )
            )
        elif noise_mechanism in (
            NoiseMechanism.GAUSSIAN,
            NoiseMechanism.DISCRETE_GAUSSIAN,
        ):
            if delta > 0:
                # Once supported, we will compute the corresponding zCDP budget and set
                # the ouptut measure to zCDP.
                raise UnsupportedCombinationError(
                    (noise_mechanism, output_measure, d_out),
                    (
                        "Spending an ApproxDP budget with delta > 0 using mechanism"
                        f" {noise_mechanism} is not yet supported. Use either"
                        f" {NoiseMechanism.LAPLACE} or {NoiseMechanism.GEOMETRIC}."
                    ),
                )
            raise UnsupportedCombinationError(
                (noise_mechanism, output_measure, d_out),
                (
                    f"Cannot spend a budget with delta = 0 using {noise_mechanism}. Set"
                    f" delta > 0 or use either {NoiseMechanism.LAPLACE} or"
                    f" {NoiseMechanism.GEOMETRIC}."
                ),
            )
        else:
            assert False
    elif isinstance(output_measure, (RhoZCDP, PureDP)):
        d_out = PrivacyBudget.cast(output_measure, d_out).value
    else:
        assert False
    # help mypy
    assert isinstance(output_measure, (PureDP, RhoZCDP))
    lower = ExactNumber(lower)
    upper = ExactNumber(upper)
    d_in = ExactNumber(d_in)
    if not standard_deviation_column:
        standard_deviation_column = f"stddev({measure_column})"
    variance_measurement = create_variance_measurement(
        input_domain=input_domain,
        input_metric=input_metric,
        measure_column=measure_column,
        lower=lower,
        upper=upper,
        noise_mechanism=noise_mechanism,
        d_in=d_in,
        d_out=d_out,
        groupby_transformation=groupby_transformation,
        variance_column=standard_deviation_column,
        keep_intermediates=keep_intermediates,
        sum_of_deviations_column=sum_of_deviations_column,
        sum_of_squared_deviations_column=sum_of_squared_deviations_column,
        count_column=count_column,
        output_measure=output_measure,
    )

    if groupby_transformation is None:

        def postprocess_variance(
            answer: Union[Dict[str, ArrayLike], ArrayLike],
        ) -> Union[Dict[str, ArrayLike], ArrayLike]:
            """Computes variance from noisy standard deviation."""
            if isinstance(answer, dict):
                answer["standard-deviation"] = np.sqrt(answer["variance"])
                del answer["variance"]
                return answer
            return np.sqrt(answer)

        return PostProcess(measurement=variance_measurement, f=postprocess_variance)

    def postprocess_variance_df(sdf: DataFrame) -> DataFrame:
        # Give mypy some help -- this can't be None, but mypy has trouble figuring
        # that out because of how closures are handled.
        assert standard_deviation_column is not None
        return sdf.withColumn(
            standard_deviation_column, sf.sqrt(sf.col(standard_deviation_column))
        )

    return PostProcess(measurement=variance_measurement, f=postprocess_variance_df)


@typechecked
def create_quantile_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    d_out: PrivacyBudgetInput,
    measure_column: str,
    quantile: float,
    lower: Union[int, float],
    upper: Union[int, float],
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    quantile_column: Optional[str] = None,
) -> Union[PostProcess, PureDPToApproxDP]:
    """Returns a noisy quantile measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are ``d_in``-close under the ``input_metric``,
    M(x) and M(x') are sampled from distributions that are ``d_out`` apart under the
    ``output_measure``.

    Note:
        ``d_out`` is interpreted as the "epsilon" parameter if ``output_measure`` is
        :class:`~.PureDP`, the "rho" parameter if ``output_measure`` is
        :class:`~.RhoZCDP`, and ("epsilon", "delta") if ``output_measure`` is
        :class:`~.ApproxDP`.

    Note:
        :class:`~.ApproxDP` budgets with delta>0 are not yet supported.

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP`,
            :class:`~.RhoZCDP`, or :class:`~.ApproxDP`).
        d_out: Desired distance between output distributions w.r.t. ``d_in``. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP`, "rho" if it
            is :class:`~.RhoZCDP`, and ("epsilon", "delta") if it is
            :class:`~.ApproxDP`.
        measure_column: Name to column to compute quantile of.
        quantile: The quantile to produce.
        lower: Lower clipping bound for ``measure_column``.
        upper: Upper clipping bound for ``measure_column``.
        d_in: Distance between inputs under the ``input_metric``. The returned
            measurement is guaranteed to have output distributions that are ``d_out``
            apart for inputs that are ``d_in`` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy quantiles for each group obtained by applying groupby.
            If None, this measurement outputs a single number - the noisy quantile.
        quantile_column: If a ``groupby_transformation`` is supplied, this is
            the column name to be used for noisy quantile in the DataFrame
            output by the measurement. If None, this column will be named
            "q_(<quantile>)_(<measure_column>)".
    """
    if isinstance(output_measure, ApproxDP):
        epsilon, delta = ApproxDPBudget(d_out).value
        if delta > 0:
            # We could support this by finding the corresponding zCDP budget and calling
            # the zCDP version of quantile. We should be careful about this though,
            # because for some values of delta, this strategy might be worse than just
            # using the epsilon budget with PureDP.
            raise ValueError(
                "Spending an ApproxDP budget with delta > 0 is not yet supported. Use"
                " ApproxDP with delta = 0 or PureDP."
            )
        return PureDPToApproxDP(
            create_quantile_measurement(
                input_domain=input_domain,
                input_metric=input_metric,
                output_measure=PureDP(),
                d_out=epsilon,
                measure_column=measure_column,
                quantile=quantile,
                lower=lower,
                upper=upper,
                d_in=d_in,
                groupby_transformation=groupby_transformation,
                quantile_column=quantile_column,
            )
        )
    elif isinstance(output_measure, (RhoZCDP, PureDP)):
        d_out = PrivacyBudget.cast(output_measure, d_out).value
    else:
        assert False
    # help mypy
    assert isinstance(output_measure, (PureDP, RhoZCDP))
    d_in = ExactNumber(d_in)
    if not quantile_column:
        quantile_column = f"q_({quantile})_({measure_column})"

    postprocess = lambda df: df.withColumnRenamed(measure_column, quantile_column)
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise UnsupportedMetricError(
                input_metric,
                (
                    "IfGroupedBy must be accompanied by an appropriate groupby "
                    "transformation."
                ),
            )
        spark = SparkSession.builder.getOrCreate()
        groupby_transformation = GroupBy(
            input_domain=input_domain,
            input_metric=input_metric,
            use_l2=False,
            group_keys=spark.createDataFrame([], schema=StructType([])),
        )
        # Postprocess to obtain the answer if no groupby transformation
        postprocess = lambda df: df.collect()[0][measure_column]
    if groupby_transformation.input_metric != input_metric:
        raise MetricMismatchError(
            (groupby_transformation.input_metric, input_metric),
            (
                "Input metric must match with groupby transformation. Expected:"
                f" ({groupby_transformation.input_metric}), actual: ({input_metric})"
            ),
        )
    if groupby_transformation.input_domain != input_domain:
        raise DomainMismatchError(
            (groupby_transformation.input_domain, input_domain),
            (
                "Input domain must match with groupby transformation. Expected:"
                f" ({groupby_transformation.input_domain}), actual: ({input_domain})"
            ),
        )
    quantile_input_domain = PandasSeriesDomain(
        input_domain[measure_column].to_numpy_domain()
    )
    d_mid = groupby_transformation.stability_function(d_in)
    if output_measure == RhoZCDP():
        epsilon = (8 * d_out) ** "1/2" / d_mid
    else:
        assert output_measure == PureDP()
        epsilon = d_out / d_mid

    noisy_quantile_measurement = NoisyQuantile(
        input_domain=quantile_input_domain,
        quantile=quantile,
        lower=lower,
        upper=upper,
        epsilon=epsilon,
        output_measure=output_measure,
    )

    pandas_schema = {
        measure_column: PandasSeriesDomain(
            input_domain[measure_column].to_numpy_domain()
        )
    }
    df_aggregation_function = AggregateByColumn(
        input_domain=PandasDataFrameDomain(pandas_schema),
        column_to_aggregation={measure_column: noisy_quantile_measurement},
    )
    assert isinstance(groupby_transformation.output_domain, SparkGroupedDataFrameDomain)
    assert isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared))

    apply_quantile_measurement = ApplyInPandas(
        input_domain=groupby_transformation.output_domain,
        input_metric=groupby_transformation.output_metric,
        aggregation_function=df_aggregation_function,
    )

    quantile_measurement = PostProcess(
        measurement=groupby_transformation | apply_quantile_measurement, f=postprocess
    )
    assert quantile_measurement.privacy_function(d_in) == d_out
    return quantile_measurement


def get_midpoint(
    lower: ExactNumberInput, upper: ExactNumberInput, integer_midpoint: bool = False
) -> Tuple[Union[float, int], ExactNumber]:
    """Returns the midpoint of lower and upper.

    If integer_midpoint is True, the midpoint is rounded to the nearest integer using
    :func:`round`.

    Examples:
        >>> get_midpoint(1, 2)
        (1.5, 3/2)
        >>> get_midpoint(1, 5)
        (3.0, 3)
        >>> get_midpoint("0.2", "0.3")
        (0.25, 1/4)
        >>> get_midpoint(1, 9, integer_midpoint=True)
        (5, 5)
    """
    lower = ExactNumber(lower)
    upper = ExactNumber(upper)
    lower_ceil = lower.to_float(round_up=True)
    upper_floor = upper.to_float(round_up=False)
    if integer_midpoint:
        midpoint: Union[int, float] = round(lower_ceil * 0.5 + upper_floor * 0.5)
        return (midpoint, ExactNumber(int(midpoint)))
    midpoint = lower_ceil * 0.5 + upper_floor * 0.5
    exact_midpoint = (lower + upper) / 2
    return midpoint, exact_midpoint


def _create_map_to_compute_deviations(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    measure_column: str,
    lower: ExactNumber,
    upper: ExactNumber,
) -> Tuple[Map, str, str]:
    """Returns a map to produce deviations and squared deviations of measure column."""
    midpoint_of_measure_column, _ = get_midpoint(
        lower,
        upper,
        integer_midpoint=isinstance(
            input_domain[measure_column], SparkIntegerColumnDescriptor
        ),
    )

    lower_after_squaring: ExactNumber = (
        ExactNumber(0) if lower <= 0 <= upper else min(lower**2, upper**2)
    )
    upper_after_squaring: ExactNumber = max(lower**2, upper**2)
    (midpoint_of_squared_measure_column, _) = get_midpoint(
        lower_after_squaring,
        upper_after_squaring,
        integer_midpoint=isinstance(
            input_domain[measure_column], SparkIntegerColumnDescriptor
        ),
    )

    deviations_column = get_nonconflicting_string(list(input_domain.schema))
    squared_deviations_column = get_nonconflicting_string(
        list(input_domain.schema) + [deviations_column]
    )

    return (
        Map(
            row_transformer=RowToRowTransformation(
                input_domain=SparkRowDomain(input_domain.schema),
                output_domain=SparkRowDomain(
                    {
                        **input_domain.schema,
                        deviations_column: input_domain[measure_column],
                        squared_deviations_column: input_domain[measure_column],
                    }
                ),
                trusted_f=lambda row: {
                    deviations_column: row[measure_column] - midpoint_of_measure_column,
                    squared_deviations_column: row[measure_column] ** 2
                    - midpoint_of_squared_measure_column,
                },
                augment=True,
            ),
            metric=input_metric,
        ),
        deviations_column,
        squared_deviations_column,
    )


@typechecked
def create_partition_selection_measurement(
    input_domain: SparkDataFrameDomain,
    epsilon: ExactNumberInput,
    delta: ExactNumberInput,
    d_in: ExactNumberInput = 1,
    count_column: Optional[str] = None,
) -> GeometricPartitionSelection:
    """Returns a partition selection measurement.

    A partition selection measurement created by this function will have a
    privacy guarantee such that
    ``measurement.privacy_function(d_in) = (epsilon, delta)``.

    Args:
        input_domain: Domain of the input Spark DataFrames. Input cannot contain
            floating point columns.
        epsilon: The epsilon portion of the (epsilon, delta) privacy budget that
            you want this measurement to satisfy.
        delta: The delta portion of the (epsilon, delta) privacy budget that
            you want this measurement to satisfy.
        d_in: The given d_in such that
            ``measurement.privacy_function(d_in) = (epsilon, delta)``.
        count_column: Column name for output group counts. If None, output column
            will be named "count".
    """
    d_in = ExactNumber(d_in)
    epsilon = ExactNumber(epsilon)
    delta = ExactNumber(delta)
    alpha: ExactNumberInput
    threshold: int

    if d_in < 1:
        raise NotImplementedError(
            "Creating a partition selection measurement with d_in < 1 is not yet"
            " supported."
        )

    # Constructing the ApproxDPBudget does validation on the values.
    if ApproxDPBudget((epsilon, delta)).is_finite():
        if d_in != 1:
            # Convert everything to its d_in=1 equivalent
            epsilon = epsilon / d_in
            delta = delta / (d_in * sp.E ** (d_in * epsilon))

        alpha = 1 / epsilon
        # Despite the name, to_float will return an integer here.
        # The value of round_up is irrelevant.
        threshold = (
            double_sided_geometric_inverse_cmf_exact(1 - delta, alpha)  # type: ignore
            + 2
        ).to_float(round_up=True)
        assert isinstance(threshold, int)

    else:  # If the budget is infinite.
        alpha = 0
        threshold = 0

    return GeometricPartitionSelection(
        input_domain=input_domain,
        threshold=threshold,
        alpha=alpha,
        count_column=count_column,
    )


@typechecked
def create_bounds_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, IfGroupedBy],
    output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    d_out: PrivacyBudgetInput,
    measure_column: str,
    threshold: float,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    upper_bound_column: Optional[str] = None,
    lower_bound_column: Optional[str] = None,
) -> Union[PostProcess, PureDPToApproxDP, PureDPToRhoZCDP]:
    """Returns a bounds measurement.

    The bounds measurement returns either a tuple of `(lower bound, upper_bound)` if no
    groupby transformation is provided, or a dataframe with one column for the lower
    bound and one column for the upper bound if a groupby transformation is provided.

    The bounds measurement created by this function will have a
    privacy guarantee such that
    ``measurement.privacy_function(d_in) = d_out``.

    Args:
        input_domain: Domain of the input Spark DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee.
        d_out: Desired distance between output distributions w.r.t. ``d_in``. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP`, "rho" if it
            is :class:`~.RhoZCDP`, and ("epsilon", "delta") if it is
            :class:`~.ApproxDP`.
        measure_column: Column name to calculate the bounds for. The column
            must be an integer or floating point column.
        threshold: The threshold for the bound selection measurement.
        d_in: The given d_in such that
            ``measurement.privacy_function(d_in) = epsilon``.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy bounds for each group obtained by applying groupby.
            If None, this measurement outputs a single tuple - the noisy bounds.
        upper_bound_column: If a `groupby_transformation` is supplied, this is
            the column name to be used for the upper bound in the DataFrame
            output by the measurement. If None, this column will be named
            "upper_bound(<measure_column>)".
        lower_bound_column: If a `groupby_transformation` is supplied, this is
            the column name to be used for the lower bound in the DataFrame
            output by the measurement. If None, this column will be named
            "lower_bound(<measure_column>)".
    """
    d_in = ExactNumber(d_in)
    alpha: ExactNumber

    if isinstance(output_measure, ApproxDP):
        epsilon, delta = ApproxDPBudget(d_out).value
        if delta > 0:
            raise ValueError(
                "Cannot spend an ApproxDP budget with delta > 0."
                "Use RhoZCDP, ApproxDP with delta = 0, or PureDP."
            )
        return PureDPToApproxDP(
            create_bounds_measurement(
                input_domain=input_domain,
                input_metric=input_metric,
                output_measure=PureDP(),
                d_out=epsilon,
                measure_column=measure_column,
                threshold=threshold,
                d_in=d_in,
                groupby_transformation=groupby_transformation,
                upper_bound_column=upper_bound_column,
                lower_bound_column=lower_bound_column,
            )
        )

    if isinstance(output_measure, RhoZCDP):
        rho = RhoZCDPBudget(d_out).value
        epsilon = sp.sqrt(ExactNumber(sp.Integer(2) * rho).expr)
        return PureDPToRhoZCDP(
            create_bounds_measurement(
                input_domain=input_domain,
                input_metric=input_metric,
                output_measure=PureDP(),
                d_out=epsilon,
                measure_column=measure_column,
                threshold=threshold,
                d_in=d_in,
                groupby_transformation=groupby_transformation,
                upper_bound_column=upper_bound_column,
                lower_bound_column=lower_bound_column,
            )
        )

    if isinstance(output_measure, PureDP):
        d_out = PrivacyBudget.cast(output_measure, d_out).value
    else:
        assert False

    if not upper_bound_column:
        upper_bound_column = f"upper_bound({measure_column})"
    if not lower_bound_column:
        lower_bound_column = f"lower_bound({measure_column})"

    spark = SparkSession.builder.getOrCreate()
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise UnsupportedMetricError(
                input_metric,
                (
                    "IfGroupedBy must be accompanied by an appropriate groupby "
                    "transformation."
                ),
            )
        input_or_constructed_groupby_transformation = GroupBy(
            input_domain=input_domain,
            input_metric=input_metric,
            use_l2=False,
            group_keys=spark.createDataFrame([], schema=StructType([])),
        )
    else:
        input_or_constructed_groupby_transformation = groupby_transformation

    if input_or_constructed_groupby_transformation.input_metric != input_metric:
        raise MetricMismatchError(
            (input_or_constructed_groupby_transformation.input_metric, input_metric),
            (
                "Input metric must match with groupby transformation. Expected:"
                f" ({input_or_constructed_groupby_transformation.input_metric}), "
                "actual: ({input_metric})"
            ),
        )
    if input_or_constructed_groupby_transformation.input_domain != input_domain:
        raise DomainMismatchError(
            (input_or_constructed_groupby_transformation.input_domain, input_domain),
            (
                "Input domain must match with groupby transformation. Expected:"
                f" ({input_or_constructed_groupby_transformation.input_domain}), "
                f"actual: ({input_domain})"
            ),
        )
    if d_in < 1:
        raise NotImplementedError(
            "Creating a bounds measurement with d_in < 1 is not yet supported."
        )

    # Create map transformation.
    rank_column = get_nonconflicting_string(list(input_domain.schema))
    count_column = get_nonconflicting_string(list(input_domain.schema) + [rank_column])
    element_type = input_domain[measure_column]

    bucket_group_keys = spark.createDataFrame(
        [(i,) for i in range(element_type.size - 1)]
        if isinstance(element_type, SparkIntegerColumnDescriptor)
        else [(i,) for i in range(-100, 101)],
        schema=StructType([StructField(rank_column, IntegerType(), False)]),
    )

    def _add_bucket_index(row: Row) -> Union[Dict[str, int], Dict[str, float]]:
        def clamp(x, lower, upper):  # type: ignore
            return max(min(x, upper), lower)

        return {
            rank_column: clamp(
                -100
                if row[measure_column] == 0
                else ceil(log2(abs(row[measure_column]))),
                -100 if isinstance(element_type, SparkFloatColumnDescriptor) else 0,
                100,
            )
        }

    add_bucket_index = Map(
        metric=input_metric,
        row_transformer=RowToRowTransformation(
            input_domain=SparkRowDomain(input_domain.schema),
            output_domain=SparkRowDomain(
                {
                    **input_domain.schema,
                    rank_column: convert_spark_schema(bucket_group_keys.schema)[
                        rank_column
                    ],
                }
            ),
            trusted_f=_add_bucket_index,
            augment=True,
        ),
    )

    # help mypy
    assert isinstance(add_bucket_index.output_domain, SparkDataFrameDomain)
    # Unwrap IfGroupedBy if necessary
    maybe_unwrap = (
        UnwrapIfGroupedBy(
            domain=add_bucket_index.output_domain,
            input_metric=add_bucket_index.output_metric,
        )
        if isinstance(add_bucket_index.output_metric, IfGroupedBy)
        else Identity(
            domain=add_bucket_index.output_domain,
            metric=add_bucket_index.output_metric,
        )
    )

    # help mypy
    assert isinstance(maybe_unwrap.output_domain, SparkDataFrameDomain)
    assert isinstance(maybe_unwrap.output_metric, SymmetricDifference)
    # Redefine groupby transformation to include the bucket.
    augmented_groupby = GroupBy(
        input_domain=maybe_unwrap.output_domain,
        input_metric=maybe_unwrap.output_metric,
        use_l2=False,
        group_keys=input_or_constructed_groupby_transformation.group_keys.join(
            bucket_group_keys, how="outer"
        ),
    )

    # Define count transformation.
    # help mypy
    assert isinstance(augmented_groupby.output_domain, SparkGroupedDataFrameDomain)
    assert isinstance(augmented_groupby.output_metric, SumOf)
    count_transformation = CountGrouped(
        input_domain=augmented_groupby.output_domain,
        input_metric=augmented_groupby.output_metric,
        count_column=count_column,
    )

    prefix_transformation = (
        add_bucket_index | maybe_unwrap | augmented_groupby | count_transformation
    )
    d_mid = prefix_transformation.stability_function(d_in)

    if d_out == float("inf"):
        alpha = ExactNumber(0)
    else:
        alpha = (4 / d_out) * d_mid

    # help mypy
    assert isinstance(prefix_transformation.output_domain, SparkDataFrameDomain)
    apply_bounds_measurement = SparseVectorPrefixSums(
        input_domain=prefix_transformation.output_domain,
        count_column=count_column,
        rank_column=rank_column,
        alpha=alpha,
        grouping_columns=input_or_constructed_groupby_transformation.groupby_columns,
        threshold_fraction=threshold,
    )

    if upper_bound_column is None:
        upper_bound_column = f"upper_bound({measure_column})"
    if lower_bound_column is None:
        lower_bound_column = f"lower_bound({measure_column})"

    def postprocess(df: DataFrame) -> Union[DataFrame, Tuple]:
        """Postprocess by computing bounds from index."""
        # help mypy
        assert upper_bound_column is not None
        assert lower_bound_column is not None
        if groupby_transformation is None:
            bound = 2 ** df.collect()[0][rank_column]
            return (-bound, bound)
        else:
            return (
                df.withColumn(
                    upper_bound_column,
                    (2 ** sf.col(rank_column)).cast(element_type.data_type),
                )
                .withColumn(lower_bound_column, -sf.col(upper_bound_column))
                .drop(rank_column)
                .drop(count_column)
            )

    bounds_measurement = PostProcess(
        measurement=prefix_transformation | apply_bounds_measurement,
        f=postprocess,
    )
    assert bounds_measurement.privacy_function(d_in) == d_out
    return bounds_measurement
