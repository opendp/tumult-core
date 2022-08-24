"""Derived measurements for computing noisy aggregates on spark DataFrames."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-member

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import StructType
from typeguard import typechecked

from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.composition import Composition
from tmlt.core.measurements.converters import PureDPToRhoZCDP
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.dataframe import AggregateByColumn
from tmlt.core.measurements.pandas_measurements.series import (
    AddNoiseToSeries,
    NoisyQuantile,
)
from tmlt.core.measurements.postprocess import PostProcess
from tmlt.core.measurements.spark_measurements import AddNoiseToColumn, ApplyInPandas
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.spark_transformations.agg import (
    SumGrouped,
    create_count_aggregation,
    create_count_distinct_aggregation,
    create_sum_aggregation,
)
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.map import (
    Map,
    RowToRowTransformation,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import get_nonconflicting_string
from tmlt.core.utils.parameters import calculate_noise_scale


class NoiseMechanism(Enum):
    """Enumerating noise mechanisms."""

    LAPLACE = 1
    GEOMETRIC = 2
    DISCRETE_GAUSSIAN = 3


@typechecked
def create_count_measurement(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    output_measure: Union[PureDP, RhoZCDP],
    d_out: ExactNumberInput,
    noise_mechanism: NoiseMechanism,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    count_column: Optional[str] = None,
) -> Measurement:
    """Returns a noisy count measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are `d_in`-close under the `input_metric`, M(x) and
    M(x') are sampled from distributions that are `d_out` apart under the
    `output_measure`. Noise scale is computed appropriately for the specified
    `noise_mechanism` such that the stated privacy property is guaranteed.

    Note:
        `d_out` is interpreted as the "epsilon" parameter if `output_measure` is
        :class:`~.PureDP`, otherwise it is interpreted as the "rho" parameter (if
        `output_measure` is :class:`~.RhoZCDP`).

    Args:
        input_domain: Domain of input spark DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP` or
            :class:`~.RhoZCDP`).
        d_out: Desired distance between output distributions w.r.t. `d_in`. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP` and as
            "rho" if it is :class:`~.RhoZCDP`.
        noise_mechanism: Noise mechanism to apply to count(s).
        d_in: Distance between inputs under the `input_metric`. The returned
            measurement is guaranteed to have output distributions that are `d_out`
            apart for inputs that are `d_in` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy counts for each group obtained by applying the groupby transformation
            . Otherwise, this measurement outputs a single number - the noisy count.
        count_column: If a `groupby_transformation` is provided, this is the column
            name to be used for counts in the dataframe output by the measurement. If
            None, this column will be named "count".
    """
    d_in = ExactNumber(d_in)
    d_out = ExactNumber(d_out)
    if (
        noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
        and output_measure != RhoZCDP()
    ):
        raise ValueError(
            "Discrete Gaussian noise mechanism can only be used with RhoZCDP."
        )
    count_aggregation: Transformation
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise ValueError(
                "Cannot use IfGroupedBy input metric if no groupby_transformation is "
                "provided"
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
        else:
            assert noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
            add_noise_to_number = AddDiscreteGaussianNoise(
                sigma_squared=noise_scale ** 2
            )
        count_measurement: Measurement = count_aggregation | add_noise_to_number
        if (
            output_measure == RhoZCDP()
            and noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
        ):
            # count_measurement has output_measure PureDP and needs to be wrapped in a
            # converter.
            count_measurement = PureDPToRhoZCDP(count_measurement)
        assert count_measurement.privacy_function(d_in) == d_out
        return count_measurement
    assert isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared))
    assert isinstance(groupby_transformation.output_domain, SparkGroupedDataFrameDomain)
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
    else:
        assert noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
        add_noise_to_series = AddNoiseToSeries(
            AddDiscreteGaussianNoise(sigma_squared=noise_scale ** 2)
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
        and noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
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
    output_measure: Union[PureDP, RhoZCDP],
    d_out: ExactNumberInput,
    noise_mechanism: NoiseMechanism,
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    count_column: Optional[str] = None,
) -> Measurement:
    """Returns a noisy count_distinct measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are `d_in`-close under the `input_metric`,
    M(x) and M(x') are sampled from distributions that are `d_out` apart
    under the `output_measure`. Noise scale is computed appropriately for the
    specified `noise_mechanism` such that the stated privacy property
    is guaranteed.

    Note:
        `d_out` is interpreted as the "epsilon" parameter if `output_measure` is
        :class:`~.PureDP`, otherwise it is interpreted as the "rho" parameter
        (if `output_measure` is :class:`~.RhoZCDP`).

    Args:
        input_domain: Domain of input spark DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP` or
            :class:`~.RhoZCDP`).
        d_out: Desired distance between output distributions with respect to
            `d_in`. This is interpreted as "epsilon" if `output_measure` is
            :class:`~.PureDP` and as "rho" if `output_measure` is
            :class:`~.RhoZCDP`.
        noise_mechanism: Noise mechanism to apply to count(s).
        d_in: Distance between inputs under the `input_metric`. The returned
            measurement is guaranteed to have output distributions that are
            `d_out` apart for inputs that are `d_in` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame
            with noisy counts for each group obtained by applying the groupby
            transformation. Otherwise, this measurement outputs a single number -
            the noisy count of distinct items.
        count_column: If a `groupby_transformation` is provided, this is the
            column name to be used for counts in the dataframe output by the
            measurement. If None, this column will be named "count".
    """
    d_in = ExactNumber(d_in)
    d_out = ExactNumber(d_out)
    if (
        noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
        and output_measure != RhoZCDP()
    ):
        raise ValueError(
            "Discrete Gaussian noise mechanism can only be used with RhoZCDP."
        )
    count_distinct_aggregation: Transformation
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise ValueError(
                "Cannot use IfGroupedBy input metric if no"
                "groupby_transformation is provided."
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
        else:
            if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN:
                raise ValueError(
                    f"Unrecognized noise mechanism {noise_mechanism}. "
                    "Supported noise mechanisms are LAPLACE, "
                    "GEOMETRIC, and DISCRETE_GAUSSIAN."
                )
            add_noise_to_number = AddDiscreteGaussianNoise(
                sigma_squared=noise_scale ** 2
            )
        count_distinct_measurement: Measurement = (
            count_distinct_aggregation | add_noise_to_number
        )
        if (
            output_measure == RhoZCDP()
            and noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
        ):
            # the measurement created above has output_measure PureDP,
            # so it needs to be converted
            count_distinct_measurement = PureDPToRhoZCDP(count_distinct_measurement)
        assert count_distinct_measurement.privacy_function(d_in) == d_out
        return count_distinct_measurement
    if not isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared)):
        raise ValueError(
            "A groupby_transformation for count_distinct_measurement must have an "
            "output metric of either SumOf or RootSumOfSquared."
        )
    if not isinstance(
        groupby_transformation.output_domain, SparkGroupedDataFrameDomain
    ):
        raise ValueError(
            "A groupby_transformation for count_distinct_measurement must have an "
            "output domain of SparkGroupedDataFrameDomain."
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
    else:
        if noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN:
            raise ValueError(
                f"Unrecognized noise mechanism {noise_mechanism}. "
                "Supported noise mechanisms are LAPLACE, "
                "GEOMETRIC, and DISCRETE_GAUSSIAN."
            )
        add_noise_to_series = AddNoiseToSeries(
            AddDiscreteGaussianNoise(sigma_squared=noise_scale ** 2)
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
        and noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
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
    output_measure: Union[PureDP, RhoZCDP],
    d_out: ExactNumberInput,
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
    for any two inputs x, x' that are `d_in`-close under the `input_metric`, M(x) and
    M(x') are sampled from distributions that are `d_out` apart under the
    `output_measure`. Noise scale is computed appropriately for the specified
    `noise_mechanism` such that the stated privacy property is guaranteed.

    Note:
        `d_out` is interpreted as the "epsilon" parameter if `output_measure` is
        :class:`~.PureDP`, otherwise it is interpreted as the "rho" parameter (if
        `output_measure` is :class:`~.RhoZCDP`).

    Args:
        input_domain: Domain of input spark DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP` or
            :class:`~.RhoZCDP`).
        d_out: Desired distance between output distributions w.r.t. `d_in`. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP` and as
            "rho" if it is :class:`~.RhoZCDP`.
        noise_mechanism: Noise mechanism to be applied to the sum(s).
        measure_column: Column to be summed.
        lower: Lower clipping bound on `measure_column`.
        upper: Upper clipping bound on `measure_column`.
        d_in: Distance between inputs under the `input_metric`. The returned
            measurement is guaranteed to have output distributions that are `d_out`
            apart for inputs that are `d_in` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy sums for each group obtained by applying the groupby transformation.
            If None, this measurement outputs a single number - the noisy sum.
        sum_column: If a `groupby_transformation` is supplied, this is the column
            name to be used for sums in the DataFrame output by the measurement. If
            None, this column will be named "sum(<measure_column>)".
    """
    lower = ExactNumber(lower)
    upper = ExactNumber(upper)
    d_in = ExactNumber(d_in)
    d_out = ExactNumber(d_out)
    if (
        noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
        and output_measure != RhoZCDP()
    ):
        raise ValueError(
            "Discrete Gaussian noise mechanism can only be used with RhoZCDP."
        )
    sum_aggregation: Transformation
    measure_column_domain = input_domain[measure_column].to_numpy_domain()
    if not isinstance(measure_column_domain, (NumpyIntegerDomain, NumpyFloatDomain)):
        raise ValueError(f"Measure column must be numeric, not {measure_column_domain}")
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise ValueError(
                "IfGroupedBy must be accompanied by an appropriate groupby "
                "transformation."
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
        else:
            assert noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
            add_noise_to_number = AddDiscreteGaussianNoise(
                sigma_squared=noise_scale ** 2
            )
        sum_measurement: Measurement = sum_aggregation | add_noise_to_number
        if (
            output_measure == RhoZCDP()
            and noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
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
    else:
        assert noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
        add_noise_to_series = AddNoiseToSeries(
            AddDiscreteGaussianNoise(sigma_squared=noise_scale ** 2)
        )
    assert isinstance(sum_aggregation.output_domain, SparkDataFrameDomain)
    add_noise_to_column = AddNoiseToColumn(
        input_domain=sum_aggregation.output_domain,
        measure_column=cast(SumGrouped, sum_aggregation).sum_column,
        measurement=add_noise_to_series,
    )
    sum_measurement = groupby_sum | add_noise_to_column
    if (
        output_measure == RhoZCDP()
        and noise_mechanism != NoiseMechanism.DISCRETE_GAUSSIAN
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
    output_measure: Union[PureDP, RhoZCDP],
    d_out: ExactNumberInput,
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
) -> PostProcess:
    """Returns a noisy average measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are `d_in`-close under the `input_metric`, M(x) and
    M(x') are sampled from distributions that are `d_out` apart under the
    `output_measure`. Noise scale is computed appropriately for the specified
    `noise_mechanism` such that the stated privacy property is guaranteed.

    Note:
        `d_out` is interpreted as the "epsilon" parameter if `output_measure` is
        :class:`~.PureDP`, otherwise it is interpreted as the "rho" parameter (if
        `output_measure` is :class:`~.RhoZCDP`).

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP` or
            :class:`~.RhoZCDP`).
        d_out:  Desired distance between output distributions w.r.t. `d_in`. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP` and as
            "rho" if it is :class:`~.RhoZCDP`.
        noise_mechanism: Noise mechanism to apply.
        measure_column: Name to column to compute average of.
        lower: Lower clipping bound for `measure_column`.
        upper: Upper clipping bound for `measure_column`.
        d_in: Distance between inputs under the `input_metric`. The returned
            measurement is guaranteed to have output distributions that are `d_out`
            apart for inputs that are `d_in` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy averages for each group obtained from the groupby transformation.
            If None, this measurement outputs a single number - the noisy average.
        average_column: If a `groupby_transformation` is supplied, this is the column
            name to be used for noisy average in the DataFrame output by the
            measurement. If None, this column will be named "avg(<measure_column>)".
        keep_intermediates: If True, intermediates (noisy sum of deviations and noisy
            count) will also be output in addition to the noisy average.
        sum_column: If a `groupby_transformation` is supplied and `keep_intermediates`
            is True, this is the column name to be used for intermediate sums in the
            DataFrame output by the measurement. If None, this column will be named
            "sum(<measure_column>)".
        count_column: If a `groupby_transformation` is supplied and `keep_intermediates`
            is True, this is the column name to be used for intermediate counts in the
            DataFrame output by the measurement. If None, this column will be named
            "count".
    """
    lower = ExactNumber(lower)
    upper = ExactNumber(upper)
    d_in = ExactNumber(d_in)
    d_out = ExactNumber(d_out)
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
            answers: List[Union[np.int64, np.float64]]
        ) -> Union[np.int64, np.float64, Dict[str, Union[np.int64, np.float64]]]:
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
            df_with_sod_and_count = sod_df.join(count_df, on=groupby.groupby_columns)
        else:
            temp_column = get_nonconflicting_string(sod_df.columns + count_df.columns)
            df_with_sod_and_count = (
                sod_df.withColumn(temp_column, sf.lit(1))
                .join(count_df.withColumn(temp_column, sf.lit(1)), on=[temp_column])
                .drop(temp_column)
            )

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
    output_measure: Union[PureDP, RhoZCDP],
    d_out: ExactNumberInput,
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
) -> PostProcess:
    """Returns a noisy variance measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are `d_in`-close under the `input_metric`, M(x) and
    M(x') are sampled from distributions that are `d_out` apart under the
    `output_measure`. Noise scale is computed appropriately for the specified
    `noise_mechanism` such that the stated privacy property is guaranteed.

    Note:
        `d_out` is interpreted as the "epsilon" parameter if `output_measure` is
        :class:`~.PureDP`, otherwise it is interpreted as the "rho" parameter (if
        `output_measure` is :class:`~.RhoZCDP`).

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP` or
            :class:`~.RhoZCDP`).
        d_out:  Desired distance between output distributions w.r.t. `d_in`. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP` and as
            "rho" if it is :class:`~.RhoZCDP`.
        noise_mechanism: Noise mechanism to apply.
        measure_column: Name to column to compute variance of.
        lower: Lower clipping bound for `measure_column`.
        upper: Upper clipping bound for `measure_column`.
        d_in: Distance between inputs under the `input_metric`. The returned
            measurement is guaranteed to have output distributions that are `d_out`
            apart for inputs that are `d_in` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            a noisy variance for each group obtained from the groupby transformation.
            If None, this measurement outputs a single number - the noisy variance.
        variance_column: If a `groupby_transformation` is supplied, this is
            the column name to be used for noisy variance in the DataFrame
            output by the measurement. If None, this column will be named
            "var(<measure_column>)".
        keep_intermediates: If True, intermediates (noisy sum of deviations, noisy sum
            of squared deviations and noisy count) will also be output in addition to
            the noisy variance.
        sum_of_deviations_column: If a `groupby_transformation` is supplied and
            `keep_intermediates` is True, this is the column name to be used for
            intermediate sums of deviations in the DataFrame output by the measurement.
            If None, this column will be named "sod(<measure_column>)".
        sum_of_squared_deviations_column: If a `groupby_transformation` is supplied
            and `keep_intermediates` is True, this is the column name to be used for
            intermediate sums of squared deviations in the DataFrame output by the
            measurement. If None, this column will be named "sos(<measure_column>)".
        count_column: If a `groupby_transformation` is supplied and `keep_intermediates`
            is True, this is the column name to be used for intermediate counts in the
            DataFrame output by the measurement. If None, this column will be named
            "count".
    """
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
    d_out = ExactNumber(d_out)
    midpoint_of_measure_column, exact_midpoint_of_measure_column = get_midpoint(
        lower,
        upper,
        integer_midpoint=isinstance(
            input_domain[measure_column], SparkIntegerColumnDescriptor
        ),
    )

    lower_after_squaring: ExactNumber = (
        ExactNumber(0) if lower <= 0 <= upper else min(lower ** 2, upper ** 2)
    )
    upper_after_squaring: ExactNumber = max(lower ** 2, upper ** 2)
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
            answers: List[Union[np.int64, np.float64]]
        ) -> Union[np.int64, np.float64, Dict[str, Union[np.int64, np.float64]]]:
            """Computes variance from noisy count and sums of deviations."""
            sod, sos, count = answers
            if count <= 1:
                variance = (
                    midpoint_of_squared_measure_column - midpoint_of_measure_column ** 2
                )
            else:
                variance = (sos / count + midpoint_of_squared_measure_column) - (
                    sod / count + midpoint_of_measure_column
                ) ** 2
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
            df_with_sums_and_count = sod_df.join(
                sos_df, on=groupby.groupby_columns
            ).join(count_df, on=groupby.groupby_columns)
        else:
            temp_column = get_nonconflicting_string(sod_df.columns + count_df.columns)
            df_with_sums_and_count = (
                sod_df.withColumn(temp_column, sf.lit(1))
                .join(sos_df.withColumn(temp_column, sf.lit(1)), on=[temp_column])
                .join(count_df.withColumn(temp_column, sf.lit(1)), on=[temp_column])
                .drop(temp_column)
            )
        df_with_all_columns = df_with_sums_and_count.withColumn(
            variance_column,
            sf.when(
                sf.col(count_column) <= 1,
                sf.lit(midpoint_of_squared_measure_column)
                - sf.lit(midpoint_of_measure_column) ** 2,
            ).otherwise(
                sf.lit(
                    (
                        (
                            sf.col(sum_of_squared_deviations_column)
                            / sf.col(count_column)
                            + sf.lit(midpoint_of_squared_measure_column)
                        )
                        - (
                            sf.col(sum_of_deviations_column) / sf.col(count_column)
                            + sf.lit(midpoint_of_measure_column)
                        )
                        ** 2
                    )
                )
            ),
        ).withColumn(
            variance_column,
            sf.when(sf.col(variance_column) < 0.0, 0.0).otherwise(
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
    output_measure: Union[PureDP, RhoZCDP],
    d_out: ExactNumberInput,
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
) -> PostProcess:
    """Returns a noisy standard deviation measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are `d_in`-close under the `input_metric`, M(x) and
    M(x') are sampled from distributions that are `d_out` apart under the
    `output_measure`. Noise scale is computed appropriately for the specified
    `noise_mechanism` such that the stated privacy property is guaranteed.

    Note:
        `d_out` is interpreted as the "epsilon" parameter if `output_measure` is
        :class:`~.PureDP`, otherwise it is interpreted as the "rho" parameter (if
        `output_measure` is :class:`~.RhoZCDP`).

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (one of :class:`~.PureDP` or
            :class:`~.RhoZCDP`).
        d_out:  Desired distance between output distributions w.r.t. `d_in`. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP` and as
            "rho" if it is :class:`~.RhoZCDP`.
        noise_mechanism: Noise mechanism to apply.
        measure_column: Name to column to compute standard deviation of.
        lower: Lower clipping bound for `measure_column`.
        upper: Upper clipping bound for `measure_column`.
        d_in: Distance between inputs under the `input_metric`. The returned
            measurement is guaranteed to have output distributions that are `d_out`
            apart for inputs that are `d_in` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy standard deviations for each group obtained by applying the groupby
            transformation. If None, this measurement outputs a single number - the
            noisy standard deviation of `measure_column`.
        standard_deviation_column: If a `groupby_transformation` is supplied, this is
            the column name to be used for noisy standard deviation in the DataFrame
            output by the measurement. If None, this column will be named
            "stddev(<measure_column>)".
        keep_intermediates: If True, intermediates (noisy sum of deviations, noisy sum
            of squared deviations noisy count) will also be output in addition to the
            noisy standard deviation.
        sum_of_deviations_column: If a `groupby_transformation` is supplied and
            `keep_intermediates` is True, this is the column name to be used for
            intermediate sums of deviations in the DataFrame output by the measurement.
            If None, this column will be named "sod(<measure_column>)".
        sum_of_squared_deviations_column: If a `groupby_transformation` is supplied
            and `keep_intermediates` is True, this is the column name to be used for
            intermediate sums of squared_deviations in the DataFrame output by the
            measurement. If None, this column will be named "sos(<measure_column>)".
        count_column: If a `groupby_transformation` is supplied and `keep_intermediates`
            is True, this is the column name to be used for intermediate counts in the
            DataFrame output by the measurement. If None, this column will be named
            "count".
    """
    lower = ExactNumber(lower)
    upper = ExactNumber(upper)
    d_in = ExactNumber(d_in)
    d_out = ExactNumber(d_out)
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

        def postprocess_variance(answer):
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
    output_measure: Union[PureDP, RhoZCDP],
    d_out: ExactNumberInput,
    measure_column: str,
    quantile: float,
    lower: Union[int, float],
    upper: Union[int, float],
    d_in: ExactNumberInput = 1,
    groupby_transformation: Optional[GroupBy] = None,
    quantile_column: Optional[str] = None,
) -> PostProcess:
    """Returns a noisy quantile measurement.

    This function constructs a measurement M with the following privacy contract -
    for any two inputs x, x' that are `d_in`-close under the `input_metric`, M(x) and
    M(x') are sampled from distributions that are `d_out` apart under the
    `output_measure`.

    Note:
        `d_out` is interpreted as the "epsilon" parameter if `output_measure` is
        :class:`~.PureDP`, otherwise it is interpreted as the "rho" parameter (if
        `output_measure` is :class:`~.RhoZCDP`).

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Distance metric on input DataFrames.
        output_measure: Desired privacy guarantee (:class:`~.PureDP` or
            :class:`~.RhoZCDP`).
        d_out:  Desired distance between output distributions w.r.t. `d_in`. This is
            interpreted as "epsilon" if output_measure is :class:`~.PureDP` and as
            "rho" if it is :class:`~.RhoZCDP`.
        measure_column: Name to column to compute quantile of.
        quantile: The quantile to produce.
        lower: Lower clipping bound for `measure_column`.
        upper: Upper clipping bound for `measure_column`.
        d_in: Distance between inputs under the `input_metric`. The returned
            measurement is guaranteed to have output distributions that are `d_out`
            apart for inputs that are `d_in` apart. Defaults to 1.
        groupby_transformation: If provided, this measurement returns a DataFrame with
            noisy quantiles for each group obtained by applying groupby.
            If None, this measurement outputs a single number - the noisy quantile.
        quantile_column: If a `groupby_transformation` is supplied, this is
            the column name to be used for noisy quantile in the DataFrame
            output by the measurement. If None, this column will be named
            "q_(<quantile>)_(<measure_column>)".
    """
    d_in = ExactNumber(d_in)
    d_out = ExactNumber(d_out)
    if not quantile_column:
        quantile_column = f"q_({quantile})_({measure_column})"

    postprocess = lambda df: df.withColumnRenamed(measure_column, quantile_column)
    if groupby_transformation is None:
        if isinstance(input_metric, IfGroupedBy):
            raise ValueError(
                "IfGroupedBy must be accompanied by an appropriate groupby "
                "transformation."
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
        raise ValueError(
            "Input metric must match with groupby transformation. Expected:"
            f" ({groupby_transformation.input_metric}), actual: ({input_metric})"
        )
    if groupby_transformation.input_domain != input_domain:
        raise ValueError(
            "Input domain must match with groupby transformation. Expected:"
            f" ({groupby_transformation.input_domain}), actual: ({input_domain})"
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
        ExactNumber(0) if lower <= 0 <= upper else min(lower ** 2, upper ** 2)
    )
    upper_after_squaring: ExactNumber = max(lower ** 2, upper ** 2)
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
