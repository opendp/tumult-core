"""Derived measurements for computing noisy aggregates on pandas DataFrames.

This is the pandas counterpart of :mod:`tmlt.core.measurements.aggregations`,
which currently covers the two count measurements. Each factory here has the
same signature as its Spark twin -- same parameter names, in the same order,
with the same defaults -- typed on the pandas domains and transformations, and
the same privacy function: the two backends spend exactly the same budget on the
same query, which is what lets a caller switch backends without re-accounting.

The noise mechanisms and the privacy accounting are shared with the Spark
factories rather than restated:
``tmlt.core.measurements.aggregations._add_noise_to_series`` builds the
same :class:`~.AddNoiseToSeries` from the same
:class:`~tmlt.core.measurements.aggregations.NoiseMechanism`, and
:func:`~tmlt.core.utils.parameters.calculate_noise_scale` computes the scale
from the same distances.

Divergences from the Spark twins
================================

Two, both deliberate:

* :func:`create_count_measurement` validates its ``groupby_transformation``'s
  output domain and metric with the typed errors its Spark twin's
  ``count_distinct`` sibling uses --
  :class:`~tmlt.core.exceptions.UnsupportedDomainError` and
  :class:`~tmlt.core.exceptions.UnsupportedMetricError` -- rather than the bare
  ``assert isinstance(...)`` the Spark ``count`` factory uses. The two Spark
  factories disagree with each other about this; an assert is stripped under
  ``-O``, and reports nothing about what was passed when it is not, so the
  factories here both follow the typed-error one.
* The scalar (no-``groupby_transformation``) path post-processes with a pandas
  idiom rather than the Spark one. See ``_scalar_answer``.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from typing import Any, Callable, Optional, Union

import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.pandas_domains import PandasGroupedTableDomain, PandasTableDomain
from tmlt.core.exceptions import (
    DomainMismatchError,
    MetricMismatchError,
    UnsupportedCombinationError,
    UnsupportedDomainError,
    UnsupportedMetricError,
)
from tmlt.core.measurements.aggregations import NoiseMechanism, _add_noise_to_series
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.converters import PureDPToApproxDP, PureDPToRhoZCDP
from tmlt.core.measurements.pandas_measurements.table import AddNoiseToColumn
from tmlt.core.measurements.postprocess import PostProcess
from tmlt.core.measures import (
    ApproxDP,
    ApproxDPBudget,
    PrivacyBudget,
    PrivacyBudgetInput,
    PureDP,
    RhoZCDP,
)
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.pandas_transformations.agg import (
    CountDistinctGrouped,
    CountGrouped,
)
from tmlt.core.transformations.pandas_transformations.groupby import GroupBy
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.parameters import calculate_noise_scale


def _total_groupby_for_scalar(
    input_domain: PandasTableDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    noise_mechanism: NoiseMechanism,
) -> GroupBy:
    """Validate metric, and build a total aggregation groupby.

    Args:
        input_domain: Domain of input pandas DataFrames.
        input_metric: Distance metric on input DataFrames.
        noise_mechanism: Noise mechanism the counts will be noised with.

    Raises:
        UnsupportedMetricError: If the input metric is an
            :class:`~.IfGroupedBy`, which needs a groupby transformation to
            unwrap it.
    """
    if isinstance(input_metric, IfGroupedBy):
        raise UnsupportedMetricError(
            input_metric,
            (
                "Cannot use IfGroupedBy input metric if no "
                "groupby_transformation is provided."
            ),
        )
    return GroupBy(
        input_domain=input_domain,
        input_metric=input_metric,
        use_l2=noise_mechanism
        in [NoiseMechanism.GAUSSIAN, NoiseMechanism.DISCRETE_GAUSSIAN],
        group_keys=None,
    )


def _scalar_answer(column: str) -> Callable[[pd.DataFrame], Any]:
    """Returns the post-processor reading a scalar count out of a one-row frame.

    The Spark factories post-process a total aggregation with
    ``lambda x: x.head()[column]``, where ``head()`` is Spark's "first
    :class:`~pyspark.sql.Row`" and indexing that Row yields a Python scalar.
    Neither half of that idiom survives translation: a pandas ``head()`` returns
    the first *five rows*, as a DataFrame, and indexing a pandas Series yields a
    numpy scalar. This is the documented carve-out to copying the Spark bodies
    verbatim; that the two return the same answer, of the same Python type, is
    pinned by test rather than by inspection.

    Args:
        column: The name of the count column to read.
    """

    def answer(df: pd.DataFrame) -> Any:
        value = df[column].iloc[0]
        # .item() rather than int()/float(): it gives back whichever Python
        # scalar the numpy one holds, which is the type the Spark Row yields
        # for the same column.
        return value.item() if hasattr(value, "item") else value

    return answer


@typechecked
def create_count_measurement(
    input_domain: PandasTableDomain,
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

    Note:
        This is the pandas counterpart of
        :func:`tmlt.core.measurements.aggregations.create_count_measurement`, and
        has the same privacy function.

    Args:
        input_domain: Domain of input pandas DataFrames.
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

    Raises:
        UnsupportedCombinationError: If the ``output_measure`` is an
            :class:`~.ApproxDP` budget the ``noise_mechanism`` cannot spend.
    """
    if groupby_transformation is None:
        groupby = _total_groupby_for_scalar(input_domain, input_metric, noise_mechanism)
        grouped_count = create_count_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            d_out=d_out,
            noise_mechanism=noise_mechanism,
            d_in=d_in,
            groupby_transformation=groupby,
            count_column=count_column,
        )
        column = "count" if not count_column else count_column
        return PostProcess(grouped_count, _scalar_answer(column))

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
    # The Spark twin asserts these two instead; see this module's docstring.
    if not isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared)):
        raise UnsupportedMetricError(
            groupby_transformation.output_metric,
            (
                "A groupby_transformation for count_measurement must have an "
                "output metric of either SumOf or RootSumOfSquared."
            ),
        )
    if not isinstance(groupby_transformation.output_domain, PandasGroupedTableDomain):
        raise UnsupportedDomainError(
            groupby_transformation.output_domain,
            (
                "A groupby_transformation for count_measurement must have an "
                "output domain of PandasGroupedTableDomain."
            ),
        )

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

    count_aggregation = CountGrouped(
        input_domain=groupby_transformation.output_domain,
        input_metric=groupby_transformation.output_metric,
        count_column=count_column,
    )
    groupby_count = groupby_transformation | count_aggregation
    d_mid = groupby_count.stability_function(d_in)
    noise_scale = calculate_noise_scale(
        d_in=d_mid, d_out=d_out, output_measure=output_measure
    )
    add_noise_to_series = _add_noise_to_series(noise_mechanism, noise_scale)

    assert isinstance(groupby_count.output_domain, PandasTableDomain)
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
    input_domain: PandasTableDomain,
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

    Note:
        This is the pandas counterpart of
        :func:`tmlt.core.measurements.aggregations.create_count_distinct_measurement`,
        and has the same privacy function.

    Args:
        input_domain: Domain of input pandas DataFrames.
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
            measurement. If None, this column will be named "count_distinct".

    Raises:
        UnsupportedCombinationError: If the ``output_measure`` is an
            :class:`~.ApproxDP` budget the ``noise_mechanism`` cannot spend.
    """
    if groupby_transformation is None:
        groupby = _total_groupby_for_scalar(input_domain, input_metric, noise_mechanism)
        groupby_count = create_count_distinct_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            d_out=d_out,
            noise_mechanism=noise_mechanism,
            d_in=d_in,
            groupby_transformation=groupby,
            count_column=count_column,
        )
        column = "count_distinct" if not count_column else count_column
        return PostProcess(groupby_count, _scalar_answer(column))
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
    if not isinstance(groupby_transformation.output_metric, (SumOf, RootSumOfSquared)):
        raise UnsupportedMetricError(
            groupby_transformation.output_metric,
            (
                "A groupby_transformation for count_distinct_measurement must have an "
                "output metric of either SumOf or RootSumOfSquared."
            ),
        )
    if not isinstance(groupby_transformation.output_domain, PandasGroupedTableDomain):
        raise UnsupportedDomainError(
            groupby_transformation.output_domain,
            (
                "A groupby_transformation for count_distinct_measurement must have an "
                "output domain of PandasGroupedTableDomain."
            ),
        )

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

    count_distinct_aggregation = CountDistinctGrouped(
        input_domain=groupby_transformation.output_domain,
        input_metric=groupby_transformation.output_metric,
        count_column=count_column,
    )
    groupby_count_distinct = groupby_transformation | count_distinct_aggregation
    d_mid = groupby_count_distinct.stability_function(d_in)
    noise_scale = calculate_noise_scale(
        d_in=d_mid, d_out=d_out, output_measure=output_measure
    )
    add_noise_to_series = _add_noise_to_series(noise_mechanism, noise_scale)
    assert isinstance(groupby_count_distinct.output_domain, PandasTableDomain)
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
