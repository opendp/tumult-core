"""Transformations for casting Spark DataFrame columns to other types."""

# SPDX-License-Identifier: Apache-2.0
# Copyright the Tumult Core Contributors 2026-present

from typing import Collection, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from typeguard import typechecked

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.exceptions import (
    DomainColumnError,
    UnsupportedDomainError,
    UnsupportedMetricError,
)
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import ConciseFrozenSet
from tmlt.core.utils.validation import validate_column_set


class CastToFloat(Transformation):
    """Casts one or more integer columns of a Spark DataFrame to floats.

    The grouping columns of an :class:`~.IfGroupedBy` metric cannot be cast, since
    Tumult Core does not support grouping by floating-point columns.

    :class:`~.CastToFloat` may lose precision for sufficiently large integers; this
    does not impact the stability guarantees under the metrics that this component
    supports. Do not use the same logic for transformations that depend on
    preserving the distance between numeric values, like
    :class:`~.AbsoluteDifference`.

    Example:
        ..
            >>> from pyspark.sql import SparkSession
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3"],
            ...             "B": [1, 2, 3],
            ...         }
            ...     )
            ... )

        >>> print_sdf(spark_dataframe)
            A  B
        0  a1  1
        1  a2  2
        2  a3  3
        >>> cast_b = CastToFloat(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkIntegerColumnDescriptor(),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     columns=["B"],
        ... )
        >>> print_sdf(cast_b(spark_dataframe))
            A    B
        0  a1  1.0
        1  a2  2.0
        2  a3  3.0

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy` (matches input metric)

        >>> cast_b.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkIntegerColumnDescriptor(allow_null=False)})
        >>> cast_b.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkFloatColumnDescriptor(allow_nan=False, allow_inf=False, allow_null=False)})
        >>> cast_b.input_metric
        SymmetricDifference()
        >>> cast_b.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.CastToFloat`'s :meth:`~.stability_function` returns ``d_in``.

            >>> cast_b.stability_function(1)
            1
            >>> cast_b.stability_function(2)
            2
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        columns: Collection[str],
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames.
            metric: Distance metric for the input and output Spark DataFrames.
                If the metric is :class:`~.IfGroupedBy`, its inner metric must be
                :class:`~.SymmetricDifference`, or L1 or L2 over
                :class:`~.SymmetricDifference`, and ``columns`` must not contain
                its grouping columns.
            columns: Integer columns to cast to floats.
        """
        validate_column_set(columns, "columns")
        if isinstance(metric, IfGroupedBy):
            if metric.inner_metric not in (
                SymmetricDifference(),
                SumOf(SymmetricDifference()),
                RootSumOfSquared(SymmetricDifference()),
            ):
                raise UnsupportedMetricError(
                    metric,
                    (
                        "Inner metric for IfGroupedBy metric must be"
                        " SymmetricDifference, SumOf(SymmetricDifference()), or"
                        " RootSumOfSquared(SymmetricDifference())"
                    ),
                )
        if not set(columns) <= set(input_domain.schema):
            raise DomainColumnError(
                input_domain,
                set(columns) - set(input_domain.schema),
                (
                    "One or more columns do not exist in the input domain:"
                    f" {set(columns) - set(input_domain.schema)}"
                ),
            )
        for column in columns:
            if not isinstance(input_domain[column], SparkIntegerColumnDescriptor):
                raise UnsupportedDomainError(
                    input_domain,
                    f"Cannot cast {input_domain[column]} to float. Only integer"
                    " columns can be cast.",
                )

        output_domain = SparkDataFrameDomain(
            {
                column: (
                    SparkFloatColumnDescriptor(
                        allow_nan=False,
                        allow_inf=False,
                        allow_null=descriptor.allow_null,
                    )
                    if column in columns
                    else descriptor
                )
                for column, descriptor in input_domain.schema.items()
            }
        )
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=output_domain,
            output_metric=metric,
        )
        self._columns = ConciseFrozenSet(columns)

    @property
    def columns(self) -> frozenset[str]:
        """Returns the columns to cast to floats."""
        return self._columns

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Casts ``self.columns`` to floats."""
        return sdf.select(
            [
                sf.col(c).cast("double").alias(c) if c in self._columns else sf.col(c)
                for c in sdf.columns
            ]
        )
