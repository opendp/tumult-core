"""Transformations for filtering Spark DataFrames."""
# TODO(#1320): Add link to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Union

from pyspark.sql import DataFrame, SparkSession
from typeguard import typechecked

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class Filter(Transformation):
    """Keeps only selected rows in a Spark DataFrame using an expression.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import SparkStringColumnDescriptor
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2"],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> # Create the transformation
        >>> filter_transformation = Filter(
        ...     domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     filter_expr="A = 'a1' or B = 'b2'",
        ... )
        >>> # Apply transformation to data
        >>> filtered_spark_dataframe = filter_transformation(spark_dataframe)
        >>> print_sdf(filtered_spark_dataframe)
            A   B
        0  a1  b1
        1  a3  b2
        2  a3  b2

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain` (matches input domain)
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          (matches input metric)

        >>> filter_transformation.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> filter_transformation.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> filter_transformation.input_metric
        SymmetricDifference()
        >>> filter_transformation.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.Filter`'s :meth:`~.stability_function` is the identity function.

            >>> filter_transformation.stability_function(1)
            1
            >>> filter_transformation.stability_function(123)
            123
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, IfGroupedBy],
        filter_expr: str,
    ):
        """Constructor.

        Args:
            filter_expr: A string of SQL expression specifying the filter to apply to the
                data. The language is the same as the one used by
                :meth:`pyspark.sql.DataFrame.filter`.
            domain: Domain of the input/output Spark DataFrames.
            metric: Distance metric for the input and output Spark DataFrames. If the metric
                is :class:`~.IfGroupedBy`, the innermost metric must be
                :class:`~.SymmetricDifference`.
        """
        spark = SparkSession.builder.getOrCreate()
        test_df = spark.createDataFrame([], schema=domain.spark_schema)
        if isinstance(metric, IfGroupedBy):
            if metric.inner_metric not in (
                SymmetricDifference(),
                SumOf(SymmetricDifference()),
                RootSumOfSquared(SymmetricDifference()),
            ):
                raise ValueError(
                    "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                    "SumOf(SymmetricDifference()), or "
                    "RootSumOfSquared(SymmetricDifference())"
                )
            if metric.column not in domain.schema:
                raise ValueError(
                    f"Invalid IfGroupedBy metric: {metric.column} not in domain."
                )
        try:
            test_df.filter(filter_expr)
        except:
            raise ValueError(f"Invalid filter_expr: {filter_expr}.")
        super().__init__(
            input_domain=domain,
            input_metric=metric,
            output_domain=domain,
            output_metric=metric,
        )
        self._filter_expr = filter_expr

    @property
    def filter_expr(self) -> str:
        """Returns the filter expression."""
        return self._filter_expr

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns the filtered DataFrame."""
        return sdf.filter(self._filter_expr)
