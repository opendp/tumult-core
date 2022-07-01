"""Transformations for selecting columns from Spark DataFrames."""
# TODO(#1320): Add link to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import List, Union

from pyspark.sql import DataFrame
from typeguard import typechecked

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class Select(Transformation):
    """Keep a subset of columns from a Spark DataFrame.

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
        >>> drop_b = Select(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     columns=["A"],
        ...     metric=SymmetricDifference(),
        ... )
        >>> # Apply transformation to data
        >>> spark_dataframe_without_b = drop_b(spark_dataframe)
        >>> print_sdf(spark_dataframe_without_b)
            A
        0  a1
        1  a2
        2  a3
        3  a3

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
          (matches input metric)

        >>> drop_b.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> drop_b.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False)})
        >>> drop_b.input_metric
        SymmetricDifference()
        >>> drop_b.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.Select`'s :meth:`~.stability_function` returns `d_in`.

            >>> drop_b.stability_function(1)
            1
            >>> drop_b.stability_function(2)
            2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        columns: List[str],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            metric: Distance metric for input and output DataFrames.
            columns: A list of existing column names to keep.
        """
        if len(columns) != len(set(columns)):
            raise ValueError(f"Column name appears more than once in {columns}")
        nonexistent_columns = set(columns) - set(input_domain.schema)
        if nonexistent_columns:
            raise ValueError(
                f"Non existent columns in select columns : {nonexistent_columns}"
            )
        output_columns = {col: input_domain[col] for col in columns}
        if isinstance(metric, IfGroupedBy):
            if metric.column not in columns:
                raise ValueError(
                    "Column used in IfGroupedBy metric must be"
                    f" selected: {metric.column}."
                )
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
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=SparkDataFrameDomain(output_columns),
            output_metric=metric,
        )
        self._columns = columns.copy()

    @property
    def columns(self) -> List[str]:
        """Returns columns being selected."""
        return self._columns.copy()

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
        """Selects columns."""
        return sdf.select(self._columns)
