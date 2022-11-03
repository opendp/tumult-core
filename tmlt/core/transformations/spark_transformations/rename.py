"""Transformations for renaming Spark DataFrame columns."""
# TODO: Open question regarding "switching" column names.
# TODO(#1320): Add link to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Dict, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
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


class Rename(Transformation):
    """Rename one or more columns in a Spark DataFrame.

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
        >>> rename_b_to_c = Rename(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     rename_mapping={"B": "C"},
        ... )
        >>> # Apply transformation to data
        >>> renamed_spark_dataframe = rename_b_to_c(spark_dataframe)
        >>> print_sdf(renamed_spark_dataframe)
            A   C
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`. Matches input metric, unless :class:`~.IfGroupedBy`
          and the grouping column is renamed.

        >>> rename_b_to_c.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'C': SparkStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c.input_metric
        SymmetricDifference()
        >>> rename_b_to_c.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.Rename`'s :meth:`~.stability_function` returns `d_in`.

            >>> rename_b_to_c.stability_function(1)
            1
            >>> rename_b_to_c.stability_function(2)
            2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        rename_mapping: Dict[str, str],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            metric: Distance metric for input DataFrames.
            rename_mapping: Dictionary from existing column names to target column
                names.
        """
        nonexistent_columns = rename_mapping.keys() - set(input_domain.schema)
        if nonexistent_columns:
            raise ValueError(
                f"Non existent keys in rename_mapping : {nonexistent_columns}"
            )
        for old, new in rename_mapping.items():
            if new in input_domain.schema and new != old:
                raise ValueError(f"Cannot rename {old} to {new}. {new} already exists.")
        output_metric = metric
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
            if metric.column in rename_mapping:
                # If we add support multiple grouping columns, make sure that
                # two grouping columns can't switch names for FilterValue
                output_metric = IfGroupedBy(
                    rename_mapping[metric.column], metric.inner_metric
                )

        output_columns = {
            rename_mapping.get(column, column): input_domain[column]
            for column in input_domain.schema
        }

        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=SparkDataFrameDomain(output_columns),
            output_metric=output_metric,
        )
        self._rename_mapping = rename_mapping.copy()

    @property
    def rename_mapping(self) -> Dict[str, str]:
        """Returns mapping from old column names to new column names."""
        return self._rename_mapping.copy()

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
        """Renames columns."""
        return sdf.select(
            [
                sf.col(c).alias(self._rename_mapping[c])  # pylint: disable=no-member
                if c in self._rename_mapping
                else sf.col(c)  # pylint: disable=no-member
                for c in sdf.columns
            ]
        )
