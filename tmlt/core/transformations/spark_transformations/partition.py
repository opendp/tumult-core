"""Transformations for partitioning Spark DataFrames."""
# TODO(#1320): Add links to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import List, Optional, Tuple, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as sf
from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class Partition(Transformation):
    """Base class for partition transformations."""

    @typechecked
    def __init__(
        self,
        input_domain: Domain,
        input_metric: Union[IfGroupedBy, SymmetricDifference],
        output_metric: Union[SumOf, RootSumOfSquared],
        num_partitions: Optional[int] = None,
    ):
        """Constructor.

        Args:
            input_domain: Domain of inputs to transformation.
            input_metric: Distance metric for inputs.
            output_metric: Metric for output list.
            num_partitions: Number of partitions produced by the transformation.
        """
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=ListDomain(input_domain, length=num_partitions),
            output_metric=output_metric,
        )
        self._num_partitions = num_partitions

    @property
    def num_partitions(self) -> Optional[int]:
        """Returns the number of partitions produced by the transformation.

        If this number is not known, this returns None.
        """
        return self._num_partitions

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)


class PartitionByKeys(Partition):
    """Partition a Spark DataFrame by a list of keys and corresponding domain.

    Example:
        ..
            >>> from pyspark.sql import SparkSession
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkStringColumnDescriptor,
            ...     SparkIntegerColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a1", "a1", "a1", "a2"],
            ...             "B": ["b1", "b1", "b1", "b2", "b2", "b1"],
            ...             "X": [2, 3, 5, -1, 4, -5],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B  X
        0  a1  b1  2
        1  a1  b1  3
        2  a1  b1  5
        3  a1  b2 -1
        4  a1  b2  4
        5  a2  b1 -5
        >>> # notice that ("a2", "b1") is skipped,
        >>> # and that there were no values for ("a2", "b2") in the input data
        >>> list_values = [("a1", "b1"), ("a1", "b2"), ("a2", "b2")]
        >>> # Create the transformation
        >>> partition = PartitionByKeys(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ...     use_l2=False,
        ...     keys=["A", "B"],
        ...     list_values=list_values,
        ... )
        >>> # Apply transformation to data
        >>> partitioned_dataframes = partition(spark_dataframe)
        >>> for list_value, dataframe in zip(list_values, partitioned_dataframes):
        ...     print(list_value)
        ...     print_sdf(dataframe)
        ('a1', 'b1')
            A   B  X
        0  a1  b1  2
        1  a1  b1  3
        2  a1  b1  5
        ('a1', 'b2')
            A   B  X
        0  a1  b2 -1
        1  a1  b2  4
        ('a2', 'b2')
        Empty DataFrame
        Columns: [A, B, X]
        Index: []

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.ListDomain` of :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SumOf` or :class:`~.RootSumOfSquared` of
          :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`

        >>> partition.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> partition.output_domain
        ListDomain(element_domain=SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)}), length=3)
        >>> partition.input_metric
        SymmetricDifference()
        >>> partition.output_metric
        SumOf(inner_metric=SymmetricDifference())

        Stability Guarantee:
            :class:`~.PartitionByKeys`' :meth:`~.stability_function` returns `d_in`.

            >>> partition.stability_function(1)
            1
            >>> partition.stability_function(2)
            2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[IfGroupedBy, SymmetricDifference],
        use_l2: bool,
        keys: List[str],
        list_values: List[Tuple],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrames.
            input_metric: Distance metric for input DataFrames.
            use_l2: If True, use :class:`~.RootSumOfSquared` instead of :class:`~.SumOf`
                in the output metric.
            keys: List of column names to partition by.
            list_values: Domain for key columns in the DataFrame. This is a list
                of unique n-tuples, where each value is a tuple corresponds to a key.
        """
        for key in keys:
            if key not in input_domain.schema:
                raise ValueError(f"Partition key does not exist in input domain: {key}")

        if len(set(list_values)) != len(list_values):
            raise ValueError("Partition key values list contains duplicate.")

        for values in list_values:
            if not len(values) == len(keys):
                raise ValueError(
                    f"Length of values tuple ({len(values)}) does not match length "
                    f"of partition keys ({len(keys)}): {values}"
                )
            for k, v in zip(keys, values):
                if not input_domain[k].valid_py_value(v):
                    raise ValueError(f"Invalid value for partition key {k}: {v}")

        output_metric: Union[SumOf, RootSumOfSquared]
        if isinstance(input_metric, IfGroupedBy):
            if not (
                (isinstance(input_metric.inner_metric, RootSumOfSquared) and use_l2)
                or isinstance(input_metric.inner_metric, SumOf)
                and not use_l2
            ):
                raise ValueError("IfGroupedBy inner metric must match use_l2")
            if input_metric.column in keys:
                output_metric = input_metric.inner_metric
            else:
                output_metric = (
                    RootSumOfSquared(input_metric) if use_l2 else SumOf(input_metric)
                )
        else:
            output_metric = (
                RootSumOfSquared(SymmetricDifference())
                if use_l2
                else SumOf(SymmetricDifference())
            )
        self._partition_keys = keys.copy()
        self._list_values = list_values.copy()
        super().__init__(
            input_domain, input_metric, output_metric, num_partitions=len(list_values)
        )

    @property
    def keys(self) -> List[str]:
        """Returns list of column names to partition on."""
        return self._partition_keys.copy()

    @property
    def list_values(self) -> List[Tuple]:
        """Returns list of values corresponding to the partition keys."""
        return self._list_values.copy()

    def __call__(self, sdf: DataFrame) -> List[DataFrame]:
        """Returns a list of partitions of input DataFrame."""

        def construct_partition_filter(values: Tuple) -> Column:
            """Returns a filter expression for given partition key.

            Args:
                values: Tuple corresponding to a partition key.
            """
            # pylint: disable=no-member
            exp = sf.lit(True)
            for column, value in zip(self._partition_keys, values):
                exp = exp & sf.col(column).eqNullSafe(value)
            return exp

        return [
            sdf.filter(construct_partition_filter(values))
            for values in self._list_values
        ]
