"""Transformations for performing groupby on Spark DataFrames."""
# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from __future__ import annotations

import datetime
import itertools
from functools import reduce
from typing import Any, List, Mapping, Optional, Tuple, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DataType,
    DateType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from typeguard import typechecked

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkGroupedDataFrameDomain,
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
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.validation import validate_groupby_domains


class GroupBy(Transformation):
    """Groups a Spark DataFrame by given group keys.

    Example:
        ..
            >>> from pyspark.sql import SparkSession
            >>> from pyspark.sql.functions import count
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
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
        >>> groupby_B = GroupBy(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ...     use_l2=False,
        ...     group_keys=spark.createDataFrame(
        ...         pd.DataFrame(
        ...             {
        ...                 "B":["b1", "b2"]
        ...             }
        ...         )
        ...     )
        ... )
        >>> # Apply transformation to data
        >>> grouped_dataframe = groupby_B(spark_dataframe)
        >>> counts_df = grouped_dataframe.agg(count("*").alias("count"), fill_value=0)
        >>> print(counts_df.sort("B").toPandas())
            B  count
        0  b1      2
        1  b2      2

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkGroupedDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.HammingDistance`
          or :class:`~.IfGroupedBy` (with inner metric :class:`~.SymmetricDifference`)
        * Output metric - :class:`~.SumOf` or :class:`~.RootSumOfSquared` of
          :class:`~.SymmetricDifference`

        >>> groupby_B.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> groupby_B.output_domain
        SparkGroupedDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)}, group_keys=DataFrame[B: string])
        >>> groupby_B.input_metric
        SymmetricDifference()
        >>> groupby_B.output_metric
        SumOf(inner_metric=SymmetricDifference())

        Stability Guarantee:
            :class:`~.GroupBy`'s :meth:`~stability_function` returns the `d_in` if the
            `input_metric` is :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`, otherwise it returns `d_in` times `2`.

            >>> groupby_B.stability_function(1)
            1
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[HammingDistance, SymmetricDifference, IfGroupedBy],
        use_l2: bool,
        group_keys: DataFrame,
    ):
        """Constructor.

        Args:
            input_domain: Input domain.
            input_metric: Input metric.
            use_l2: If True, use :class:`~.RootSumOfSquared` instead of :class:`~.SumOf`
                in the output metric.
            group_keys: DataFrame where rows correspond to group keys.

        Note:
            `group_keys` must be public.
        """
        output_metric: Union[SumOf, RootSumOfSquared] = (
            RootSumOfSquared(SymmetricDifference())
            if use_l2
            else SumOf(SymmetricDifference())
        )
        if isinstance(input_metric, IfGroupedBy):
            if input_metric.column not in group_keys.columns:
                raise ValueError(
                    f"Must group by IfGroupedBy metric column: {input_metric.column}"
                )
            expected_input_metric = IfGroupedBy(input_metric.column, output_metric)
            if input_metric != expected_input_metric:
                raise ValueError(
                    "Input metric does not have the expected inner metric. "
                    f"Maybe {expected_input_metric}?"
                )
        output_domain = SparkGroupedDataFrameDomain(
            schema=input_domain.schema, group_keys=group_keys
        )
        if not group_keys.count():
            if group_keys.columns:
                raise ValueError(
                    "Group keys cannot have no rows, unless it also has no columns"
                )
        self._group_keys = group_keys
        self._use_l2 = use_l2
        self._groupby_columns = group_keys.columns
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )

    @property
    def use_l2(self) -> bool:
        """Returns whether the output metric will use :class:`~.RootSumOfSquared`."""
        return self._use_l2

    @property
    def group_keys(self):
        """Returns DataFrame containing group keys."""
        return self._group_keys

    @property
    def groupby_columns(self):
        """Returns list of columns to groupby."""
        return self._groupby_columns.copy()

    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under `input_metric`.
        """
        self.input_metric.validate(d_in)
        d_in = ExactNumber(d_in)
        if self.input_metric == HammingDistance():
            return d_in * 2
        return d_in

    def __call__(self, sdf: DataFrame) -> GroupedDataFrame:
        """Performs groupby."""
        return GroupedDataFrame(dataframe=sdf, group_keys=self.group_keys)


# Don't use a type alias for the mapping here;
# you will make our Sphinx jobs fail
def create_groupby_from_column_domains(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    use_l2: bool,
    column_domains: Mapping[
        str,
        Union[
            List[str],
            List[Optional[str]],
            List[int],
            List[Optional[int]],
            List[datetime.date],
            List[Optional[datetime.date]],
        ],
    ],
) -> GroupBy:
    """Returns GroupBy transformation with Cartesian product of column domains as keys.

    Example:
        ..
            >>> from pyspark.sql import SparkSession
            >>> from pyspark.sql.functions import count
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2"],
            ...             "C": ["c1", "c2", "c1", "c1"],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B   C
        0  a1  b1  c1
        1  a2  b1  c2
        2  a3  b2  c1
        3  a3  b2  c1
        >>> groupby_B_C = create_groupby_from_column_domains(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...             "C": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ...     use_l2=False,
        ...     column_domains={
        ...         "B": ["b1", "b2"],
        ...         "C": ["c1", "c2"],
        ...     }
        ... )
        >>> # Apply transformation to data
        >>> grouped_dataframe = groupby_B_C(spark_dataframe)
        >>> groups_df = grouped_dataframe.agg(count("*").alias("count"), fill_value=0)
        >>> print(groups_df.toPandas().sort_values(["B", "C"], ignore_index=True))
            B   C  count
        0  b1  c1      1
        1  b1  c2      1
        2  b2  c1      2
        3  b2  c2      0
        >>> # Note that the group key ("b2", "c2") does not appear in the DataFrame
        >>> # but appears in the aggregation output with the given fill value.

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Metric on input DataFrames.
        use_l2: If True, use :class:`~.RootSumOfSquared` instead of :class:`~.SumOf`
            in the output metric.
        column_domains: Mapping from column name to list of distinct values.

    Note:
        `column_domains` must be public.
    """
    validate_groupby_domains(column_domains, input_domain)
    return GroupBy(
        input_domain=input_domain,
        input_metric=input_metric,
        use_l2=use_l2,
        group_keys=compute_full_domain_df(column_domains),
    )


def create_groupby_from_list_of_keys(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    use_l2: bool,
    groupby_columns: List[str],
    keys: List[Tuple[Union[str, int], ...]],
) -> GroupBy:
    """Returns a GroupBy transformation using user-supplied list of group keys.

    Example:
        ..
            >>> from pyspark.sql import SparkSession
            >>> from pyspark.sql.functions import count
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2"],
            ...             "C": ["c1", "c2", "c1", "c1"],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B   C
        0  a1  b1  c1
        1  a2  b1  c2
        2  a3  b2  c1
        3  a3  b2  c1
        >>> groupby_B_C = create_groupby_from_list_of_keys(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...             "C": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ...     use_l2=False,
        ...     groupby_columns=["B", "C"],
        ...     keys=[("b1", "c1"), ("b2", "c2")]
        ... )
        >>> # Apply transformation to data
        >>> grouped_dataframe = groupby_B_C(spark_dataframe)
        >>> groups_df = grouped_dataframe.agg(count("*").alias("count"), fill_value=0)
        >>> print(groups_df.toPandas().sort_values(["B", "C"], ignore_index=True))
            B   C  count
        0  b1  c1      1
        1  b2  c2      0
        >>> # Note that there is no record corresponding to the key ("b1", "c2")
        >>> # since we did not specify this key while constructing the GroupBy even
        >>> # though this key appears in the input DataFrame.

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Metric on input DataFrames.
        use_l2: If True, use :class:`~.RootSumOfSquared` instead of :class:`~.SumOf`
            in the output metric.
        groupby_columns: List of column names to groupby.
        keys: List of distinct tuples corresponding to group keys.

    Note:
        `keys` must be public list of tuples with no duplicates.
    """
    spark = SparkSession.builder.getOrCreate()
    return GroupBy(
        input_domain=input_domain,
        input_metric=input_metric,
        use_l2=use_l2,
        group_keys=spark.createDataFrame(
            keys, schema=input_domain.project(groupby_columns).spark_schema
        ),
    )


def _spark_type(values: List[Any]) -> DataType:
    """Get the Spark type of a list of values.

    Some of the values might be None.
    """
    if len(values) == 0:
        raise ValueError("Cannot determine type of empty list.")
    for v in values:
        if isinstance(v, str):
            return StringType()
        elif isinstance(v, int):
            return LongType()
        elif isinstance(v, datetime.date) and not isinstance(v, datetime.datetime):
            return DateType()
        elif v is None:
            continue
        else:
            raise ValueError(f"Type {type(v).__qualname__} is not supported")
    raise ValueError("Cannot determine type of list where every entry is None")


# Don't use a type alias for the mapping here;
# you will make our Sphinx jobs fail
def compute_full_domain_df(
    column_domains: Mapping[
        str,
        Union[
            List[str],
            List[Optional[str]],
            List[int],
            List[Optional[int]],
            List[datetime.date],
            List[Optional[datetime.date]],
        ],
    ]
):
    """Returns a DataFrame containing the Cartesian product of given column domains."""
    spark = SparkSession.builder.getOrCreate()
    if not column_domains:
        return SparkSession.builder.getOrCreate().createDataFrame([], StructType())
    full_domain_size = reduce(lambda acc, x: acc * len(x), column_domains.values(), 1)

    domain_spark_types = {
        column: _spark_type(values) for column, values in column_domains.items()
    }
    if full_domain_size <= 10 ** 6:
        # Perform in-memory crossjoin using itertools if fewer than 1m rows
        return spark.createDataFrame(
            spark.sparkContext.parallelize(
                itertools.product(*column_domains.values()),
                numSlices=2 + full_domain_size // 10000,
            ),
            schema=StructType(
                [
                    StructField(column, spark_type)
                    for column, spark_type in domain_spark_types.items()
                ]
            ),
        )
    cores_per_executor = spark.conf.get("spark.executor.cores", "2")
    num_executors = spark.conf.get("spark.executor.instances", "8")
    num_cores = int(cores_per_executor) * int(num_executors) * 3
    full_domain_columns = [
        spark.createDataFrame(
            spark.sparkContext.parallelize(
                [(value,) for value in values], numSlices=2 + len(values) // 10000
            ),
            schema=StructType([StructField(column, domain_spark_types[column])]),
        )
        for column, values in column_domains.items()
    ]

    def crossjoin_with_partition_control(
        left: DataFrame, right: DataFrame
    ) -> DataFrame:
        """Cross Join and repartition DataFrames."""
        joined_df = left.crossJoin(right)
        if joined_df.rdd.getNumPartitions() > num_cores:
            joined_df = joined_df.repartition(num_cores)
        return joined_df

    return reduce(crossjoin_with_partition_control, full_domain_columns)
