"""Functions for truncating Spark DataFrames."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import List

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from tmlt.core.utils.misc import get_nonconflicting_string


def truncate_large_groups(
    df: DataFrame, grouping_columns: List[str], threshold: int
) -> DataFrame:
    """Order rows by a hash function and keep at most `threshold` rows for each group.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import SparkStringColumnDescriptor
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2", "b3"],
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
        4  a3  b3
        >>> print_sdf(truncate_large_groups(spark_dataframe, ["A"], 3))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_sdf(truncate_large_groups(spark_dataframe, ["A"], 2))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> print_sdf(truncate_large_groups(spark_dataframe, ["A"], 1))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        threshold: Maximum number of rows to include for each group.
    """
    rank_column = get_nonconflicting_string(df.columns)
    hash_column = get_nonconflicting_string(df.columns + [rank_column])
    shuffled_partitions = Window.partitionBy(*grouping_columns).orderBy(
        hash_column, *df.columns
    )
    return (
        df.withColumn(hash_column, sf.hash(*df.columns))  # pylint: disable=no-member
        .withColumn(
            rank_column,
            sf.row_number().over(shuffled_partitions),  # pylint: disable=no-member
        )
        .filter(f"{rank_column}<={threshold}")
        .drop(rank_column, hash_column)
    )


def drop_large_groups(
    df: DataFrame, grouping_columns: List[str], threshold: int
) -> DataFrame:
    """Drop all rows for groups that have more than `threshold` rows.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import SparkStringColumnDescriptor
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2", "b3"],
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
        4  a3  b3
        >>> print_sdf(drop_large_groups(spark_dataframe, ["A"], 3))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_sdf(drop_large_groups(spark_dataframe, ["A"], 2))
            A   B
        0  a1  b1
        1  a2  b1
        >>> print_sdf(drop_large_groups(spark_dataframe, ["A"], 1))
            A   B
        0  a1  b1
        1  a2  b1

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        threshold: Threshold for dropping groups. If more than `threshold` rows belong
            to the same group, all rows in that group are dropped.
    """
    count_column = get_nonconflicting_string(df.columns)
    partitions = Window.partitionBy(*grouping_columns)
    return (
        df.withColumn(
            count_column,
            sf.count(sf.lit(1)).over(partitions),  # pylint: disable=no-member
        )
        .filter(f"{count_column}<={threshold}")
        .drop(count_column)
    )


def limit_keys_per_group(
    df: DataFrame, grouping_columns: List[str], key_columns: List[str], threshold: int
) -> DataFrame:
    """Order keys by a hash function and keep at most `threshold` keys for each group.

    .. note::

        After truncation there may still be an unbounded number of rows per key, but
        at most `threshold` keys per group

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import SparkStringColumnDescriptor
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4"],
            ...             "B": ["b1", "b1", "b2", "b2", "b3", "b1", "b2", "b3"],
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
        4  a3  b3
        5  a4  b1
        6  a4  b2
        7  a4  b3
        >>> print_sdf(
        ...     limit_keys_per_group(
        ...         df=spark_dataframe,
        ...         grouping_columns=["A"],
        ...         key_columns=["B"],
        ...         threshold=2,
        ...     )
        ... )
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        5  a4  b2
        6  a4  b3
        >>> print_sdf(
        ...     limit_keys_per_group(
        ...         df=spark_dataframe,
        ...         grouping_columns=["A"],
        ...         key_columns=["B"],
        ...         threshold=1,
        ...     )
        ... )
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a4  b3

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        key_columns: Column defining the keys.
        threshold: Maximum number of keys to include for each group.
    """
    rank_column = get_nonconflicting_string(df.columns)
    hash_column = get_nonconflicting_string(df.columns + [rank_column])
    shuffled_partitions = Window.partitionBy(*grouping_columns).orderBy(hash_column)
    return (
        df.withColumn(
            hash_column, sf.hash(*grouping_columns, *key_columns)
        )  # pylint: disable=no-member
        .withColumn(
            rank_column,
            sf.dense_rank().over(shuffled_partitions),  # pylint: disable=no-member
        )
        .filter(f"{rank_column}<={threshold}")
        .drop(rank_column, hash_column)
    )
