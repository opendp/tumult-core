"""Functions for truncating Spark DataFrames."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import List, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    BinaryType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    TimestampType,
)

from tmlt.core.utils.misc import get_nonconflicting_string


def _hash_column(df: DataFrame, column: str) -> Tuple[DataFrame, str]:
    """Hashes every value in the column.

    Returns:
        The updated DataFrame and the name of the new column.
    """
    new_column = get_nonconflicting_string(df.columns + [column])
    dataType = df.schema[column].dataType
    if (
        dataType == IntegerType()
        or dataType == LongType()
        or dataType == FloatType()
        or dataType == DoubleType()
    ):
        df = df.withColumn(new_column, sf.sha2(sf.bin(column), 256))
    elif dataType == BinaryType() or dataType == StringType():
        df = df.withColumn(new_column, sf.sha2(sf.col(column), 256))
    elif dataType == DateType() or dataType == TimestampType():
        # Casts to a type that can be hashed but doesn't lose information.
        df = df.withColumn(new_column, sf.sha2(sf.col(column).cast("string"), 256))
    else:
        raise NotImplementedError(f"Unsupported data type {dataType}")
    return df, new_column


def _hash_columns(df: DataFrame, columns: List[str]) -> Tuple[DataFrame, str]:
    """Hashes every value in the columns and combines them into a single column.

    Returns:
        The updated DataFrame and the name of the new column.
    """
    # We need to avoid hash collisions when concatenating the columns.
    # If we naively concatenate the columns, we may get the same hash for different
    # rows:
    #   "a," and "b" -> "a,b" -> hash("a,b")
    #   "a" and ",b" -> "a,b" -> hash("a,b")
    # To avoid this, we add a separator to the concatenated string
    #   "a," and "b" -> hash("a,") + "," + hash("b")
    #   "a" and ",b" -> hash("a") + "," + hash(",b")
    # Additionally, the separator we are using won't be contained in the hashed
    # columns, so we don't have to worry about the hashed columns having an analogous
    # problem.
    columns_to_drop = []
    for column in columns:
        df, hashed_column = _hash_column(df, column)
        columns_to_drop.append(hashed_column)
    concatenated_column = get_nonconflicting_string(df.columns)
    # At this point we have handled collisions, but we also need to ensure that the
    # resulting hashes are distributed uniformly.
    # Without handling this:
    #  "a" and "b" -> hash("a") + "," + hash("b")
    #  "a" and "c" -> hash("a") + "," + hash("c")
    #
    # The above values both have the same prefix, and would be nearby after sorting.
    #
    # We can avoid this by hashing again:
    #  "a" and "b" -> hash("a") + "," + hash("b") -> hash(hash("a") + "," + hash("b"))
    #  "a" and "c" -> hash("a") + "," + hash("c") -> hash(hash("a") + "," + hash("c"))
    df = df.withColumn(
        concatenated_column, sf.sha2(sf.concat_ws(",", *columns_to_drop), 256)
    )
    columns_to_drop.append(concatenated_column)
    df, new_column = _hash_column(df, concatenated_column)
    df = df.drop(*columns_to_drop)
    return df, new_column


def truncate_large_groups(
    df: DataFrame, grouping_columns: List[str], threshold: int
) -> DataFrame:
    """Order rows by a hash function and keep at most ``threshold`` rows for each group.

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
        3  a3  b3
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
    starting_columns = list(df.columns)
    row_index_column = get_nonconflicting_string(starting_columns)
    distinct_row_partitions = Window.partitionBy(*starting_columns).orderBy(
        *starting_columns
    )
    df = df.withColumn(row_index_column, sf.row_number().over(distinct_row_partitions))
    df, hash_column = _hash_columns(df, starting_columns + [row_index_column])
    shuffled_partitions = Window.partitionBy(*grouping_columns).orderBy(
        hash_column, *starting_columns
    )
    rank_column = get_nonconflicting_string(df.columns)
    return (
        df.withColumn(
            rank_column,
            sf.row_number().over(shuffled_partitions),
        )
        .filter(f"{rank_column}<={threshold}")
        .drop(rank_column, hash_column, row_index_column)
    )


def drop_large_groups(
    df: DataFrame, grouping_columns: List[str], threshold: int
) -> DataFrame:
    """Drop all rows for groups that have more than ``threshold`` rows.

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
        threshold: Threshold for dropping groups. If more than ``threshold`` rows belong
            to the same group, all rows in that group are dropped.
    """
    count_column = get_nonconflicting_string(df.columns)
    partitions = Window.partitionBy(*grouping_columns)
    return (
        df.withColumn(
            count_column,
            sf.count(sf.lit(1)).over(partitions),
        )
        .filter(f"{count_column}<={threshold}")
        .drop(count_column)
    )


def limit_keys_per_group(
    df: DataFrame, grouping_columns: List[str], key_columns: List[str], threshold: int
) -> DataFrame:
    """Order keys by a hash function and keep at most ``threshold`` keys for each group.

    .. note::

        After truncation there may still be an unbounded number of rows per key, but
        at most ``threshold`` keys per group

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
        2  a3  b3
        3  a4  b3

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        key_columns: Column defining the keys.
        threshold: Maximum number of keys to include for each group.
    """
    df, hash_column = _hash_columns(df, grouping_columns + key_columns)
    shuffled_partitions = Window.partitionBy(*grouping_columns).orderBy(
        hash_column, *key_columns
    )
    rank_column = get_nonconflicting_string(df.columns)
    return (
        df.withColumn(
            rank_column,
            sf.dense_rank().over(shuffled_partitions),
        )
        .filter(f"{rank_column}<={threshold}")
        .drop(rank_column, hash_column)
    )
