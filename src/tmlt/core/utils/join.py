"""Utilities related to joining dataframes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import dataclasses
from typing import Dict, List, Optional, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from tmlt.core.utils.misc import escape_column_name, get_nonconflicting_string

# pylint: disable=no-member


def natural_join_columns(
    left_columns: List[str], right_columns: List[str]
) -> List[str]:
    """Returns the columns to join on to perform a natural join.

    The columns to join on are the ones that are in both `left_columns` and
    `right_columns`, in the order they appear in `left_columns`.

    .. note::

        Returns an empty list if there are no common columns.

    Examples:
        >>> natural_join_columns(["a", "b", "c"], ["b", "c", "d"])
        ['b', 'c']
        >>> natural_join_columns(["a", "b", "c"], ["d", "e", "f"])
        []

    Args:
        left_columns: Columns of the left dataframe.
        right_columns: Columns of the right dataframe.
    """
    return [column for column in left_columns if column in right_columns]


@typechecked
def columns_after_join(
    left_columns: List[str], right_columns: List[str], on: Optional[List[str]] = None
) -> Dict[str, Union[Tuple[str, str], Tuple[str, None], Tuple[None, str]]]:
    """Return the expected output columns and their origin from joining two dataframes.

    The keys are the names of the output columns in the order they will appear in the
    output dataframe. The values are tuples of the form (left_column, right_column),
    where left_column and right_column are the names of the columns in the left and
    right dataframes that the output column is derived from. If the output column is
    not derived from a column in the left or right dataframe, the corresponding value
    is None.

    The output columns are ordered as follows:

        - Join columns (in the order given by the user, or the order they are in the
          left table if not provided) appear first.
        - Columns of left table (with _left appended as required) appear
          next in the input order. (excluding join columns)
        - Columns of the right table (with _right appended as required) appear
          last in the input order. (excluding join columns)

    _left and _right are appended to output column names if they are in both
    `left_columns` and `right_columns`, but are not joined on.

    Also does input validation. Checks:

        - The join involves at least one column.
        - Join columns are in both tables.
        - None of the column names are duplicated in any of the inputs.
        - No name collisions when adding _left or _right to a column name.

    Examples:
        >>> columns_after_join(["a", "b", "c"], ["b", "c", "d"])
        {'b': ('b', 'b'), 'c': ('c', 'c'), 'a': ('a', None), 'd': (None, 'd')}
        >>> columns_after_join(["a", "b", "c"], ["b", "c", "d"], ["b"])
        {'b': ('b', 'b'), 'a': ('a', None), 'c_left': ('c', None), 'c_right': (None, 'c'), 'd': (None, 'd')}
        >>> columns_after_join(["a", "b", "c"], ["d", "e", "f"])
        Traceback (most recent call last):
        ...
        ValueError: Join must involve at least one column.
        >>> columns_after_join(["a", "b", "c"], ["a", "b", "c"], ["d"])
        Traceback (most recent call last):
        ...
        ValueError: Join column 'd' not in the left table.
        >>> columns_after_join(["a", "b", "c"], ["a", "b", "c"], ["a", "a"])
        Traceback (most recent call last):
        ...
        ValueError: Join columns (`on`) contain duplicates.
        >>> columns_after_join(["a", "b"], ["a", "b", "b_right"], ["a"])
        Traceback (most recent call last):
        ...
        ValueError: Name collision, 'b_right' would appear more than once in the output.


    Args:
        left_columns: Columns of the left dataframe.
        right_columns: Columns of the right dataframe.
        on: Columns to join on. If None, join on all columns with the same
            name.
    """  # pylint: disable=line-too-long
    if on is None:
        on = natural_join_columns(left_columns, right_columns)
    if len(on) == 0:
        raise ValueError("Join must involve at least one column.")
    for column in on:
        if column not in left_columns:
            raise ValueError(f"Join column '{column}' not in the left table.")
        if column not in right_columns:
            raise ValueError(f"Join column '{column}' not in the right table.")
    if len(set(on)) != len(on):
        raise ValueError("Join columns (`on`) contain duplicates.")
    if len(set(left_columns)) != len(left_columns):
        raise ValueError("Left columns contain duplicates.")
    if len(set(right_columns)) != len(right_columns):
        raise ValueError("Right columns contain duplicates.")

    output_columns: Dict[
        str, Union[Tuple[str, str], Tuple[str, None], Tuple[None, str]]
    ] = {column: (column, column) for column in on}
    for column in left_columns:
        if column in on:
            continue
        if column in right_columns:
            new_column = f"{column}_left"
        else:
            new_column = column
        if new_column in output_columns:
            raise ValueError(
                f"Name collision, '{new_column}' would appear more than once in the"
                " output."
            )
        output_columns[new_column] = (column, None)
    for column in right_columns:
        if column in on:
            continue
        if column in left_columns:
            new_column = f"{column}_right"
        else:
            new_column = column
        if new_column in output_columns:
            raise ValueError(
                f"Name collision, '{new_column}' would appear more than once in the"
                " output."
            )
        output_columns[new_column] = (None, column)
    return output_columns


@typechecked
def domain_after_join(
    left_domain: Domain,
    right_domain: Domain,
    on: Optional[List[str]] = None,
    how: str = "inner",
    nulls_are_equal: bool = False,
) -> SparkDataFrameDomain:
    """Returns the domain of the join of two dataframes.

    Also does input validation. Checks:

        - All checks from :func:`~.columns_after_join`.
        - `how` is one of "left", "right", "inner", or "outer".
        - Join columns have the same data type.
        - Left and right domains are SparkDataFrameDomains.

    .. note::

        This takes into account extra metadata about the columns, such as whether nulls/
        infs are allowed, and what kind of join is performed.

        See :ref:`NaNs, nulls, and infs <special-values>` for more information about
        comparisons involving special values.

    Args:
        left_domain: Domain of the left dataframe.
        right_domain: Domain of the right dataframe.
        on: Columns to join on. If None, join on all columns with the same
            name.
        how: Join type. Must be one of "left", "right", "inner", "outer". This
            defaults to "inner".
        nulls_are_equal: If True, treats null values as equal. Defaults to False.
    """
    if not isinstance(left_domain, SparkDataFrameDomain):
        raise TypeError("Left join input domain must be a SparkDataFrameDomain.")
    if not isinstance(right_domain, SparkDataFrameDomain):
        raise TypeError("Right join input domain must be a SparkDataFrameDomain.")
    if on is None:
        on = natural_join_columns(
            left_columns=list(left_domain.schema),
            right_columns=list(right_domain.schema),
        )
    if how not in ["left", "right", "inner", "outer"]:
        raise ValueError(
            "Join type (`how`) must be one of 'left', 'right', 'inner', or 'outer', not"
            f" '{how}'."
        )
    output_columns = columns_after_join(
        left_columns=list(left_domain.schema),
        right_columns=list(right_domain.schema),
        on=on,
    )
    output_descriptors: Dict[str, SparkColumnDescriptor] = {}
    for output_column, (left_column, right_column) in output_columns.items():
        left_descriptor = left_domain.schema.get(left_column, None)  # type: ignore
        right_descriptor = right_domain.schema.get(right_column, None)  # type: ignore
        if left_descriptor is None:
            assert right_descriptor is not None
            output_descriptors[output_column] = dataclasses.replace(  # type: ignore
                right_descriptor,  # type: ignore
                allow_null=right_descriptor.allow_null or how in ["left", "outer"],
            )
            continue
        if right_descriptor is None:
            assert left_descriptor is not None
            output_descriptors[output_column] = dataclasses.replace(  # type: ignore
                left_descriptor,  # type: ignore
                allow_null=left_descriptor.allow_null or how in ["right", "outer"],
            )
            continue
        assert left_descriptor is not None
        assert right_descriptor is not None
        # The only remaining case is when the output column is a join column.
        assert output_column in on
        if left_descriptor.data_type != right_descriptor.data_type:
            # str(left_descriptor.datatype) changes based on the version of Spark
            left_dtype = str(left_descriptor.data_type).replace("()", "")
            right_dtype = str(right_descriptor.data_type).replace("()", "")
            raise ValueError(
                f"'{output_column}' has different data types in left "
                f"({left_dtype}) and right "
                f"({right_dtype}) domains."
            )
        # All column types are nullable
        allow_null = None
        if how == "left":
            allow_null = left_descriptor.allow_null
        elif how == "right":
            allow_null = right_descriptor.allow_null
        elif how == "inner":
            allow_null = (
                left_descriptor.allow_null and right_descriptor.allow_null
                if nulls_are_equal
                else False
            )
        elif how == "outer":
            allow_null = left_descriptor.allow_null or right_descriptor.allow_null
        assert allow_null is not None
        new_descriptor: SparkColumnDescriptor
        if isinstance(left_descriptor, SparkIntegerColumnDescriptor):
            assert isinstance(right_descriptor, SparkIntegerColumnDescriptor)
            assert left_descriptor.size == right_descriptor.size
            new_descriptor = SparkIntegerColumnDescriptor(
                allow_null=allow_null, size=left_descriptor.size
            )
        elif isinstance(left_descriptor, SparkFloatColumnDescriptor):
            assert isinstance(right_descriptor, SparkFloatColumnDescriptor)
            allow_nan = None
            allow_inf = None
            if how == "left":
                allow_nan = left_descriptor.allow_nan
                allow_inf = left_descriptor.allow_inf
            elif how == "right":
                allow_nan = right_descriptor.allow_nan
                allow_inf = right_descriptor.allow_inf
            elif how == "inner":
                allow_nan = left_descriptor.allow_nan and right_descriptor.allow_nan
                allow_inf = left_descriptor.allow_inf and right_descriptor.allow_inf
            elif how == "outer":
                allow_nan = left_descriptor.allow_nan or right_descriptor.allow_nan
                allow_inf = left_descriptor.allow_inf or right_descriptor.allow_inf
            assert allow_nan is not None
            assert allow_inf is not None
            assert left_descriptor.size == right_descriptor.size
            new_descriptor = SparkFloatColumnDescriptor(
                allow_nan=allow_nan,
                allow_inf=allow_inf,
                allow_null=allow_null,
                size=left_descriptor.size,
            )
        elif isinstance(
            left_descriptor,
            (
                SparkStringColumnDescriptor,
                SparkDateColumnDescriptor,
                SparkTimestampColumnDescriptor,
            ),
        ):
            descriptor_class = left_descriptor.__class__
            assert isinstance(right_descriptor, descriptor_class)
            new_descriptor = descriptor_class(allow_null=allow_null)
        else:
            raise NotImplementedError(
                f"Unsupported column descriptor {left_descriptor}."
            )
        output_descriptors[output_column] = new_descriptor
    return SparkDataFrameDomain(output_descriptors)


def join(
    left: DataFrame,
    right: DataFrame,
    on: Optional[List[str]] = None,
    how: str = "inner",
    nulls_are_equal: bool = False,
) -> DataFrame:
    """Returns the join of two dataframes.

    Args:
        left: Left dataframe.
        right: Right dataframe.
        on: Columns to join on. If None, join on all columns with the same
            name.
        how: Join type. Must be one of "left", "right", "inner", "outer". If None,
            defaults to "inner".
        nulls_are_equal: If True, treats null values as equal. Defaults to False.
    """
    # `columns_after_join` and `domain_after_join` are only called for validation.
    columns_after_join(left_columns=left.columns, right_columns=right.columns, on=on)
    domain_after_join(
        left_domain=SparkDataFrameDomain.from_spark_schema(left.schema),
        right_domain=SparkDataFrameDomain.from_spark_schema(right.schema),
        on=on,
        how=how,
    )
    if nulls_are_equal:
        return _join_where_nulls_are_equal(left=left, right=right, on=on, how=how)
    return _join_where_nulls_are_not_equal(left=left, right=right, on=on, how=how)


def _rename_columns(
    left: DataFrame, right: DataFrame, on: Optional[List[str]] = None
) -> Tuple[
    DataFrame,
    DataFrame,
    Dict[str, Union[Tuple[str, str], Tuple[str, None], Tuple[None, str]]],
]:
    """Rename columns in left and right dataframes to avoid conflicts.

    Example:
        ..
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> left = spark.createDataFrame(
            ...     [
            ...         (1, 2, 3),
            ...         (4, 5, 6),
            ...     ],
            ...     ["a", "b", "c"],
            ... )
            >>> right = spark.createDataFrame(
            ...     [
            ...         (2, 7, 8),
            ...         (5, 9, 10),
            ...     ],
            ...     ["b", "c", "d"],
            ... )

        >>> print_sdf(left)
           a  b  c
        0  1  2  3
        1  4  5  6
        >>> print_sdf(right)
           b  c   d
        0  2  7   8
        1  5  9  10
        >>> left_renamed, right_renamed, output_columns = _rename_columns(
        ...     left, right, on=["b"]
        ... )
        >>> print_sdf(left_renamed)
           a  b  c_left
        0  1  2       3
        1  4  5       6
        >>> print_sdf(right_renamed)
           b  c_right   d
        0  2        7   8
        1  5        9  10
        >>> output_columns
        {'b': ('b', 'b'), 'a': ('a', None), 'c_left': ('c', None), 'c_right': (None, 'c'), 'd': (None, 'd')}

    Args:
        left: Left dataframe.
        right: Right dataframe.
        on: Columns to join on. If None, join on all columns with the same
            name.

    Returns:
        A tuple containing

            * Left dataframe with renamed columns.
            * Right dataframe with renamed columns.
            * Mapping from output column name to
              (left column name, right column name). See :func:`columns_after_join`.
    """  # pylint: disable=line-too-long
    if on is None:
        on = natural_join_columns(
            left_columns=left.columns, right_columns=right.columns
        )
    output_columns = columns_after_join(
        left_columns=left.columns, right_columns=right.columns, on=on
    )
    for output_column, (left_column, right_column) in output_columns.items():
        if left_column is not None and left_column != output_column:
            left = left.withColumnRenamed(left_column, output_column)
        if right_column is not None and right_column != output_column:
            right = right.withColumnRenamed(right_column, output_column)
    return left, right, output_columns


def _join_where_nulls_are_not_equal(
    left: DataFrame,
    right: DataFrame,
    on: Optional[List[str]] = None,
    how: str = "inner",
) -> DataFrame:
    """Returns the join of two dataframes, with null values not being equal.

    Args:
        left: Left dataframe.
        right: Right dataframe.
        on: Columns to join on. If None, join on all columns with the same name.
        how: Join type. Must be one of "left", "right", "inner", "outer". If None,
            defaults to "inner".
    """
    left, right, output_columns = _rename_columns(left=left, right=right, on=on)
    return left.join(right, on=on, how=how).select(
        [escape_column_name(column) for column in output_columns]
    )


def _join_where_nulls_are_equal(
    left: DataFrame,
    right: DataFrame,
    on: Optional[List[str]] = None,
    how: str = "inner",
) -> DataFrame:
    """Returns the join of two dataframes, with null values being equal.

    Args:
        left: Left dataframe.
        right: Right dataframe.
        on: Columns to join on. If None, join on all columns with the same name.
        how: Join type. Must be one of "left", "right", "inner", "outer". If None,
            defaults to "inner".
    """
    left, right, output_columns = _rename_columns(left=left, right=right, on=on)
    if on is None:
        on = natural_join_columns(
            left_columns=left.columns, right_columns=right.columns
        )
    # Rename left and right columns to avoid confusing Spark
    left_temporary_names: Dict[str, str] = {}
    right_temporary_names: Dict[str, str] = {}
    for column in on:
        left_temporary_names[column] = get_nonconflicting_string(
            left.columns
            + right.columns
            + list(left_temporary_names)
            + list(right_temporary_names)
        )
        right_temporary_names[column] = get_nonconflicting_string(
            left.columns
            + right.columns
            + list(left_temporary_names)
            + list(right_temporary_names)
        )
    left = left.select(
        [
            sf.col(column).alias(left_temporary_names[column])
            if column in on
            else sf.col(escape_column_name(column))
            for column in left.columns
        ]
    )
    right = right.select(
        [
            sf.col(column).alias(right_temporary_names[column])
            if column in on
            else sf.col(escape_column_name(column))
            for column in right.columns
        ]
    )
    # Join
    condition = [
        left[left_temporary_names[column]].eqNullSafe(
            right[right_temporary_names[column]]
        )
        for column in on
    ]
    result = left.join(right, on=condition, how=how)
    # Merge columns to use original names
    # Consider joining the following inputs:
    # left: {"A": [1]}
    # right: {"A": [2]}
    # using condition = [left["A"].eqNullSafe(right["A"])]
    # and how = "outer"
    # The result so far will be:
    # {"A": [1, None], "A(2)": [None, 2]}
    # Notice that neither of these columns are [1, 2] which is what we eventually want.
    # We need to merge the columns to get the desired result.
    for column in on:
        left_column = left_temporary_names[column]
        right_column = right_temporary_names[column]
        merged_column = (
            sf.when(sf.col(left_column).isNotNull(), sf.col(left_column))
            .when(sf.col(right_column).isNotNull(), sf.col(right_column))
            .otherwise(sf.lit(None))
        )
        result = result.withColumn(column, merged_column)
    # Drop temporary columns and fix order
    return result.select([escape_column_name(column) for column in output_columns])
