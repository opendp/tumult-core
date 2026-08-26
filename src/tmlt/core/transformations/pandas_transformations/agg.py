"""Transformations for aggregating grouped pandas tables.

This is the pandas counterpart of
:mod:`tmlt.core.transformations.spark_transformations.agg`, which currently
covers the two count aggregations. Both mirror their Spark twins exactly,
including their stability guarantees, which are copied from them.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from __future__ import annotations

from typing import Optional, Union, cast

import numpy as np
import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.domains.pandas_domains import (
    PandasGroupedTableDomain,
    PandasIntegerColumnDescriptor,
    PandasTableColumnsDescriptor,
    PandasTableDomain,
)
from tmlt.core.exceptions import UnsupportedMetricError
from tmlt.core.metrics import (
    AbsoluteDifference,
    OnColumn,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable
from tmlt.core.utils.pandas_grouping import group_ids


def _count_rows(positions: np.ndarray) -> int:
    """Returns the number of rows in a group.

    Args:
        positions: The positions of one group's rows, as
            :meth:`~.PandasGroupedTable.agg_by_position` hands them over. A
            count needs nothing else, and asking for the rows themselves would
            mean copying them out of the table first.
    """
    return int(positions.size)


def _row_ids(df: pd.DataFrame) -> np.ndarray:
    """Returns one id per distinct row of a frame, over every column.

    Rows are compared over *every* column, the grouping columns included, and
    with :func:`~tmlt.core.utils.pandas_grouping.group_ids`' notion of equality,
    so that two rows are the same row here exactly when Spark's
    ``size(collect_set(struct("*")))`` counts them once. In particular a row
    holding a null counts, where ``count_distinct`` would drop it, and a null is
    distinct from a NaN.

    Row identity is a property of the whole frame rather than of a group -- two
    rows are the same row, or not, wherever they sit -- so this is computed once
    for the frame and every group reads its own rows' ids out of it. Numbering
    each group's rows separately, which is what asking a group-shaped
    aggregation for its distinct rows would do, repeats the whole comparison per
    group.

    Args:
        df: The frame whose rows are numbered.
    """
    return group_ids(df, list(df.columns))


def _count_distinct_rows(row_ids: np.ndarray, positions: np.ndarray) -> int:
    """Returns the number of distinct rows among a group's rows.

    Args:
        row_ids: The whole frame's row ids, as :func:`_row_ids` returns them.
        positions: The positions of one group's rows.
    """
    return int(np.unique(row_ids[positions]).size)


def _groupby_columns_schema(
    input_domain: PandasGroupedTableDomain,
) -> PandasTableColumnsDescriptor:
    """Returns the descriptors of a grouped domain's groupby columns.

    The columns are in the *domain's* order, which is the order the group keys,
    and so an aggregation's output, must be in. The Spark implementations
    iterate over ``groupby_columns`` itself, which is a frozenset and therefore
    in an order that has nothing to do with either.

    Args:
        input_domain: The domain whose groupby columns are wanted.
    """
    return {
        column: descriptor
        for column, descriptor in input_domain.schema.items()
        if column in input_domain.groupby_columns
    }


class CountGrouped(Transformation):
    r"""Counts the number of records in each group in a :class:`~.PandasGroupedTable`.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasGroupedTableDomain,
            ...     PandasIntegerColumnDescriptor,
            ...     PandasStringColumnDescriptor,
            ... )
            >>> from tmlt.core.metrics import (
            ...     SymmetricDifference,
            ...     SumOf,
            ... )
            >>> from tmlt.core.utils.pandas_grouped_table import (
            ...     PandasGroupedTable,
            ... )
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a1", "a2", "a2"],
            ...         "X": [2, 3, 5, -1],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A  X
        0  a1  2
        1  a1  3
        2  a2 -1
        3  a2  5
        >>> # Specify group keys
        >>> group_keys = pd.DataFrame({"A": ["a0", "a1"]})
        >>> # Note that we have omitted 'a2' from our group keys
        >>> # and included 'a0' which doesn't exist in the DataFrame
        >>> # Create the transformation
        >>> count_by_A = CountGrouped(
        ...     input_domain=PandasGroupedTableDomain(
        ...         schema={
        ...             "A": PandasStringColumnDescriptor(),
        ...             "X": PandasIntegerColumnDescriptor(),
        ...         },
        ...         groupby_columns=["A"],
        ...     ),
        ...     input_metric=SumOf(SymmetricDifference()),
        ... )
        >>> # Create PandasGroupedTable
        >>> grouped_table = PandasGroupedTable(
        ...     dataframe=dataframe,
        ...     group_keys=group_keys,
        ... )
        >>> # Apply transformation to data
        >>> print_pandas(count_by_A(grouped_table))
            A  count
        0  a0      0
        1  a1      2
        >>> # Note that the output does not contain an entry
        >>> # for group key 'a2' but it does contain an entry
        >>> # for group key 'a0'.

    Transformation Contract:
        * Input domain - :class:`~.PandasGroupedTableDomain`
        * Output domain - :class:`~.PandasTableDomain`
        * Input metric - :class:`~.SumOf` or :class:`~.RootSumOfSquared`
          of :class:`~.SymmetricDifference`
        * Output metric - :class:`~.OnColumn`

        >>> count_by_A.input_domain
        PandasGroupedTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'X': PandasIntegerColumnDescriptor(allow_null=False, size=64)}, groupby_columns={'A'})
        >>> count_by_A.output_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'count': PandasIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> count_by_A.input_metric
        SumOf(inner_metric=SymmetricDifference())
        >>> count_by_A.output_metric
        OnColumn(column='count', metric=SumOf(inner_metric=AbsoluteDifference()))

        Stability Guarantee:
            :class:`~.CountGrouped`'s :meth:`~.stability_function` returns ``d_in``.

            >>> count_by_A.stability_function(1)
            1
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: PandasGroupedTableDomain,
        input_metric: Union[SumOf, RootSumOfSquared],
        count_column: Optional[str] = None,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input PandasGroupedTables produced by some
                GroupBy transformation.
            input_metric: Distance metric on inputs.
            count_column: Column name for output group counts. If None, output column
                will be named "count".
        """
        if count_column is None:
            count_column = "count"
        if input_metric.inner_metric != SymmetricDifference():
            raise UnsupportedMetricError(
                input_metric,
                (
                    "Inner metric for the input metric must be SymmetricDifference,"
                    f" not {input_metric.inner_metric}."
                ),
            )
        if count_column in input_domain.groupby_columns:
            raise ValueError(
                f"Invalid count column name: ({count_column}) column already exists"
            )
        output_domain = PandasTableDomain(
            schema={
                **_groupby_columns_schema(input_domain),
                count_column: PandasIntegerColumnDescriptor(),
            }
        )
        output_metric = (
            OnColumn(count_column, SumOf(AbsoluteDifference()))
            if isinstance(input_metric, SumOf)
            else OnColumn(count_column, RootSumOfSquared(AbsoluteDifference()))
        )

        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )
        self._count_column = count_column

    @property
    def input_domain(self) -> PandasGroupedTableDomain:
        """Returns input domain."""
        return cast(PandasGroupedTableDomain, super().input_domain)

    @property
    def count_column(self) -> str:
        """Returns the count column name."""
        return self._count_column

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        d_in = ExactNumber(d_in)
        self.input_metric.validate(d_in)
        return d_in

    def __call__(self, grouped_data: PandasGroupedTable) -> pd.DataFrame:
        """Returns a DataFrame containing counts for each group."""
        result = grouped_data.agg_by_position(
            func=_count_rows, fill_value=0, output_column=self.count_column
        )
        # Ensure the new column has the expected output type. Counts are
        # integers, so this only bites when there are no groups at all, where
        # there is no value for pandas to infer a dtype from.
        return _with_count_dtype(result, self.count_column, self.output_domain)


class CountDistinctGrouped(Transformation):
    r"""Counts distinct records in each group of a :class:`~.PandasGroupedTable`.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasGroupedTableDomain,
            ...     PandasIntegerColumnDescriptor,
            ...     PandasStringColumnDescriptor,
            ... )
            >>> from tmlt.core.metrics import (
            ...     SymmetricDifference,
            ...     SumOf,
            ... )
            >>> from tmlt.core.utils.pandas_grouped_table import (
            ...     PandasGroupedTable,
            ... )
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a1", "a1", "a2", "a2"],
            ...         "X": [2, 2, 3, 5, -1],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A  X
        0  a1  2
        1  a1  2
        2  a1  3
        3  a2 -1
        4  a2  5
        >>> # Specify group keys
        >>> group_keys = pd.DataFrame({"A": ["a0", "a1"]})
        >>> # Note that we have omitted 'a2' from our group keys
        >>> # and included 'a0' which doesn't exist in the DataFrame
        >>> # Create the transformation
        >>> count_distinct_by_A = CountDistinctGrouped(
        ...     input_domain=PandasGroupedTableDomain(
        ...         schema={
        ...             "A": PandasStringColumnDescriptor(),
        ...             "X": PandasIntegerColumnDescriptor(),
        ...         },
        ...         groupby_columns=["A"],
        ...     ),
        ...     input_metric=SumOf(SymmetricDifference()),
        ... )
        >>> # Create PandasGroupedTable
        >>> grouped_table = PandasGroupedTable(
        ...     dataframe=dataframe,
        ...     group_keys=group_keys,
        ... )
        >>> # Apply transformation to data
        >>> print_pandas(count_distinct_by_A(grouped_table))
            A  count_distinct
        0  a0               0
        1  a1               2
        >>> # Note that the output does not contain an entry
        >>> # for group key 'a2' but it does contain an entry
        >>> # for group key 'a0'.

    Transformation Contract:
        * Input domain - :class:`~.PandasGroupedTableDomain`
        * Output domain - :class:`~.PandasTableDomain`
        * Input metric - :class:`~.SumOf` or :class:`~.RootSumOfSquared`
          of :class:`~.SymmetricDifference`
        * Output metric - :class:`~.OnColumn`

        >>> count_distinct_by_A.input_domain
        PandasGroupedTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'X': PandasIntegerColumnDescriptor(allow_null=False, size=64)}, groupby_columns={'A'})
        >>> count_distinct_by_A.output_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'count_distinct': PandasIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> count_distinct_by_A.input_metric
        SumOf(inner_metric=SymmetricDifference())
        >>> count_distinct_by_A.output_metric
        OnColumn(column='count_distinct', metric=SumOf(inner_metric=AbsoluteDifference()))

        Stability Guarantee:
            :class:`~.CountDistinctGrouped`'s :meth:`~.stability_function` returns
            ``d_in``.

            >>> count_distinct_by_A.stability_function(1)
            1
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: PandasGroupedTableDomain,
        input_metric: Union[SumOf, RootSumOfSquared],
        count_column: Optional[str] = None,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input PandasGroupedTables produced by some
                GroupBy transformation.
            input_metric: Distance metric on inputs.
            count_column: Column name for output group counts. If None, output column
                will be named "count_distinct".
        """
        if count_column is None:
            count_column = "count_distinct"
        if input_metric.inner_metric != SymmetricDifference():
            raise UnsupportedMetricError(
                input_metric,
                (
                    "Inner metric for the input metric must be SymmetricDifference,"
                    f" not {input_metric.inner_metric}."
                ),
            )
        if count_column in input_domain.groupby_columns:
            raise ValueError(
                f"Invalid count column name: ({count_column}) column already exists"
            )
        output_domain = PandasTableDomain(
            schema={
                **_groupby_columns_schema(input_domain),
                count_column: PandasIntegerColumnDescriptor(),
            }
        )
        output_metric = (
            OnColumn(count_column, SumOf(AbsoluteDifference()))
            if isinstance(input_metric, SumOf)
            else OnColumn(count_column, RootSumOfSquared(AbsoluteDifference()))
        )

        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )
        self._count_column = count_column

    @property
    def input_domain(self) -> PandasGroupedTableDomain:
        """Returns input domain."""
        return cast(PandasGroupedTableDomain, super().input_domain)

    @property
    def count_column(self) -> str:
        """Returns the count column name."""
        return self._count_column

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        d_in = ExactNumber(d_in)
        self.input_metric.validate(d_in)
        return d_in

    def __call__(self, grouped_data: PandasGroupedTable) -> pd.DataFrame:
        """Returns a DataFrame containing counts for each group."""
        # Note: this cannot use a pandas nunique, which -- like the Spark
        # implementation's count_distinct, and for the same reason -- ignores
        # rows with nulls.
        row_ids = _row_ids(grouped_data.dataframe)
        result = grouped_data.agg_by_position(
            func=lambda positions: _count_distinct_rows(row_ids, positions),
            fill_value=0,
            output_column=self.count_column,
        )
        # Ensure the new column has the expected output type.
        return _with_count_dtype(result, self.count_column, self.output_domain)


def _with_count_dtype(
    df: pd.DataFrame, count_column: str, output_domain: Domain
) -> pd.DataFrame:
    """Returns ``df`` with its count column cast to the output domain's dtype.

    Args:
        df: The aggregation's output.
        count_column: The name of the count column.
        output_domain: The transformation's output domain.
    """
    descriptor = cast(PandasTableDomain, output_domain)[count_column]
    return df.astype({count_column: descriptor.pandas_dtype})
