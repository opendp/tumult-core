"""Grouped pandas table aware of group keys when performing aggregations.

This is the pandas counterpart of :mod:`tmlt.core.utils.grouped_dataframe`.
:class:`PandasGroupedTable` holds the same two things
:class:`~tmlt.core.utils.grouped_dataframe.GroupedDataFrame` holds -- a table
and an explicit, public set of group keys -- and gives the same guarantee: an
aggregation produces exactly one row per declared group key, whatever the data
contains.

Two properties of that guarantee are load-bearing for differential privacy, and
both are stated as contracts here rather than left to pandas:

* **Which rows come out.** A declared key with no rows still produces a row,
  carrying the aggregation's fill value; a key the data contains but which was
  not declared produces nothing. Neither the presence nor the absence of a row
  in the output therefore reveals anything about the data.
* **What order they come out in.** The output is ordered by the *group keys*,
  which are public, and by nothing else. Two inputs that differ only in the
  order of their rows produce byte-identical output, so row order carries no
  information about the data;
  :func:`~tmlt.core.utils.pandas_grouping.row_keys` and the group-key frame
  decide it entirely.

Grouping is delegated to :mod:`tmlt.core.utils.pandas_grouping`, so two rows
belong to the same group here exactly when Spark would put them in one: a null
and a NaN are different keys, ``-0.0`` and ``0.0`` are one key, and timestamps
group at Spark's microsecond resolution. A bare ``pandas.DataFrame.groupby``
gets all three wrong, and is never used.

The aggregation interface:
    :meth:`PandasGroupedTable.agg` deliberately does not mirror the *signature*
    of :meth:`~tmlt.core.utils.grouped_dataframe.GroupedDataFrame.agg`. Spark's
    takes a :class:`~pyspark.sql.Column`, an expression that carries both the
    computation and, through ``alias``, the name of the column it produces.
    pandas has no such object: the natural pandas equivalent is a plain
    callable over a group's rows, which carries the computation alone. The
    output column's name is therefore a separate argument. See
    :meth:`PandasGroupedTable.agg` for the full contract.

    :meth:`PandasGroupedTable.agg_by_position` is the same aggregation over the
    *positions* of a group's rows rather than over the rows themselves, for the
    aggregations that do not need the values. Spark has no counterpart because
    it needs none: a ``Column`` expression is compiled rather than called, so a
    count there never materializes anything. Here, handing a callable a group's
    rows means copying them out of the table first, which for a count is the
    whole of the cost.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tmlt.core.utils.pandas_grouping import (
    _reindexed_from_zero,
    distinct_rows,
    group_indices,
    row_keys,
)


class PandasGroupedTable:
    """Grouped pandas table implementation supporting explicit group keys.

    A PandasGroupedTable object encapsulates the pandas DataFrame to be grouped
    by as well as the group keys. The output of an aggregation on a
    PandasGroupedTable object is guaranteed to have exactly one row for each
    group key, unless there are no group keys, in which case it will have a
    single row.

    Mutability:
        Neither the table nor the group keys are ever modified: every method
        here builds a new frame. The frames handed to the constructor are held
        by reference and must not be modified by their owner afterwards, as
        described in :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas

        >>> table = PandasGroupedTable(
        ...     dataframe=pd.DataFrame(
        ...         {"A": ["a1", "a1", "a2"], "B": [1, 2, 3]}
        ...     ),
        ...     group_keys=pd.DataFrame({"A": ["a0", "a1"]}),
        ... )
        >>> print_pandas(table.agg(len, fill_value=0, output_column="count"))
            A  count
        0  a0      0
        1  a1      2
    """

    def __init__(self, dataframe: pd.DataFrame, group_keys: Optional[pd.DataFrame]):
        """Constructor.

        Args:
            dataframe: DataFrame to perform groupby on.
            group_keys: DataFrame where each row corresponds to a group key.
                Duplicate rows are silently dropped, under
                :func:`~tmlt.core.utils.pandas_grouping.distinct_rows`' null-safe
                notion of a duplicate. None triggers a total aggregation, as does
                a DataFrame with no columns and no rows. Only the values matter;
                the index is ignored.

        Raises:
            ValueError: If either frame has duplicate column names, if a group
                key column is not a column of ``dataframe``, or if ``group_keys``
                has rows but no columns.
        """
        if len(dataframe.columns) != len(set(dataframe.columns)):
            raise ValueError("DataFrame contains duplicate column names")
        if group_keys is None:
            self._group_keys: Optional[pd.DataFrame] = None
            self._groupby_columns: List[str] = []
        else:
            if len(group_keys.columns) != len(set(group_keys.columns)):
                raise ValueError("Group keys contains duplicate column names")
            invalid_groupby_columns = set(group_keys.columns) - set(dataframe.columns)
            if invalid_groupby_columns:
                raise ValueError(f"Invalid groupby columns: {invalid_groupby_columns}")
            group_keys = distinct_rows(group_keys)
            self._group_keys = group_keys
            self._groupby_columns = list(group_keys.columns)
            # Not `if not group_keys.columns`, which is how the Spark
            # implementation spells this: the truth value of a pandas Index is
            # ambiguous, and asking for it raises.
            if len(group_keys.columns) == 0:
                if len(group_keys) > 0:
                    raise ValueError(
                        "Groupby keys cannot have records without columns."
                    )
                # empty groupkeys means total aggregation
                self._group_keys = None
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns the DataFrame being grouped."""
        return self._dataframe

    @property
    def group_keys(self) -> Optional[pd.DataFrame]:
        """Returns DataFrame containing group keys. None means total aggregation.

        The returned frame is deduplicated and indexed from zero, and its row
        order is the order every aggregation's output is in.
        """
        return self._group_keys

    @property
    def groupby_columns(self) -> List[str]:
        """Returns the columns being grouped on."""
        return self._groupby_columns.copy()

    def select(self, columns: Sequence[str]) -> "PandasGroupedTable":
        """Returns a new PandasGroupedTable with specified subset of columns.

        Note:
            ``columns`` must contain the groupby columns.

        Args:
            columns: List of column names to keep. This must include the groupby
                columns.

        Raises:
            ValueError: If ``columns`` has duplicates, omits a groupby column,
                or names a column the table does not have.
        """
        columns = list(columns)
        if len(set(columns)) != len(columns):
            raise ValueError(f"List contains duplicate column names: {columns}")
        if not set(self.groupby_columns) <= set(columns):
            raise ValueError("Groupby columns must be selected.")
        invalid_columns = [
            column for column in columns if column not in self._dataframe.columns
        ]
        if invalid_columns:
            raise ValueError(f"Invalid columns: {invalid_columns}")
        return PandasGroupedTable(
            dataframe=self._dataframe[columns], group_keys=self.group_keys
        )

    def agg(
        self,
        func: Callable[[pd.DataFrame], Any],
        fill_value: Any,
        output_column: str,
    ) -> pd.DataFrame:
        """Applies given function to each group.

        The output DataFrame is guaranteed to have exactly one row for each
        group key, in the order of :attr:`group_keys`, and indexed from zero.
        For group keys corresponding to empty groups, the output column will
        contain the supplied ``fill_value``; ``func`` is not called for those.
        Groups present in the data but not among the group keys are dropped.

        The output columns are the groupby columns, in :attr:`group_keys`'
        order, followed by ``output_column``. For a total aggregation, where
        there are no groupby columns, the output is a single row holding
        ``output_column`` alone.

        Note:
            The output's row order is a function of the *public* group keys and
            nothing else, so an input's row order cannot leak through it. That
            is why this orders by the group keys rather than by the order the
            groups happen to appear in the data, and why the group keys are
            deduplicated in the constructor rather than here.

        Args:
            func: Function to apply to each non-empty group. It is called with
                that group's rows as a DataFrame carrying *every* column of the
                table -- the groupby columns included, since a value there is
                part of the row -- indexed from zero, in the order the rows
                appear in the table. It must not modify that frame, and must
                return a single value. This is the pandas counterpart of the
                Spark implementation's :class:`~pyspark.sql.Column`; see this
                module's docstring for why the two differ.
            fill_value: Output value for empty groups.
            output_column: Name of the column holding the aggregated values.
                The Spark implementation takes this as part of ``func``, through
                :meth:`~pyspark.sql.Column.alias`, which a pandas callable has
                no equivalent of.
        """
        if self._group_keys is None:
            # Total aggregation. The Spark implementation aggregates and then
            # overwrites the result for an empty input; there is no need to
            # compute a value that is about to be discarded, and not computing
            # it means func is never handed an empty group here either.
            if len(self._dataframe) == 0:
                return pd.DataFrame({output_column: [fill_value]})
            # reset_index rather than _reindexed_from_zero: this is the frame
            # this table holds, not a fresh selection out of it, and reindexing
            # it in place would reindex the caller's frame.
            return pd.DataFrame(
                {output_column: [func(self._dataframe.reset_index(drop=True))]}
            )

        return self._by_group(
            lambda positions: func(self._rows_at(positions)), fill_value, output_column
        )

    def agg_by_position(
        self,
        func: Callable[[np.ndarray], Any],
        fill_value: Any,
        output_column: str,
    ) -> pd.DataFrame:
        """Applies given function to the row positions of each group.

        This is :meth:`agg` for an aggregation that only needs to know *which*
        rows a group holds. It makes every promise :meth:`agg` makes about the
        output -- one row per group key, in :attr:`group_keys`' order, indexed
        from zero, with ``fill_value`` for the keys with no rows -- and differs
        only in what ``func`` is handed.

        The grouping has already computed those positions, so an aggregation
        taking them pays nothing per group beyond its own work, where one taking
        a frame pays for a copy of the group's rows first. See this module's
        docstring for why both entry points exist.

        Args:
            func: Function to apply to each non-empty group. It is called with
                the positions of that group's rows in :attr:`dataframe`, as an
                array in ascending order, and must return a single value. It
                must not modify that array.
            fill_value: Output value for empty groups.
            output_column: Name of the column holding the aggregated values.
        """
        if self._group_keys is None:
            if len(self._dataframe) == 0:
                return pd.DataFrame({output_column: [fill_value]})
            return pd.DataFrame(
                {output_column: [func(np.arange(len(self._dataframe)))]}
            )

        return self._by_group(func, fill_value, output_column)

    def _by_group(
        self,
        func: Callable[[np.ndarray], Any],
        fill_value: Any,
        output_column: str,
    ) -> pd.DataFrame:
        """Returns one row per group key, holding ``func`` of the group's positions.

        This is the body both aggregation methods share, and where the promises
        about the output live: which rows come out, and in what order.

        Args:
            func: The aggregation, over a non-empty group's row positions.
            fill_value: Output value for empty groups.
            output_column: Name of the column holding the aggregated values.
        """
        assert self._group_keys is not None
        positions_by_key = group_indices(self._dataframe, self._groupby_columns)
        values = [
            func(positions_by_key[key]) if key in positions_by_key else fill_value
            for key in row_keys(self._group_keys, self._groupby_columns)
        ]
        output = self._group_keys.copy()
        output[output_column] = _aggregated_column(values, output.index)
        return output.reset_index(drop=True)

    def _rows_at(self, positions: np.ndarray) -> pd.DataFrame:
        """Returns the rows at the given positions, indexed from zero.

        Args:
            positions: The positions of the rows to take.
        """
        return _reindexed_from_zero(self._dataframe.iloc[positions])

    def get_groups(self) -> Dict[Tuple[Any, ...], pd.DataFrame]:
        r"""Returns the groups as a dictionary of DataFrames.

        There is one entry per group key, in :attr:`group_keys`' order,
        including for group keys with no rows, whose value is an empty frame.
        Each frame holds the group's rows, indexed from zero, with the groupby
        columns dropped. A total aggregation has no group keys, and so no
        groups: the returned dictionary is empty, as it is in the Spark
        implementation.

        The keys are the opaque, hashable group keys of
        :func:`~tmlt.core.utils.pandas_grouping.row_keys`, which take the place
        of the Spark implementation's :class:`~pyspark.sql.Row`\\ s. Compare them
        with each other; do not read values back out of them.
        """
        if self._group_keys is None:
            return {}
        non_grouping_columns = [
            column
            for column in self._dataframe.columns
            if column not in self._groupby_columns
        ]
        positions_by_key = group_indices(self._dataframe, self._groupby_columns)
        groups = {}
        for key in row_keys(self._group_keys, self._groupby_columns):
            rows = (
                self._dataframe.iloc[positions_by_key[key]]
                if key in positions_by_key
                else self._dataframe.iloc[:0]
            )
            # Selecting a list of columns is a copy of its own, so the frame
            # reindexed in place here is never one anything else holds.
            groups[key] = _reindexed_from_zero(rows[non_grouping_columns])
        return groups


def concat_rows(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Returns the rows of several frames with the same columns, stacked.

    This is ``pandas.concat`` with the null values left alone.
    ``pandas.concat`` of *frames* rewrites every missing value of an ``object``
    column as ``None``, so a NaN in such a column silently becomes a null and
    two rows :mod:`tmlt.core.utils.pandas_grouping` calls different rows become
    the same row. Concatenating each column as a Series, which this does, has no
    such step. The dtypes are unified as ``pandas.concat`` unifies them.

    Args:
        frames: The frames to stack, at least one, all with the same columns in
            the same order.

    Returns:
        A frame holding every row of every frame, in order, indexed from zero.
    """
    columns = list(frames[0].columns)
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.DataFrame(
        {
            column: pd.concat(
                [frame[column] for frame in frames], ignore_index=True
            ).rename(column)
            for column in columns
        },
        columns=columns,
    )


def _aggregated_column(values: List[Any], index: pd.Index) -> pd.Series:
    """Returns an aggregation's output column, letting pandas infer its dtype.

    An empty list of values has no dtype to infer, and pandas' fallback for one
    has changed between versions; the object dtype is used explicitly instead,
    so that a zero-group aggregation's output is the same on every supported
    pandas. Callers that need a particular dtype -- the count transformations
    in :mod:`tmlt.core.transformations.pandas_transformations.agg`, whose output
    domains fix one -- cast the column afterwards.

    Args:
        values: One value per group, in the group keys' order.
        index: The index to give the column, which must be the group keys'.
    """
    if not values:
        return pd.Series([], index=index, dtype=object)
    return pd.Series(values, index=index)
