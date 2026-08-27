"""Transformations for truncating pandas DataFrames.

This is the pandas counterpart of
:mod:`tmlt.core.transformations.spark_transformations.truncation`. Each
transformation takes the same arguments as its Spark twin, rejects the same ones
with the same errors, and has the same stability function; the truncation itself
is delegated to :mod:`tmlt.core.utils.pandas_truncation`, which keeps exactly the
rows the Spark utilities keep.

See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
for more information.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from typing import Collection, Union

import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.pandas_domains import PandasTableDomain
from tmlt.core.exceptions import UnsupportedMetricError
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import ConciseFrozenSet
from tmlt.core.utils.pandas_truncation import (
    limit_keys_per_group,
    truncate_large_groups,
)


class LimitRowsPerGroup(Transformation):
    """Keep at most k rows per group.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitRowsPerGroup`,
    and keeps the same rows it does.

    See :func:`~tmlt.core.utils.pandas_truncation.truncate_large_groups` for more
    information about truncation.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4", "a4"],
            ...         "B": ["b1", "b1", "b2", "b2", "b2", "b1", "b2", "b3", "b4"],
            ...     }
            ... )

        >>> # Example input
        >>> print(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b2
        5  a4  b1
        6  a4  b2
        7  a4  b3
        8  a4  b4
        >>> truncate = LimitRowsPerGroup(
        ...     input_domain=PandasTableDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_metric=SymmetricDifference(),
        ...     grouping_columns=["A"],
        ...     threshold=2,
        ... )
        >>> # Apply transformation to data
        >>> truncated_dataframe = truncate(dataframe)
        >>> print(truncated_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a4  b3
        5  a4  b4

    Transformation Contract:
        * Input domain - :class:`~.PandasTableDomain`
        * Output domain - :class:`~.PandasTableDomain` (matches input domain)
        * Input metric - :class:`~.IfGroupedBy` on the grouping column, with inner
          metric :class:`~.SymmetricDifference`
        * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          on the grouping column, with inner metric :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> truncate.output_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> truncate.input_metric
        IfGroupedBy(columns={'A'}, inner_metric=SymmetricDifference())
        >>> truncate.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.LimitRowsPerGroup` 's :meth:`~.stability_function` returns
            ``threshold * d_in`` if ``output_metric`` is ``SymmetricDifference()`` and
            ``d_in`` otherwise.

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: PandasTableDomain,
        output_metric: Union[SymmetricDifference, IfGroupedBy],
        grouping_columns: Collection[str],
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            output_metric: Distance metric for output DataFrames. This should be
                ``SymmetricDifference()`` or
                ``IfGroupedBy(grouping_columns, SymmetricDifference())``.
            grouping_columns: Names of the columns defining the groups to truncate.
            threshold: The maximum number of rows per group after truncation.
        """
        if threshold < 0:
            raise ValueError("Threshold must be nonnegative")
        self._grouping_columns = ConciseFrozenSet(grouping_columns)
        self._threshold = threshold
        if isinstance(output_metric, IfGroupedBy):
            if (
                output_metric.columns != self.grouping_columns
                or output_metric.inner_metric != SymmetricDifference()
            ):
                raise UnsupportedMetricError(
                    output_metric,
                    (
                        "Output metric must be `SymmetricDifference()` or"
                        f" `IfGroupedBy({grouping_columns}, SymmetricDifference())`, "
                        f"but got: {output_metric}"
                    ),
                )
        # super init checks that grouping_columns is in the domain
        super().__init__(
            input_domain=input_domain,
            input_metric=IfGroupedBy(grouping_columns, SymmetricDifference()),
            output_domain=input_domain,
            output_metric=output_metric,
        )

    @property
    def grouping_columns(self) -> frozenset[str]:
        """Returns the column defining the groups to truncate."""
        return self._grouping_columns

    @property
    def threshold(self) -> int:
        """Returns the maximum number of rows per group after truncation."""
        return self._threshold

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if self.output_metric == SymmetricDifference():
            return ExactNumber(d_in) * self.threshold
        return ExactNumber(d_in)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a truncated dataframe.

        The surviving rows keep the order they arrived in, reindexed from 0; the
        input frame is left untouched.

        Args:
            df: DataFrame to truncate.
        """
        return truncate_large_groups(df, self.grouping_columns, self.threshold)


class LimitKeysPerGroup(Transformation):
    """Keep at most k keys per group.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitKeysPerGroup`,
    and keeps the same rows it does.

    See :func:`~tmlt.core.utils.pandas_truncation.limit_keys_per_group` for more
    information about truncation.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4", "a4"],
            ...         "B": ["b1", "b1", "b2", "b2", "b2", "b1", "b2", "b3", "b4"],
            ...     }
            ... )

        >>> # Example input
        >>> print(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b2
        5  a4  b1
        6  a4  b2
        7  a4  b3
        8  a4  b4
        >>> truncate = LimitKeysPerGroup(
        ...     input_domain=PandasTableDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_metric=IfGroupedBy({"B"}, SumOf(IfGroupedBy({"A"}, SymmetricDifference()))),
        ...     grouping_columns=["A"],
        ...     key_column="B",
        ...     threshold=2,
        ... )
        >>> # Apply transformation to data
        >>> truncated_dataframe = truncate(dataframe)
        >>> print(truncated_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b2
        5  a4  b3
        6  a4  b4

    Transformation Contract:
        * Input domain - :class:`~.PandasTableDomain`
        * Output domain - :class:`~.PandasTableDomain` (matches input domain)
        * Input metric - :class:`~.IfGroupedBy` on the grouping column, with inner
          metric :class:`~.SymmetricDifference`
        * Output metric - :class:`~.IfGroupedBy` on the grouping column, with inner
          metric :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy` on the
          key column, with inner metric as a :class:`~.SumOf` or
          :class:`~.RootSumOfSquared` over a :class:`~.IfGroupedBy` on the grouping
          column, with inner metric :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> truncate.output_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> truncate.input_metric
        IfGroupedBy(columns={'A'}, inner_metric=SymmetricDifference())
        >>> truncate.output_metric
        IfGroupedBy(columns={'B'}, inner_metric=SumOf(inner_metric=IfGroupedBy(columns={'A'}, inner_metric=SymmetricDifference())))

        Stability Guarantee:
            :class:`~.LimitKeysPerGroup` 's :meth:`~.stability_function` returns
            ``d_in`` if ``output_metric`` is ``IfGroupedBy(grouping_columns, SymmetricDifference())``,
            ``sqrt(threshold) * d_in`` if ``output_metric`` is
            ``IfGroupedBy({key_column}, RootSumOfSquared(IfGroupedBy(grouping_columns, SymmetricDifference())))``,
            and ``threshold * d_in`` otherwise.

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: PandasTableDomain,
        output_metric: IfGroupedBy,
        grouping_columns: Collection[str],
        key_column: str,
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            output_metric: Distance metric for output DataFrames. This should be
                ``IfGroupedBy({key_column}, SumOf(IfGroupedBy(grouping_columns, SymmetricDifference())))`` or
                ``IfGroupedBy({key_column}, RootSumOfSquared(IfGroupedBy(grouping_columns, SymmetricDifference())))``
                or ``IfGroupedBy(grouping_columns, SymmetricDifference())``.
            grouping_columns: Names of columns defining the groups to truncate.
            key_column: Name of column defining the keys.
            threshold: The maximum number of keys per group after truncation.
        """  # noqa: E501
        if threshold < 0:
            raise ValueError("Threshold must be nonnegative")
        if key_column in grouping_columns:
            raise ValueError("Key column cannot be a grouping column")
        self._grouping_columns = ConciseFrozenSet(grouping_columns)
        self._key_column = key_column
        self._threshold = threshold
        valid_output_metrics = [
            IfGroupedBy(
                [key_column],
                SumOf(IfGroupedBy(grouping_columns, SymmetricDifference())),
            ),
            IfGroupedBy(
                [key_column],
                RootSumOfSquared(IfGroupedBy(grouping_columns, SymmetricDifference())),
            ),
            IfGroupedBy(grouping_columns, SymmetricDifference()),
        ]
        if output_metric not in valid_output_metrics:
            raise UnsupportedMetricError(
                output_metric,
                (
                    f"Output metric must be one of `IfGroupedBy(['{key_column}'],"
                    f" SumOf(IfGroupedBy({grouping_columns}, SymmetricDifference())))`"
                    f" or `IfGroupedBy(['{key_column}'],"
                    f" RootSumOfSquared(IfGroupedBy({grouping_columns},"
                    f" SymmetricDifference())))` or `IfGroupedBy({grouping_columns},"
                    " SymmetricDifference())`."
                ),
            )
        # super init checks that grouping_columns and key_column are in the domain
        super().__init__(
            input_domain=input_domain,
            input_metric=IfGroupedBy(grouping_columns, SymmetricDifference()),
            output_domain=input_domain,
            output_metric=output_metric,
        )

    @property
    def grouping_columns(self) -> frozenset[str]:
        """Returns the column defining the groups to truncate."""
        return self._grouping_columns

    @property
    def key_column(self) -> str:
        """Returns the column defining the keys."""
        return self._key_column

    @property
    def threshold(self) -> int:
        """Returns the maximum number of keys per group after truncation."""
        return self._threshold

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        d_in = ExactNumber(d_in)
        self.input_metric.validate(d_in)
        if self.output_metric == IfGroupedBy(
            self.grouping_columns, SymmetricDifference()
        ):
            return d_in
        if self.output_metric == IfGroupedBy(
            [self.key_column],
            RootSumOfSquared(IfGroupedBy(self.grouping_columns, SymmetricDifference())),
        ):
            return d_in * self.threshold ** ExactNumber("1/2")
        return d_in * self.threshold

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a truncated dataframe.

        The surviving rows keep the order they arrived in, reindexed from 0; the
        input frame is left untouched.

        Args:
            df: DataFrame to truncate.
        """
        return limit_keys_per_group(
            df, self.grouping_columns, [self.key_column], self.threshold
        )


class LimitRowsPerKeyPerGroup(Transformation):
    """For each group, limit k rows per key.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitRowsPerKeyPerGroup`,
    and keeps the same rows it does.

    See :func:`~tmlt.core.utils.pandas_truncation.truncate_large_groups` for more
    information about truncation.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4", "a4"],
            ...         "B": ["b1", "b1", "b2", "b2", "b2", "b1", "b2", "b3", "b4"],
            ...     }
            ... )

        >>> # Example input
        >>> print(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b2
        5  a4  b1
        6  a4  b2
        7  a4  b3
        8  a4  b4
        >>> truncate = LimitRowsPerKeyPerGroup(
        ...     input_domain=PandasTableDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     input_metric=IfGroupedBy({"B"}, SumOf(IfGroupedBy({"A"}, SymmetricDifference()))),
        ...     grouping_columns=["A"],
        ...     key_column="B",
        ...     threshold=2,
        ... )
        >>> # Apply transformation to data
        >>> truncated_dataframe = truncate(dataframe)
        >>> print(truncated_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a4  b1
        5  a4  b2
        6  a4  b3
        7  a4  b4

    Transformation Contract:
        * Input domain - :class:`~.PandasTableDomain`
        * Output domain - :class:`~.PandasTableDomain` (matches input domain)
        * Input metric - :class:`~.IfGroupedBy` on the grouping column, with inner
          metric :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy` on the key
          column, with inner metric as a :class:`~.SumOf` or :class:`~.RootSumOfSquared`
          over a :class:`~.IfGroupedBy` on the grouping column, with inner metric
          :class:`~.SymmetricDifference`
        * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          on the key column, with inner metric as a :class:`~.RootSumOfSquared`,
          with inner metric :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          on the grouping column, with inner metric :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> truncate.output_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> truncate.input_metric
        IfGroupedBy(columns={'B'}, inner_metric=SumOf(inner_metric=IfGroupedBy(columns={'A'}, inner_metric=SymmetricDifference())))
        >>> truncate.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.LimitRowsPerKeyPerGroup` 's :meth:`~.stability_function` returns
            ``d_in`` if ``input_metric`` is
            ``IfGroupedBy(grouping_columns, SymmetricDifference())``
            and ``threshold * d_in`` otherwise.

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: PandasTableDomain,
        input_metric: IfGroupedBy,
        grouping_columns: Collection[str],
        key_column: str,
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            input_metric: Distance metric for input DataFrames. This should be
                ``IfGroupedBy({key_column}, SumOf(IfGroupedBy(grouping_columns, SymmetricDifference())))`` or
                ``IfGroupedBy({key_column}, RootSumOfSquared(IfGroupedBy(grouping_columns, SymmetricDifference())))``
                or ``IfGroupedBy(grouping_columns, SymmetricDifference())``.
            grouping_columns: Names of columns defining the groups to truncate.
            key_column: Name of column defining the keys.
            threshold: The maximum number of rows each unique (key, grouping column value)
                pair may appear in after truncation.
        """  # noqa: E501
        if threshold < 0:
            raise ValueError("Threshold must be nonnegative")
        if key_column in grouping_columns:
            raise ValueError("Key column cannot be a grouping column")
        self._grouping_columns = ConciseFrozenSet(grouping_columns)
        self._key_column = key_column
        self._threshold = threshold

        output_metric: Union[SymmetricDifference, IfGroupedBy]
        if input_metric == IfGroupedBy(
            [key_column], SumOf(IfGroupedBy(grouping_columns, SymmetricDifference()))
        ):
            output_metric = SymmetricDifference()
        elif input_metric == IfGroupedBy(
            [key_column],
            RootSumOfSquared(IfGroupedBy(grouping_columns, SymmetricDifference())),
        ):
            output_metric = IfGroupedBy(
                [key_column], RootSumOfSquared(SymmetricDifference())
            )
        elif input_metric == IfGroupedBy(grouping_columns, SymmetricDifference()):
            output_metric = input_metric
        else:
            raise UnsupportedMetricError(
                input_metric,
                (
                    f"Input metric must be one of `IfGroupedBy(['{key_column}'],"
                    f" SumOf(IfGroupedBy({grouping_columns}, SymmetricDifference())))`"
                    f" or `IfGroupedBy(['{key_column}'],"
                    f" RootSumOfSquared(IfGroupedBy({grouping_columns},"
                    f" SymmetricDifference())))` or `IfGroupedBy({grouping_columns},"
                    " SymmetricDifference())`"
                ),
            )

        # super init checks that grouping_columns is in the domain
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=input_domain,
            output_metric=output_metric,
        )

    @property
    def grouping_columns(self) -> frozenset[str]:
        """Returns the column defining the groups to truncate."""
        return self._grouping_columns

    @property
    def key_column(self) -> str:
        """Returns the column defining the keys."""
        return self._key_column

    @property
    def threshold(self) -> int:
        """The maximum number of rows per (key, group value) pair after truncation."""
        return self._threshold

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        d_in = ExactNumber(d_in)
        self.input_metric.validate(d_in)
        if self.input_metric == IfGroupedBy(
            self.grouping_columns, SymmetricDifference()
        ):
            return d_in
        return d_in * ExactNumber(self.threshold)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a truncated dataframe.

        The surviving rows keep the order they arrived in, reindexed from 0; the
        input frame is left untouched.

        Args:
            df: DataFrame to truncate.
        """
        return truncate_large_groups(
            df, self.grouping_columns | {self.key_column}, self.threshold
        )
