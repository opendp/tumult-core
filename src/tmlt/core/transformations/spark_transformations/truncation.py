"""Transformations for truncating Spark DataFrames."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026
from typing import Collection, Union

from pyspark.sql import DataFrame
from typeguard import typechecked

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.exceptions import UnsupportedMetricError
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import ConciseFrozenSet
from tmlt.core.utils.truncation import limit_groups_per_id, truncate_large_groups


class LimitRowsPerID(Transformation):
    """Keep at most k rows per ID.

    See :func:`~.truncate_large_groups` for more information about truncation.

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
            ...             "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4", "a4"],
            ...             "B": ["b1", "b1", "b2", "b2", "b2", "b1", "b2", "b3", "b4"],
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
        4  a3  b2
        5  a4  b1
        6  a4  b2
        7  a4  b3
        8  a4  b4
        >>> truncate = LimitRowsPerID(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_metric=SymmetricDifference(),
        ...     id_columns=["A"],
        ...     threshold=2,
        ... )
        >>> # Apply transformation to data
        >>> truncated_spark_dataframe = truncate(spark_dataframe)
        >>> print_sdf(truncated_spark_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a4  b3
        5  a4  b4

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain` (matches input domain)
        * Input metric - :class:`~.IfGroupedBy` on the ID column, with inner
          metric :class:`~.SymmetricDifference`
        * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          on the ID column, with inner metric :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.input_metric
        IfGroupedBy(columns={'A'}, inner_metric=SymmetricDifference())
        >>> truncate.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.LimitRowsPerID` 's :meth:`~.stability_function` returns
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
        input_domain: SparkDataFrameDomain,
        output_metric: Union[SymmetricDifference, IfGroupedBy],
        id_columns: Collection[str],
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            output_metric: Distance metric for output DataFrames. This should be
                ``SymmetricDifference()`` or
                ``IfGroupedBy(id_columns, SymmetricDifference())``.
            id_columns: Names of the columns the contain the IDs for each row.
            threshold: The maximum number of rows per ID after truncation.
        """
        if threshold < 0:
            raise ValueError("Threshold must be nonnegative")
        self._id_columns = ConciseFrozenSet(id_columns)
        self._threshold = threshold
        if isinstance(output_metric, IfGroupedBy):
            if (
                output_metric.columns != self.id_columns
                or output_metric.inner_metric != SymmetricDifference()
            ):
                raise UnsupportedMetricError(
                    output_metric,
                    (
                        "Output metric must be `SymmetricDifference()` or"
                        f" `IfGroupedBy({id_columns}, SymmetricDifference())`, "
                        f"but got: {output_metric}"
                    ),
                )
        # super init checks that id_columns are in the domain
        super().__init__(
            input_domain=input_domain,
            input_metric=IfGroupedBy(id_columns, SymmetricDifference()),
            output_domain=input_domain,
            output_metric=output_metric,
        )

    @property
    def id_columns(self) -> frozenset[str]:
        """Returns the column defining the groups to truncate."""
        return self._id_columns

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

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns a truncated dataframe."""
        return truncate_large_groups(sdf, self.id_columns, self.threshold)


class LimitGroupsPerID(Transformation):
    """Keep at most k groups per ID.

    See :func:`~.limit_groups_per_id` for more information about truncation.

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
            ...             "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4", "a4"],
            ...             "B": ["b1", "b1", "b2", "b2", "b2", "b1", "b2", "b3", "b4"],
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
        4  a3  b2
        5  a4  b1
        6  a4  b2
        7  a4  b3
        8  a4  b4
        >>> truncate = LimitGroupsPerID(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_metric=IfGroupedBy({"B"}, SumOf(IfGroupedBy({"A"}, SymmetricDifference()))),
        ...     id_columns=["A"],
        ...     grouping_column="B",
        ...     threshold=2,
        ... )
        >>> # Apply transformation to data
        >>> truncated_spark_dataframe = truncate(spark_dataframe)
        >>> print_sdf(truncated_spark_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b2
        5  a4  b3
        6  a4  b4

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain` (matches input domain)
        * Input metric - :class:`~.IfGroupedBy` on the ID column, with inner
          metric :class:`~.SymmetricDifference`
        * Output metric - :class:`~.IfGroupedBy` on the ID column, with inner
          metric :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy` on the
          group column, with inner metric as a :class:`~.SumOf` or
          :class:`~.RootSumOfSquared` over a :class:`~.IfGroupedBy` on the ID
          column, with inner metric :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.input_metric
        IfGroupedBy(columns={'A'}, inner_metric=SymmetricDifference())
        >>> truncate.output_metric
        IfGroupedBy(columns={'B'}, inner_metric=SumOf(inner_metric=IfGroupedBy(columns={'A'}, inner_metric=SymmetricDifference())))

        Stability Guarantee:
            :class:`~.LimitGroupsPerID` 's :meth:`~.stability_function` returns
            ``d_in`` if ``output_metric`` is ``IfGroupedBy(id_columns, SymmetricDifference())``,
            ``sqrt(threshold) * d_in`` if ``output_metric`` is
            ``IfGroupedBy({grouping_column}, RootSumOfSquared(IfGroupedBy(id_columns, SymmetricDifference())))``,
            and ``threshold * d_in`` otherwise.

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        output_metric: IfGroupedBy,
        id_columns: Collection[str],
        grouping_column: str,
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            output_metric: Distance metric for output DataFrames. This should be
                ``IfGroupedBy({grouping_column}, SumOf(IfGroupedBy(id_columns, SymmetricDifference())))`` or
                ``IfGroupedBy({grouping_column}, RootSumOfSquared(IfGroupedBy(id_columns, SymmetricDifference())))``
                or ``IfGroupedBy(id_columns, SymmetricDifference())``.
            id_columns: Names of columns defining the ID for each row.
            grouping_column: Name of column defining the groups to truncate.
            threshold: The maximum number of groups per ID after truncation.
        """  # noqa: E501
        if threshold < 0:
            raise ValueError("Threshold must be nonnegative")
        if grouping_column in id_columns:
            raise ValueError("ID column cannot be a grouping column")
        self._id_columns = ConciseFrozenSet(id_columns)
        self._grouping_column = grouping_column
        self._threshold = threshold
        valid_output_metrics = [
            IfGroupedBy(
                [grouping_column],
                SumOf(IfGroupedBy(id_columns, SymmetricDifference())),
            ),
            IfGroupedBy(
                [grouping_column],
                RootSumOfSquared(IfGroupedBy(id_columns, SymmetricDifference())),
            ),
            IfGroupedBy(id_columns, SymmetricDifference()),
        ]
        if output_metric not in valid_output_metrics:
            raise UnsupportedMetricError(
                output_metric,
                (
                    f"Output metric must be one of `IfGroupedBy(['{grouping_column}'],"
                    f" SumOf(IfGroupedBy({id_columns}, SymmetricDifference())))`"
                    f" or `IfGroupedBy(['{grouping_column}'],"
                    f" RootSumOfSquared(IfGroupedBy({id_columns},"
                    f" SymmetricDifference())))` or `IfGroupedBy({id_columns},"
                    " SymmetricDifference())`."
                ),
            )
        # super init checks that id_columns and grouping_column are in the domain
        super().__init__(
            input_domain=input_domain,
            input_metric=IfGroupedBy(id_columns, SymmetricDifference()),
            output_domain=input_domain,
            output_metric=output_metric,
        )

    @property
    def id_columns(self) -> frozenset[str]:
        """Returns the column defining the IDs."""
        return self._id_columns

    @property
    def grouping_column(self) -> str:
        """Returns the column defining the groups to truncate."""
        return self._grouping_column

    @property
    def threshold(self) -> int:
        """Returns the maximum number of groups per ID after truncation."""
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
        if self.output_metric == IfGroupedBy(self.id_columns, SymmetricDifference()):
            return d_in
        if self.output_metric == IfGroupedBy(
            [self.grouping_column],
            RootSumOfSquared(IfGroupedBy(self.id_columns, SymmetricDifference())),
        ):
            return d_in * self.threshold ** ExactNumber("1/2")
        return d_in * self.threshold

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns a truncated dataframe."""
        return limit_groups_per_id(
            sdf, self.id_columns, [self.grouping_column], self.threshold
        )


class LimitRowsPerGroupPerID(Transformation):
    """For each ID, limit k rows per group.

    See :func:`~.truncate_large_groups` for more information about truncation.

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
            ...             "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4", "a4"],
            ...             "B": ["b1", "b1", "b2", "b2", "b2", "b1", "b2", "b3", "b4"],
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
        4  a3  b2
        5  a4  b1
        6  a4  b2
        7  a4  b3
        8  a4  b4
        >>> truncate = LimitRowsPerGroupPerID(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     input_metric=IfGroupedBy({"B"}, SumOf(IfGroupedBy({"A"}, SymmetricDifference()))),
        ...     id_columns=["A"],
        ...     grouping_column="B",
        ...     threshold=2,
        ... )
        >>> # Apply transformation to data
        >>> truncated_spark_dataframe = truncate(spark_dataframe)
        >>> print_sdf(truncated_spark_dataframe)
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
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain` (matches input domain)
        * Input metric - :class:`~.IfGroupedBy` on the ID columns, with inner
          metric :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy` on the group
          column, with inner metric as a :class:`~.SumOf` or :class:`~.RootSumOfSquared`
          over a :class:`~.IfGroupedBy` on the ID columns, with inner metric
          :class:`~.SymmetricDifference`
        * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          on the group column, with inner metric as a :class:`~.RootSumOfSquared`,
          with inner metric :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          on the ID column, with inner metric :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.input_metric
        IfGroupedBy(columns={'B'}, inner_metric=SumOf(inner_metric=IfGroupedBy(columns={'A'}, inner_metric=SymmetricDifference())))
        >>> truncate.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.LimitRowsPerGroupPerID` 's :meth:`~.stability_function` returns
            ``d_in`` if ``input_metric`` is
            ``IfGroupedBy(id_columns, SymmetricDifference())``
            and ``threshold * d_in`` otherwise.

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: IfGroupedBy,
        id_columns: Collection[str],
        grouping_column: str,
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            input_metric: Distance metric for input DataFrames. This should be
                ``IfGroupedBy({grouping_column}, SumOf(IfGroupedBy(id_columns, SymmetricDifference())))`` or
                ``IfGroupedBy({grouping_column}, RootSumOfSquared(IfGroupedBy(id_columns, SymmetricDifference())))``
                or ``IfGroupedBy(id_columns, SymmetricDifference())``.
            id_columns: Names of columns defining the ID for each row.
            grouping_column: Name of column defining the groups to truncate.
            threshold: The maximum number of rows each unique (ID, group)
                pair may appear in after truncation.
        """  # noqa: E501
        if threshold < 0:
            raise ValueError("Threshold must be nonnegative")
        if grouping_column in id_columns:
            raise ValueError("ID column cannot be a grouping column")
        self._id_columns = ConciseFrozenSet(id_columns)
        self._grouping_column = grouping_column
        self._threshold = threshold

        output_metric: Union[SymmetricDifference, IfGroupedBy]
        if input_metric == IfGroupedBy(
            [grouping_column], SumOf(IfGroupedBy(id_columns, SymmetricDifference()))
        ):
            output_metric = SymmetricDifference()
        elif input_metric == IfGroupedBy(
            [grouping_column],
            RootSumOfSquared(IfGroupedBy(id_columns, SymmetricDifference())),
        ):
            output_metric = IfGroupedBy(
                [grouping_column], RootSumOfSquared(SymmetricDifference())
            )
        elif input_metric == IfGroupedBy(id_columns, SymmetricDifference()):
            output_metric = input_metric
        else:
            raise UnsupportedMetricError(
                input_metric,
                (
                    f"Input metric must be one of `IfGroupedBy(['{grouping_column}'],"
                    f" SumOf(IfGroupedBy({id_columns}, SymmetricDifference())))`"
                    f" or `IfGroupedBy(['{grouping_column}'],"
                    f" RootSumOfSquared(IfGroupedBy({id_columns},"
                    f" SymmetricDifference())))` or `IfGroupedBy({id_columns},"
                    " SymmetricDifference())`"
                ),
            )

        # super init checks that id_columns is in the domain
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=input_domain,
            output_metric=output_metric,
        )

    @property
    def id_columns(self) -> frozenset[str]:
        """Returns the columns defining the ID."""
        return self._id_columns

    @property
    def grouping_column(self) -> str:
        """Returns the column defining the groups to truncate."""
        return self._grouping_column

    @property
    def threshold(self) -> int:
        """The maximum number of rows per (ID, group) pair after truncation."""
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
        if self.input_metric == IfGroupedBy(self.id_columns, SymmetricDifference()):
            return d_in
        return d_in * ExactNumber(self.threshold)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns a truncated dataframe."""
        return truncate_large_groups(
            sdf, self.id_columns | {self.grouping_column}, self.threshold
        )
