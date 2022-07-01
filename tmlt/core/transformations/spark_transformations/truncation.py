"""Transformations for truncating Spark DataFrames."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from typing import Type, Union

from pyspark.sql import DataFrame
from typeguard import typechecked

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.truncation import limit_keys_per_group, truncate_large_groups


class LimitRowsPerGroup(Transformation):
    """Keep at most k rows per group.

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
        >>> truncate = LimitRowsPerGroup(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     grouping_column="A",
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
        4  a4  b2
        5  a4  b3

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain` (matches input domain)
        * Input metric - :class:`~.IfGroupedBy` on the grouping column, with inner
          metric :class:`~.SymmetricDifference`
        * Output metric - :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.input_metric
        IfGroupedBy(column='A', inner_metric=SymmetricDifference())
        >>> truncate.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.LimitRowsPerGroup`'s :meth:`~.stability_function` returns
            `threshold * d_in`.

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self, input_domain: SparkDataFrameDomain, grouping_column: str, threshold: int
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            grouping_column: Name of column defining the groups to truncate.
            threshold: The maximum number of rows per group after truncation.
        """
        if threshold < 0:
            raise ValueError("Threshold must be nonnegative")
        self._grouping_column = grouping_column
        self._threshold = threshold
        # super init checks that grouping_column is in the domain
        super().__init__(
            input_domain=input_domain,
            input_metric=IfGroupedBy(grouping_column, SymmetricDifference()),
            output_domain=input_domain,
            output_metric=SymmetricDifference(),
        )

    @property
    def grouping_column(self) -> str:
        """Returns the column defining the groups to truncate."""
        return self._grouping_column

    @property
    def threshold(self) -> int:
        """Returns the maximum number of rows per group after truncation."""
        return self._threshold

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in) * self.threshold

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns a truncated dataframe."""
        return truncate_large_groups(sdf, [self.grouping_column], self.threshold)


class LimitKeysPerGroup(Transformation):
    """Keep at most k keys per group.

    See :func:`~.limit_keys_per_group` for more information about truncation.

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
        >>> truncate = LimitKeysPerGroup(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     grouping_column="A",
        ...     key_column="B",
        ...     threshold=2,
        ...     use_l2=False,
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
        5  a4  b2
        6  a4  b3

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain` (matches input domain)
        * Input metric - :class:`~.IfGroupedBy` on the grouping column, with inner
          metric :class:`~.SymmetricDifference`
        * Output metric - :class:`~.IfGroupedBy` on the key column, with inner
          metric as a :class:`~.SumOf` (`use_l2` is `False`) or
          :class:`~.RootSumOfSquared` (`use_l2` is `True`) over a
          :class:`~.IfGroupedBy` on the grouping column, with inner metric
          :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> truncate.input_metric
        IfGroupedBy(column='A', inner_metric=SymmetricDifference())
        >>> truncate.output_metric
        IfGroupedBy(column='B', inner_metric=SumOf(inner_metric=IfGroupedBy(column='A', inner_metric=SymmetricDifference())))

        Stability Guarantee:
            :class:`~.LimitKeysPerGroup`'s :meth:`~.stability_function` returns
            `threshold * d_in` if `use_l2` is `False` and `sqrt(threshold) * d_in`
            otherwise.

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        grouping_column: str,
        key_column: str,
        threshold: int,
        use_l2: bool,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            grouping_column: Name of column defining the groups to truncate.
            key_column: Name of column defining the keys.
            threshold: The maximum number of rows per group after truncation.
            use_l2: If True, use :class:`~.RootSumOfSquared` as the inner metric
                of the output :class:`~.IfGroupedBy` metric of this transformation
                instead of :class:`~.SumOf`.
        """
        if threshold < 0:
            raise ValueError("Threshold must be nonnegative")
        if grouping_column == key_column:
            raise ValueError("Grouping and key columns must be different")
        self._grouping_column = grouping_column
        self._key_column = key_column
        self._threshold = threshold
        self._use_l2 = use_l2
        lx_class: Union[Type[SumOf], Type[RootSumOfSquared]] = (
            RootSumOfSquared if use_l2 else SumOf
        )
        # super init checks that grouping_column and key_column are in the domain
        super().__init__(
            input_domain=input_domain,
            input_metric=IfGroupedBy(grouping_column, SymmetricDifference()),
            output_domain=input_domain,
            output_metric=IfGroupedBy(
                key_column,
                lx_class(IfGroupedBy(grouping_column, SymmetricDifference())),
            ),
        )

    @property
    def grouping_column(self) -> str:
        """Returns the column defining the groups to truncate."""
        return self._grouping_column

    @property
    def key_column(self) -> str:
        """Returns the column defining the keys."""
        return self._key_column

    @property
    def threshold(self) -> int:
        """Returns the maximum number of keys per group after truncation."""
        return self._threshold

    @property
    def use_l2(self) -> bool:
        """Returns whether the output metric will use :class:`~.RootSumOfSquared`."""
        return self._use_l2

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        d_in = ExactNumber(d_in)
        self.input_metric.validate(d_in)
        if self.use_l2:
            return d_in * self.threshold ** ExactNumber("1/2")
        return d_in * self.threshold

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns a truncated dataframe."""
        return limit_keys_per_group(
            sdf, [self.grouping_column], [self.key_column], self.threshold
        )
