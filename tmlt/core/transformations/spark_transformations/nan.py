"""Transformations to drop or replace NaNs, nulls, and infs in Spark DataFrames."""

# TODO(#1320): Add link to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import warnings
from dataclasses import replace
from functools import reduce
from typing import Any, Dict, List, Tuple, Union, cast

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from typeguard import typechecked

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
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


class DropInfs(Transformation):
    # pylint: disable=line-too-long
    """Drops rows containing +inf or -inf in one or more specified columns.

    Examples:
        ..
            >>> from pyspark.sql import SparkSession
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkFloatColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a4"],
            ...             "B": [0.1, float("-inf"), float("nan"), float("inf")],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A    B
        0  a1  0.1
        1  a2 -inf
        2  a3  NaN
        3  a4  inf
        >>> drop_b_infs = DropInfs(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     columns=["B"],
        ... )
        >>> # Apply transformation to data
        >>> output_dataframe = drop_b_infs(spark_dataframe)
        >>> print_sdf(output_dataframe)
            A    B
        0  a1  0.1
        1  a3  NaN

        Transformation Contract:
            * Input domain - :class:`~.SparkDataFrameDomain`
            * Output domain - :class:`~.SparkDataFrameDomain`
            * Input metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
            * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`

            >>> drop_b_infs.input_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True, allow_null=False, size=64)})
            >>> drop_b_infs.output_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=False, allow_null=False, size=64)})
            >>> drop_b_infs.input_metric
            SymmetricDifference()
            >>> drop_b_infs.output_metric
            SymmetricDifference()

            Stability Guarantee:
                :class:`~.DropInfs`'s :meth:`~.stability_function` returns `d_in`.

                >>> drop_b_infs.stability_function(1)
                1
                >>> drop_b_infs.stability_function(2)
                2
    """
    # pylint: enable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, IfGroupedBy],
        columns: List[str],
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames.
            metric: Distance metric for the input and output Spark DataFrames.
                If the metric is :class:`~.IfGroupedBy`, its inner metric must be
                :class:`~.SymmetricDifference`.
            columns: Columns to drop +inf and -inf from.
        """
        if isinstance(metric, IfGroupedBy) and not (
            isinstance(metric.inner_metric, (SumOf, RootSumOfSquared))
            and isinstance(metric.inner_metric.inner_metric, SymmetricDifference)
            or isinstance(metric.inner_metric, SymmetricDifference)
        ):
            raise ValueError(
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference."
            )
        if not columns:
            raise ValueError("At least one column must be specified.")
        if len(columns) != len(set(columns)):
            duplicates = set(col for col in columns if columns.count(col) > 1)
            raise ValueError(
                "`columns` must not contain duplicate names. The column(s)"
                f" ({duplicates}) appear more than once."
            )
        if not set(columns) <= set(input_domain.schema):
            raise ValueError(
                "One or more columns do not exist in the input domain:"
                f" {set(columns) - set(input_domain.schema)}"
            )
        for column in columns:
            if not isinstance(input_domain[column], SparkFloatColumnDescriptor):
                raise ValueError(
                    f"Cannot drop +inf and -inf from {input_domain[column]}. Only float"
                    " columns can contain +inf or -inf."
                )

        output_domain = SparkDataFrameDomain(
            {
                column: replace(descriptor, allow_inf=False)
                if column in columns
                else descriptor
                for column, descriptor in input_domain.schema.items()
            }
        )
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=output_domain,
            output_metric=metric,
        )
        self._columns = columns.copy()

    @property
    def columns(self) -> List[str]:
        """Returns the columns to check for +inf and -inf."""
        return self._columns.copy()

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Drops rows containing +inf or -inf in `self.columns`."""
        # pylint: disable=no-member
        return sdf.filter(
            reduce(
                lambda exp, column: exp
                & ~(
                    sf.col(column).eqNullSafe(float("-inf"))
                    | sf.col(column).eqNullSafe(float("inf"))
                ),
                self.columns,
                sf.lit(True),
            )
        )
        # pylint: enable=no-member


class DropNaNs(Transformation):
    """Drops rows containing NaNs in one or more specified columns.

    Examples:
        ..
            >>> from pyspark.sql import SparkSession
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkFloatColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a4"],
            ...             "B": [0.1, 1.1, float("nan"), float("inf")],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A    B
        0  a1  0.1
        1  a2  1.1
        2  a3  NaN
        3  a4  inf
        >>> drop_b_nans = DropNaNs(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     columns=["B"],
        ... )
        >>> # Apply transformation to data
        >>> output_dataframe = drop_b_nans(spark_dataframe)
        >>> print_sdf(output_dataframe)
            A    B
        0  a1  0.1
        1  a2  1.1
        2  a4  inf

        Transformation Contract:
            * Input domain - :class:`~.SparkDataFrameDomain`
            * Output domain - :class:`~.SparkDataFrameDomain`
            * Input metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
            * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`

            >>> drop_b_nans.input_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True, allow_null=False, size=64)})
            >>> drop_b_nans.output_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkFloatColumnDescriptor(allow_nan=False, allow_inf=True, allow_null=False, size=64)})
            >>> drop_b_nans.input_metric
            SymmetricDifference()
            >>> drop_b_nans.output_metric
            SymmetricDifference()

            Stability Guarantee:
                :class:`~.DropNaNs`'s :meth:`~.stability_function` returns `d_in`.

                >>> drop_b_nans.stability_function(1)
                1
                >>> drop_b_nans.stability_function(2)
                2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, IfGroupedBy],
        columns: List[str],
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames.
            metric: Distance metric for the input and output Spark DataFrames.
                If the metric is :class:`~.IfGroupedBy`, its inner metric must be
                :class:`~.SumOf` or :class:`~.RootSumOfSquared` over
                :class:`~.SymmetricDifference`.
            columns: Columns to drop NaNs from.
        """
        if isinstance(metric, IfGroupedBy) and not (
            isinstance(metric.inner_metric, (SumOf, RootSumOfSquared))
            and isinstance(metric.inner_metric.inner_metric, SymmetricDifference)
            or isinstance(metric.inner_metric, SymmetricDifference)
        ):
            raise ValueError(
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference."
            )
        if not columns:
            raise ValueError("At least one column must be specified.")

        if len(columns) != len(set(columns)):
            duplicates = set(col for col in columns if columns.count(col) > 1)
            raise ValueError(
                f"`columns` must not contain duplicate names. Columns ({duplicates})"
                " appear more than once."
            )

        if not set(columns) <= set(input_domain.schema):
            raise ValueError(
                "One or more columns do not exist in the input domain"
                f" {set(columns)-set(input_domain.schema)}"
            )

        for column in columns:
            if not isinstance(input_domain[column], SparkFloatColumnDescriptor):
                raise ValueError(
                    f"Cannot drop NaNs from {input_domain[column]}. Only float columns"
                    " can contain NaNs."
                )
        output_domain = SparkDataFrameDomain(
            {
                column: replace(descriptor, allow_nan=False)
                if column in columns
                else descriptor
                for column, descriptor in input_domain.schema.items()
            }
        )
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=output_domain,
            output_metric=metric,
        )
        self._columns = columns.copy()

    @property
    def columns(self) -> List[str]:
        """Returns the columns to check for NaNs."""
        return self._columns.copy()

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Drops rows containing NaNs in `self.columns`."""
        # pylint: disable=no-member
        return sdf.filter(
            reduce(
                lambda exp, column: exp & ~sf.isnan(sf.col(column)),
                self.columns,
                sf.lit(True),
            )
        )


class DropNulls(Transformation):
    """Drops rows containing nulls in one or more specified columns.

    Examples:
        ..
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkFloatColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     [("a1" , 0.1), ("a2", None), (None, float("nan"))],
            ...     schema=["A", "B"]
            ... )

        >>> # Example input
        >>> spark_dataframe.sort("A").show()
        +----+----+
        |   A|   B|
        +----+----+
        |null| NaN|
        |  a1| 0.1|
        |  a2|null|
        +----+----+
        <BLANKLINE>
        >>> drop_b_nulls = DropNulls(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(allow_null=True),
        ...             "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     columns=["B"],
        ... )
        >>> # Apply transformation to data
        >>> output_dataframe = drop_b_nulls(spark_dataframe)
        >>> output_dataframe.sort("A").show()
        +----+---+
        |   A|  B|
        +----+---+
        |null|NaN|
        |  a1|0.1|
        +----+---+
        <BLANKLINE>

        Transformation Contract:
            * Input domain - :class:`~.SparkDataFrameDomain`
            * Output domain - :class:`~.SparkDataFrameDomain`
            * Input metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
            * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`

            >>> drop_b_nulls.input_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=True), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=False, allow_null=True, size=64)})
            >>> drop_b_nulls.output_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=True), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=False, allow_null=False, size=64)})
            >>> drop_b_nulls.input_metric
            SymmetricDifference()
            >>> drop_b_nulls.output_metric
            SymmetricDifference()

            Stability Guarantee:
                :class:`~.DropNulls`'s :meth:`~.stability_function` returns `d_in`.

                >>> drop_b_nulls.stability_function(1)
                1
                >>> drop_b_nulls.stability_function(2)
                2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, IfGroupedBy],
        columns: List[str],
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames.
            metric: Distance metric for the input and output Spark DataFrames.
                If the metric is :class:`~.IfGroupedBy`, its inner metric must be
                :class:`~.SumOf` or :class:`~.RootSumOfSquared` over
                :class:`~.SymmetricDifference`.
            columns: Columns to drop nulls from.
        """
        if isinstance(metric, IfGroupedBy) and not (
            isinstance(metric.inner_metric, (SumOf, RootSumOfSquared))
            and isinstance(metric.inner_metric.inner_metric, SymmetricDifference)
            or isinstance(metric.inner_metric, SymmetricDifference)
        ):
            raise ValueError(
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference."
            )
        if not columns:
            raise ValueError("At least one column must be specified.")

        if len(columns) != len(set(columns)):
            duplicates = set(col for col in columns if columns.count(col) > 1)
            raise ValueError(
                f"`columns` must not contain duplicate names. Columns ({duplicates})"
                " appear more than once."
            )

        if not set(columns) <= set(input_domain.schema):
            raise ValueError(
                "One or more columns do not exist in the input domain"
                f" {set(columns)-set(input_domain.schema)}"
            )
        output_domain = SparkDataFrameDomain(
            {
                column: replace(descriptor, allow_null=False)
                if column in columns
                else descriptor
                for column, descriptor in input_domain.schema.items()
            }
        )
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=output_domain,
            output_metric=metric,
        )
        self._columns = columns.copy()

    @property
    def columns(self) -> List[str]:
        """Returns the columns to check for nulls."""
        return self._columns.copy()

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Drops rows containing nulls in `self.columns`."""
        # pylint: disable=no-member
        return sdf.filter(
            reduce(
                lambda exp, column: exp & ~sf.isnull(sf.col(column)),
                self.columns,
                sf.lit(True),
            )
        )


class ReplaceInfs(Transformation):
    # pylint: disable=line-too-long
    """Replaces +inf and -inf in one or more specified columns.

    Examples:
        ..
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkFloatColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     [("a1" , float("inf")), ("a2", None), (None, float("nan")), ("a3", float("-inf"))],
            ...     schema=["A", "B"]
            ... )

        >>> # Example input
        >>> spark_dataframe.sort("A").show()
        +----+---------+
        |   A|        B|
        +----+---------+
        |null|      NaN|
        |  a1| Infinity|
        |  a2|     null|
        |  a3|-Infinity|
        +----+---------+
        <BLANKLINE>
        >>> replace_infs = ReplaceInfs(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(allow_null=True),
        ...             "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True, allow_inf=True),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     replace_map={"B": (-100.0, 100.0)},
        ... )
        >>> # Apply transformation to data
        >>> output_dataframe = replace_infs(spark_dataframe)
        >>> output_dataframe.sort("A").show()
        +----+------+
        |   A|     B|
        +----+------+
        |null|   NaN|
        |  a1| 100.0|
        |  a2|  null|
        |  a3|-100.0|
        +----+------+
        <BLANKLINE>

        Transformation Contract:
            * Input domain - :class:`~.SparkDataFrameDomain`
            * Output domain - :class:`~.SparkDataFrameDomain`
            * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`, or :class:`~.IfGroupedBy`
            * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`, or :class:`~.IfGroupedBy`

            >>> replace_infs.input_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=True), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True, allow_null=True, size=64)})
            >>> replace_infs.output_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=True), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=False, allow_null=True, size=64)})
            >>> replace_infs.input_metric
            SymmetricDifference()
            >>> replace_infs.output_metric
            SymmetricDifference()

            Stability Guarantee:
                :class:`~.DropNulls`'s :meth:`~.stability_function` returns `d_in`.

                >>> replace_infs.stability_function(1)
                1
                >>> replace_infs.stability_function(2)
                2
    """
    # pylint: enable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        replace_map: Dict[str, Tuple[float, float]],
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames.
            metric: Distance metric for the input and output Spark DataFrames.
            replace_map: Dictionary mapping column names to a tuple. The first
                value in the tuple will be used to replace -inf in that column,
                and the second value in the tuple will be used to replace +inf
                in that column.
        """
        if isinstance(metric, IfGroupedBy) and not (
            isinstance(metric.inner_metric, (SumOf, RootSumOfSquared))
            and isinstance(metric.inner_metric.inner_metric, SymmetricDifference)
            or isinstance(metric.inner_metric, SymmetricDifference)
        ):
            raise ValueError(
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference."
            )
        if not replace_map:
            raise ValueError(
                "At least one column must be specified (in `replace_map`)."
            )

        if not set(replace_map) <= set(input_domain.schema):
            raise ValueError(
                "One or more columns do not exist in the input domain:"
                f" {set(replace_map) - set(input_domain.schema)}"
            )
        for column in list(replace_map.keys()):
            if not isinstance(input_domain[column], SparkFloatColumnDescriptor):
                raise ValueError(
                    f"Column of type {input_domain[column]} cannot contain values of"
                    " infinity or -infinity."
                )
            if not cast(SparkFloatColumnDescriptor, input_domain[column]).allow_inf:
                warnings.warn(
                    f"Column ({column}) already disallows infinite values. This"
                    " transformation will have no effect on this column.",
                    RuntimeWarning,
                )
        output_domain = SparkDataFrameDomain(
            {
                column: replace(descriptor, allow_inf=False)
                if column in replace_map
                else descriptor
                for column, descriptor in input_domain.schema.items()
            }
        )
        for column, values in replace_map.items():
            for value in values:
                if not output_domain[column].valid_py_value(value):
                    raise ValueError(
                        f"Replacement value ({value}) is invalid for column ({column})"
                    )
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=output_domain,
            output_metric=metric,
        )
        self._replace_map = replace_map.copy()

    @property
    def replace_map(self) -> Dict[str, Tuple[float, float]]:
        """Returns mapping used to replace infinite values."""
        return self._replace_map.copy()

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns DataFrame with +inf and -inf replaced in specified columns."""
        # pylint: disable=no-member
        for column, replacement in self.replace_map.items():
            replace_negative = replacement[0]
            replace_positive = replacement[1]
            sdf = sdf.withColumn(
                column,
                sf.when(
                    sf.col(column).eqNullSafe(float("-inf")), sf.lit(replace_negative)
                ).otherwise(sf.col(column)),
            ).withColumn(
                column,
                sf.when(
                    sf.col(column).eqNullSafe(float("inf")), sf.lit(replace_positive)
                ).otherwise(sf.col(column)),
            )
        return sdf
        # pylint: enable=no-member


class ReplaceNaNs(Transformation):
    """Replaces NaNs in one or more specified columns.

    Examples:
        ..
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkFloatColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     [("a1" , 0.1), ("a2", None), (None, float("nan"))],
            ...     schema=["A", "B"]
            ... )

        >>> # Example input
        >>> spark_dataframe.sort("A").show()
        +----+----+
        |   A|   B|
        +----+----+
        |null| NaN|
        |  a1| 0.1|
        |  a2|null|
        +----+----+
        <BLANKLINE>
        >>> replace_nans = ReplaceNaNs(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(allow_null=True),
        ...             "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     replace_map={"B": 0.0},
        ... )
        >>> # Apply transformation to data
        >>> output_dataframe = replace_nans(spark_dataframe)
        >>> output_dataframe.sort("A").show()
        +----+----+
        |   A|   B|
        +----+----+
        |null| 0.0|
        |  a1| 0.1|
        |  a2|null|
        +----+----+
        <BLANKLINE>

        Transformation Contract:
            * Input domain - :class:`~.SparkDataFrameDomain`
            * Output domain - :class:`~.SparkDataFrameDomain`
            * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`, or :class:`~.IfGroupedBy`
            * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`, or :class:`~.IfGroupedBy`

            >>> replace_nans.input_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=True), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=False, allow_null=True, size=64)})
            >>> replace_nans.output_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=True), 'B': SparkFloatColumnDescriptor(allow_nan=False, allow_inf=False, allow_null=True, size=64)})
            >>> replace_nans.input_metric
            SymmetricDifference()
            >>> replace_nans.output_metric
            SymmetricDifference()

            Stability Guarantee:
                :class:`~.DropNulls`'s :meth:`~.stability_function` returns `d_in`.

                >>> replace_nans.stability_function(1)
                1
                >>> replace_nans.stability_function(2)
                2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        replace_map: Dict[str, Any],
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames.
            metric: Distance metric for the input and output Spark DataFrames.
            replace_map: Dictionary mapping column names to value to be used for
                replacing NaNs in that column.
        """
        if isinstance(metric, IfGroupedBy) and not (
            isinstance(metric.inner_metric, (SumOf, RootSumOfSquared))
            and isinstance(metric.inner_metric.inner_metric, SymmetricDifference)
            or isinstance(metric.inner_metric, SymmetricDifference)
        ):
            raise ValueError(
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference."
            )
        if not replace_map:
            raise ValueError("At least one column must be specified.")

        if not set(replace_map) <= set(input_domain.schema):
            raise ValueError(
                "One or more columns do not exist in the input domain"
                f" {set(replace_map)-set(input_domain.schema)}"
            )
        for column, value in replace_map.items():
            if not isinstance(input_domain[column], SparkFloatColumnDescriptor):
                raise ValueError(
                    f"Column of type {input_domain[column]} can not contain NaNs."
                )
            if not cast(SparkFloatColumnDescriptor, input_domain[column]).allow_nan:
                warnings.warn(
                    f"Column ({column}) already disallows NaNs. This transformation"
                    " will have no effect on this column.",
                    RuntimeWarning,
                )
        output_domain = SparkDataFrameDomain(
            {
                column: replace(descriptor, allow_nan=False)
                if column in replace_map
                else descriptor
                for column, descriptor in input_domain.schema.items()
            }
        )
        for column, value in replace_map.items():
            if not output_domain[column].valid_py_value(value):
                raise ValueError(
                    f"Replacement value ({value}) is invalid for column ({column})"
                )
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=output_domain,
            output_metric=metric,
        )
        self._replace_map = replace_map.copy()

    @property
    def replace_map(self) -> Dict[str, Any]:
        """Returns mapping used to replace NaNs and nulls."""
        return self._replace_map.copy()

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns DataFrame with NaNs replaced in specified columns."""
        # pylint: disable=no-member
        for column, replacement in self.replace_map.items():
            sdf = sdf.withColumn(
                column,
                sf.when(sf.isnan(sf.col(column)), sf.lit(replacement)).otherwise(
                    sf.col(column)
                ),
            )
        return sdf


class ReplaceNulls(Transformation):
    """Replaces nulls in one or more specified columns.

    Examples:
        ..
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkFloatColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     [("a1" , 0.1), ("a2", None), (None, float("nan"))],
            ...     schema=["A", "B"]
            ... )

        >>> # Example input
        >>> spark_dataframe.sort("A").show()
        +----+----+
        |   A|   B|
        +----+----+
        |null| NaN|
        |  a1| 0.1|
        |  a2|null|
        +----+----+
        <BLANKLINE>
        >>> replace_nulls = ReplaceNulls(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(allow_null=True),
        ...             "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True),
        ...         }
        ...     ),
        ...     metric=HammingDistance(),
        ...     replace_map={"A": "a0", "B": 0.0},
        ... )
        >>> # Apply transformation to data
        >>> output_dataframe = replace_nulls(spark_dataframe)
        >>> output_dataframe.sort("A").show()
        +---+---+
        |  A|  B|
        +---+---+
        | a0|NaN|
        | a1|0.1|
        | a2|0.0|
        +---+---+
        <BLANKLINE>

        Transformation Contract:
            * Input domain - :class:`~.SparkDataFrameDomain`
            * Output domain - :class:`~.SparkDataFrameDomain`
            * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`, or :class:`~.IfGroupedBy`
            * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`, or :class:`~.IfGroupedBy`

            >>> replace_nulls.input_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=True), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=False, allow_null=True, size=64)})
            >>> replace_nulls.output_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=False, allow_null=False, size=64)})
            >>> replace_nulls.input_metric
            HammingDistance()
            >>> replace_nulls.output_metric
            HammingDistance()

            Stability Guarantee:
                :class:`~.DropNulls`'s :meth:`~.stability_function` returns `d_in`.

                >>> replace_nulls.stability_function(1)
                1
                >>> replace_nulls.stability_function(2)
                2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        replace_map: Dict[str, Any],
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames.
            metric: Distance metric for the input and output Spark DataFrames.
            replace_map: Dictionary mapping column names to value to be used for
                replacing nulls in that column.
        """
        if isinstance(metric, IfGroupedBy) and not (
            isinstance(metric.inner_metric, (SumOf, RootSumOfSquared))
            and isinstance(metric.inner_metric.inner_metric, SymmetricDifference)
            or isinstance(metric.inner_metric, SymmetricDifference)
        ):
            raise ValueError(
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference."
            )
        if not replace_map:
            raise ValueError("At least one column must be specified.")

        if not set(replace_map) <= set(input_domain.schema):
            raise ValueError(
                "One or more columns do not exist in the input domain"
                f" {set(replace_map)-set(input_domain.schema)}"
            )
        output_domain = SparkDataFrameDomain(
            {
                column: replace(descriptor, allow_null=False)
                if column in replace_map
                else descriptor
                for column, descriptor in input_domain.schema.items()
            }
        )
        for column, value in replace_map.items():
            if not output_domain[column].valid_py_value(value):
                raise ValueError(
                    f"Replacement value ({value}) is invalid for column ({column})"
                )
            if not input_domain[column].allow_null:
                warnings.warn(
                    f"Column ({column}) already disallows nulls. This transformation"
                    " will have no effect on this column.",
                    RuntimeWarning,
                )
        if isinstance(metric, IfGroupedBy):
            if metric.column in replace_map:
                raise ValueError(
                    "Cannot replace values in the grouping column for IfGroupedBy."
                )
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=output_domain,
            output_metric=metric,
        )
        self._replace_map = replace_map.copy()

    @property
    def replace_map(self) -> Dict[str, Any]:
        """Returns mapping used to replace nulls."""
        return self._replace_map.copy()

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns DataFrame with nulls replaced in specified columns."""
        # pylint: disable=no-member
        for column, replacement in self.replace_map.items():
            sdf = sdf.withColumn(
                column,
                sf.when(sf.isnull(sf.col(column)), sf.lit(replacement)).otherwise(
                    sf.col(column)
                ),
            )
        return sdf
