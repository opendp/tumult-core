"""Transformations for grouping and aggregating Spark DataFrames."""
# TODO(#1320): Add link to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Optional, Union, cast, overload

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from typeguard import typechecked

import tmlt.core.transformations.spark_transformations.nan as nan
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.metrics import (
    AbsoluteDifference,
    HammingDistance,
    OnColumn,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame


class Count(Transformation):
    r"""Counts the number of records in a spark DataFrame.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.metrics import (
            ...     SymmetricDifference,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a2", "a2"],
            ...             "X": [2, 3, 5, -1],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A  X
        0  a1  2
        1  a1  3
        2  a2 -1
        3  a2  5
        >>> # Create the transformation
        >>> count_dataframe = Count(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ... )
        >>> # Apply transformation to data
        >>> count_dataframe(spark_dataframe)
        4

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.NumpyIntegerDomain`
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.HammingDistance`
        * Output metric - :class:`~.AbsoluteDifference`

        >>> count_dataframe.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> count_dataframe.output_domain
        NumpyIntegerDomain(size=64)
        >>> count_dataframe.input_metric
        SymmetricDifference()
        >>> count_dataframe.output_metric
        AbsoluteDifference()

        Stability Guarantee:
            :class:`~.Count`'s :meth:`~.stability_function` returns `d_in` if input metric is
            :class:`~.SymmetricDifference` and :math:`d_{in} * 2` if input metric is :class:`~.HammingDistance`.

             >>> count_dataframe.stability_function(1)
             1
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[SymmetricDifference, HammingDistance],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrames.
            input_metric: Distance metric on input DataFrames.
        """
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=NumpyIntegerDomain(),
            output_metric=AbsoluteDifference(),
        )

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        d_in = ExactNumber(d_in)
        return d_in if self.input_metric == SymmetricDifference() else d_in * 2

    def __call__(self, df: DataFrame) -> np.int64:
        """Returns the number of records in given DataFrame."""
        return np.int64(df.count())


class CountDistinct(Transformation):
    r"""Counts the number of distinct records in a spark DataFrame.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.metrics import (
            ...     SymmetricDifference,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a2", "a2"],
            ...             "X": [2, 2, -1, 5],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A  X
        0  a1  2
        1  a1  2
        2  a2 -1
        3  a2  5
        >>> # Create the transformation
        >>> count_distinct_dataframe = CountDistinct(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ... )
        >>> # Apply transformation to data
        >>> count_distinct_dataframe(spark_dataframe)
        3

    Transformation contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.NumpyIntegerDomain`
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.HammingDistance`
        * Output metric - :class:`~.AbsoluteDifference`

        >>> count_distinct_dataframe.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> count_distinct_dataframe.output_domain
        NumpyIntegerDomain(size=64)
        >>> count_distinct_dataframe.input_metric
        SymmetricDifference()
        >>> count_distinct_dataframe.output_metric
        AbsoluteDifference()

        Stability Guarantee:
            :class:`~CountDistinct`'s :meth:`~stability_function` returns
            `d_in` if input metric is :class:`~.SymmetricDifference` and
            :math:`d_{in} * 2` if input metric is :class:`~.HammingDistance`.

            >>> count_distinct_dataframe.stability_function(1)
            1
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[SymmetricDifference, HammingDistance],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrames.
            input_metric: Distance metric on input DataFrames.
        """
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=NumpyIntegerDomain(),
            output_metric=AbsoluteDifference(),
        )

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        d_in = ExactNumber(d_in)
        if self.input_metric == SymmetricDifference():
            return d_in
        else:  # input metric is HammingDistance
            return d_in * 2

    def __call__(self, df: DataFrame) -> int:
        """Returns the number of distinct records in the given DataFrame."""
        # Note: This cannot use sf.count_distinct since it ignores rows with nulls.
        return df.distinct().count()


class CountGrouped(Transformation):
    r"""Counts the number of records in each group in a :class:`~.GroupedDataFrame`.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkGroupedDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.metrics import (
            ...     SymmetricDifference,
            ...     SumOf,
            ... )
            >>> from tmlt.core.utils.grouped_dataframe import (
            ...     GroupedDataFrame,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a2", "a2"],
            ...             "X": [2, 3, 5, -1],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A  X
        0  a1  2
        1  a1  3
        2  a2 -1
        3  a2  5
        >>> # Specify group keys
        >>> group_keys = spark.createDataFrame(
        ...     [("a0",), ("a1",)],
        ...     schema=["A"],
        ... )
        >>> # Note that we have omitted 'a2' from our group keys
        >>> # and included 'a0' which doesn't exist in the DataFrame
        >>> # Create the transformation
        >>> count_by_A = CountGrouped(
        ...     input_domain=SparkGroupedDataFrameDomain(
        ...         schema={
        ...             "A": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...         group_keys=group_keys
        ...     ),
        ...     input_metric=SumOf(SymmetricDifference()),
        ... )
        >>> # Create GroupedDataFrame
        >>> grouped_dataframe = GroupedDataFrame(
        ...     dataframe=spark_dataframe,
        ...     group_keys=group_keys,
        ... )
        >>> # Apply transformation to data
        >>> print_sdf(count_by_A(grouped_dataframe))
            A  count
        0  a0      0
        1  a1      2
        >>> # Note that the output does not contain an entry
        >>> # for group key 'a2' but it does contain an entry
        >>> # for group key 'a0'.

    Transformation Contract:
        * Input domain - :class:`~.SparkGroupedDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SumOf` or :class:`~.RootSumOfSquared` of :class:`~.SymmetricDifference`
        * Output metric - :class:`~.OnColumn`

        >>> count_by_A.input_domain
        SparkGroupedDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)}, group_keys=DataFrame[A: string])
        >>> count_by_A.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'count': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> count_by_A.input_metric
        SumOf(inner_metric=SymmetricDifference())
        >>> count_by_A.output_metric
        OnColumn(column='count', metric=SumOf(inner_metric=AbsoluteDifference()))

        Stability Guarantee:
            :class:`~.CountGrouped`'s :meth:`~.stability_function` returns `d_in`.

            >>> count_by_A.stability_function(1)
            1
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkGroupedDataFrameDomain,
        input_metric: Union[SumOf, RootSumOfSquared],
        count_column: Optional[str] = None,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input GroupedDataFrames produced by some
                GroupBy transformation.
            input_metric: Distance metric on inputs.
            count_column: Column name for output group counts. If None, output column
                will be named "count".
        """
        if count_column is None:
            count_column = "count"
        if input_metric.inner_metric != SymmetricDifference():
            raise ValueError(
                "Inner metric for the input metric must be SymmetricDifference,"
                f" not {input_metric.inner_metric}."
            )
        if count_column in set(input_domain.group_keys.columns):
            raise ValueError(
                f"Invalid count column name: ({count_column}) column already exists"
            )
        groupby_columns_schema = {
            groupby_column: input_domain[groupby_column]
            for groupby_column in input_domain.group_keys.columns
        }
        output_domain = SparkDataFrameDomain(
            schema={
                **groupby_columns_schema,
                count_column: SparkIntegerColumnDescriptor(),
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
    def input_domain(self) -> SparkGroupedDataFrameDomain:
        """Returns input domain."""
        return cast(SparkGroupedDataFrameDomain, super().input_domain)

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

    def __call__(self, grouped_data: GroupedDataFrame) -> DataFrame:
        """Returns a DataFrame containing counts for each group."""
        # pylint: disable=no-member
        return grouped_data.agg(
            func=sf.count("*").alias(self.count_column), fill_value=0
        )


class CountDistinctGrouped(Transformation):
    r"""Counts the number of distinct records in each group in a :class:`~.GroupedDataFrame`.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkGroupedDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.metrics import (
            ...     SymmetricDifference,
            ...     SumOf,
            ... )
            >>> from tmlt.core.utils.grouped_dataframe import (
            ...     GroupedDataFrame,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a1", "a2", "a2"],
            ...             "X": [2, 2, 3, 5, -1],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A  X
        0  a1  2
        1  a1  2
        2  a1  3
        3  a2 -1
        4  a2  5
        >>> # Specify group keys
        >>> group_keys = spark.createDataFrame(
        ...     [("a0",), ("a1",)],
        ...     schema=["A"],
        ... )
        >>> # Note that we have omitted 'a2' from our group keys
        >>> # and included 'a0' which doesn't exist in the DataFrame
        >>> # Create the transformation
        >>> count_distinct_by_A = CountDistinctGrouped(
        ...     input_domain=SparkGroupedDataFrameDomain(
        ...         schema={
        ...             "A": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...         group_keys=group_keys
        ...     ),
        ...     input_metric=SumOf(SymmetricDifference()),
        ... )
        >>> # Create GroupedDataFrame
        >>> grouped_dataframe = GroupedDataFrame(
        ...     dataframe=spark_dataframe,
        ...     group_keys=group_keys,
        ... )
        >>> # Apply transformation to data
        >>> print_sdf(count_distinct_by_A(grouped_dataframe))
            A  count_distinct
        0  a0               0
        1  a1               2
        >>> # Note that the output does not contain an entry
        >>> # for group key 'a2' but it does contain an entry
        >>> # for group key 'a0'.

    Transformation Contract:
        * Input domain - :class:`~.SparkGroupedDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SumOf` or :class:`~.RootSumOfSquared` of :class:`~.SymmetricDifference`
        * Output metric - :class:`~.OnColumn`

        >>> count_distinct_by_A.input_domain
        SparkGroupedDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)}, group_keys=DataFrame[A: string])
        >>> count_distinct_by_A.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'count_distinct': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> count_distinct_by_A.input_metric
        SumOf(inner_metric=SymmetricDifference())
        >>> count_distinct_by_A.output_metric
        OnColumn(column='count_distinct', metric=SumOf(inner_metric=AbsoluteDifference()))

        Stability Guarantee:
            :class:`~.CountDistinctGrouped`'s :meth:`~.stability_function` returns `d_in`.

            >>> count_distinct_by_A.stability_function(1)
            1
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkGroupedDataFrameDomain,
        input_metric: Union[SumOf, RootSumOfSquared],
        count_column: Optional[str] = None,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input GroupedDataFrames produced by some
                GroupBy transformation.
            input_metric: Distance metric on inputs.
            count_column: Column name for output group counts. If None, output column
                will be named "count_distinct".
        """
        if count_column is None:
            count_column = "count_distinct"
        if input_metric.inner_metric != SymmetricDifference():
            raise ValueError(
                "Inner metric for the input metric must be SymmetricDifference,"
                f" not {input_metric.inner_metric}."
            )
        if count_column in set(input_domain.group_keys.columns):
            raise ValueError(
                f"Invalid count column name: ({count_column}) column already exists"
            )
        groupby_columns_schema = {
            groupby_column: input_domain[groupby_column]
            for groupby_column in input_domain.group_keys.columns
        }
        output_domain = SparkDataFrameDomain(
            schema={
                **groupby_columns_schema,
                count_column: SparkIntegerColumnDescriptor(),
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
    def input_domain(self) -> SparkGroupedDataFrameDomain:
        """Returns input domain."""
        return cast(SparkGroupedDataFrameDomain, super().input_domain)

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

    def __call__(self, grouped_data: GroupedDataFrame) -> DataFrame:
        """Returns a DataFrame containing counts for each group."""
        # pylint: disable=no-member
        # Note: This cannot use sf.count_distinct since it ignores rows with nulls.
        return grouped_data.agg(
            sf.size(sf.collect_set(sf.struct("*"))).alias(self.count_column),
            fill_value=0,
        )


class Sum(Transformation):
    r"""Returns the sum of a single numeric column in a spark DataFrame.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.metrics import (
            ...     SymmetricDifference,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a2", "a2"],
            ...             "X": [2, 3, 5, -1],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A  X
        0  a1  2
        1  a1  3
        2  a2 -1
        3  a2  5
        >>> # Create the transformation
        >>> sum_X = Sum(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ...     measure_column="X",
        ...     upper=4,
        ...     lower=0,
        ... )
        >>> # Apply transformation to data
        >>> sum_X(spark_dataframe)
        9

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.NumpyIntegerDomain` or :class:`~.NumpyFloatDomain`
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.HammingDistance`
        * Output metric - :class:`~.AbsoluteDifference`

        >>> sum_X.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> sum_X.output_domain
        NumpyIntegerDomain(size=64)
        >>> sum_X.input_metric
        SymmetricDifference()
        >>> sum_X.output_metric
        AbsoluteDifference()

        Stability Guarantee:
            :class:`~.Sum`'s :meth:`~.stability_function` returns `d_in` times sensitivity of
            the sum. (See below for more information).

            >>> sum_X.stability_function(1)
            4

            The sensitivity of the sum is:

            * :math:`\max(|h|, |\ell|)` if the input metric is
              :class:`~.SymmetricDifference`
            * :math:`h - \ell` if the input metric is
              :class:`~.HammingDistance`
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        input_metric: Union[SymmetricDifference, HammingDistance],
        measure_column: str,
        lower: ExactNumberInput,
        upper: ExactNumberInput,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrames.
            input_metric: Metric on input DataFrames.
            measure_column: Name of the column to be summed. This must be a numeric column.
            lower: Lower clipping bound for measure column.
            upper: Upper clipping bound for measure column.
        """
        if measure_column not in input_domain.schema:
            raise ValueError(
                f"Invalid measure column: ({measure_column}) does not exist."
            )

        measure_column_descriptor = input_domain[measure_column]
        if not isinstance(
            measure_column_descriptor,
            (SparkIntegerColumnDescriptor, SparkFloatColumnDescriptor),
        ):
            raise ValueError(
                f"Measure column ({measure_column}) must be numeric, not"
                f" {measure_column_descriptor}"
            )
        if measure_column_descriptor.allow_null or (
            isinstance(measure_column_descriptor, SparkFloatColumnDescriptor)
            and measure_column_descriptor.allow_nan
        ):
            raise ValueError(
                "Input domain must not allow nulls or NaNs on the measure column"
                f" ({measure_column}). See {nan.__name__} for transformations to drop"
                " or replace such values."
            )
        measure_column_nonintegral = isinstance(
            measure_column_descriptor, SparkFloatColumnDescriptor
        )
        self._lower = ExactNumber(lower)
        self._upper = ExactNumber(upper)
        if not measure_column_nonintegral:
            if not self._lower.is_integer or not self._upper.is_integer:
                raise ValueError("Clipping bounds must be integral")
        if not self._lower.is_finite or not self._upper.is_finite:
            raise ValueError("Clipping bounds must be finite")
        if self._lower > self._upper:
            raise ValueError(
                "Lower clipping bound is larger than upper clipping bound."
            )
        if self._upper > 2 ** 970:
            raise ValueError("Upper clipping bound should be at most 2^970.")
        if self._lower < -(2 ** 970):
            raise ValueError("Lower clipping bound should be at least -2^970.")

        self._measure_column = measure_column
        output_domain = output_domain = (
            NumpyFloatDomain() if measure_column_nonintegral else NumpyIntegerDomain()
        )
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=AbsoluteDifference(),
        )

    @property
    def upper(self) -> ExactNumber:
        """Returns upper clipping bound."""
        return self._upper

    @property
    def lower(self) -> ExactNumber:
        """Returns lower clipping bound."""
        return self._lower

    @property
    def measure_column(self) -> str:
        """Returns name of the column to be summed."""
        return self._measure_column

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        exact_d_in = ExactNumber(d_in)
        if self.input_metric == SymmetricDifference():
            sensitivity = max(abs(self.upper), abs(self.lower))
        else:
            assert self.input_metric == HammingDistance()
            sensitivity = self.upper - self.lower
        return exact_d_in * sensitivity

    def __call__(self, df: DataFrame) -> Union[int, float]:
        """Returns the sum of specified column in the dataframe."""
        # pylint: disable=no-member
        lower_ceil = self.lower.to_float(round_up=True)
        upper_floor = (
            lower_ceil
            if self.lower == self.upper  # TODO(#1023)
            else self.upper.to_float(round_up=False)
        )

        clipped_df = df.withColumn(
            self.measure_column,
            sf.when(sf.col(self.measure_column) < lower_ceil, lower_ceil)
            .when(sf.col(self.measure_column) > upper_floor, upper_floor)
            .otherwise(sf.col(self.measure_column)),
        )
        column_sum = clipped_df.agg(sf.sum(self.measure_column).alias("sum")).collect()[
            0
        ]["sum"]
        if column_sum is None:  # This happens if there are 0 rows in the df.
            return self.output_domain.carrier_type(0)
        return self.output_domain.carrier_type(column_sum)


class SumGrouped(Transformation):
    r"""Computes the sum of a column for each group in a :class:`~.GroupedDataFrame`.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkGroupedDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.metrics import (
            ...     SymmetricDifference,
            ...     SumOf,
            ... )
            >>> from tmlt.core.utils.grouped_dataframe import (
            ...     GroupedDataFrame,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a2", "a2"],
            ...             "X": [2, 3, 6, -1],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A  X
        0  a1  2
        1  a1  3
        2  a2 -1
        3  a2  6
        >>> # Specify group keys
        >>> group_keys = spark.createDataFrame(
        ...     [("a0",), ("a2",)],
        ...     schema=["A"],
        ... )
        >>> # Note that we omit the key 'a1' even though it
        >>> # exists in the spark dataframe and include 'a0'.
        >>> # Create the transformation
        >>> sum_X_by_A = SumGrouped(
        ...     input_domain=SparkGroupedDataFrameDomain(
        ...         schema={
        ...             "A": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...         group_keys=group_keys
        ...     ),
        ...     input_metric=SumOf(SymmetricDifference()),
        ...     measure_column="X",
        ...     upper=4,
        ...     lower=0,
        ... )
        >>> # Create GroupedDataFrame
        >>> grouped_dataframe = GroupedDataFrame(
        ...     dataframe=spark_dataframe,
        ...     group_keys=group_keys,
        ... )
        >>> # Apply transformation to data
        >>> print_sdf(sum_X_by_A(grouped_dataframe))
            A  sum(X)
        0  a0       0
        1  a2       4

    Transformation Contract:
        * Input domain - :class:`~.SparkGroupedDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SumOf` or :class:`~.RootSumOfSquared`
        * Output metric - :class:`~.OnColumn`

        >>> sum_X_by_A.input_domain
        SparkGroupedDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)}, group_keys=DataFrame[A: string])
        >>> sum_X_by_A.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'sum(X)': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> sum_X_by_A.input_metric
        SumOf(inner_metric=SymmetricDifference())
        >>> sum_X_by_A.output_metric
        OnColumn(column='sum(X)', metric=SumOf(inner_metric=AbsoluteDifference()))

        Stability Guarantee:
            :class:`~.SumGrouped`'s :meth:`~.stability_function` returns `d_in` * sensitivity of the sum.

            >>> sum_X_by_A.stability_function(1)
            4

            The sensitivity of the sum is:

            * :math:`\max(|h|, |\ell|)`
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkGroupedDataFrameDomain,
        input_metric: Union[SumOf, RootSumOfSquared],
        measure_column: str,
        lower: ExactNumberInput,
        upper: ExactNumberInput,
        sum_column: Optional[str] = None,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input GroupedDataFrames.
            input_metric: Distance metric on inputs. This should be one of
                SumOf(SymmetricDifference()) or RootSumOfSquared(SymmetricDifference())
            measure_column: Name of column to be summed.
            lower: Lower clipping bound for the measure column.
            upper: Upper clipping bound for the measure column.
            sum_column: Name of the output sum column. If None, output column
                will be named 'sum(<measure_column>)'.
        """
        if sum_column is None:
            sum_column = f"sum({measure_column})"

        groupby_columns = input_domain.group_keys.columns
        if measure_column not in set(input_domain.schema) - set(groupby_columns):
            raise ValueError(f"Invalid measure column: {measure_column}")

        measure_column_descriptor = input_domain[measure_column]
        if not isinstance(
            measure_column_descriptor,
            (SparkIntegerColumnDescriptor, SparkFloatColumnDescriptor),
        ):
            raise ValueError(
                f"Measure column ({measure_column}) must be numeric, not"
                f" {measure_column_descriptor}"
            )
        if measure_column_descriptor.allow_null or (
            isinstance(measure_column_descriptor, SparkFloatColumnDescriptor)
            and measure_column_descriptor.allow_nan
        ):
            raise ValueError(
                "Input domain must not allow nulls or NaNs on the sum column"
                f" ({measure_column}). See {nan.__name__} for transformations to drop"
                " or replace such values."
            )
        measure_column_is_integral = isinstance(
            measure_column_descriptor, SparkIntegerColumnDescriptor
        )
        self._lower = ExactNumber(lower)
        self._upper = ExactNumber(upper)
        if measure_column_is_integral:
            if not self._lower.is_integer or not self._upper.is_integer:
                raise ValueError("Clipping bounds must be integral")
        if not self._lower.is_finite or not self._upper.is_finite:
            raise ValueError("Clipping bounds must be finite")
        if self._lower > self._upper:
            raise ValueError(
                "Lower clipping bound is larger than upper clipping bound."
            )
        if not isinstance(input_metric.inner_metric, SymmetricDifference):
            raise ValueError(
                "Input metric must be SumOf(SymmetricDifference()) or"
                " RootSumOfSquared(SymmetricDifference())"
            )
        if sum_column in groupby_columns:
            raise ValueError(f"Invalid sum column name: '{sum_column}' already exists")

        groupby_columns_schema = {
            groupby_column: input_domain[groupby_column]
            for groupby_column in groupby_columns
        }
        output_domain = SparkDataFrameDomain(
            schema={**groupby_columns_schema, sum_column: input_domain[measure_column]}
        )
        output_metric = (
            OnColumn(sum_column, SumOf(AbsoluteDifference()))
            if isinstance(input_metric, SumOf)
            else OnColumn(sum_column, RootSumOfSquared(AbsoluteDifference()))
        )
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )
        self._sum_column = sum_column
        self._measure_column = measure_column

    @property
    def upper(self) -> ExactNumber:
        """Returns upper clipping bound."""
        return self._upper

    @property
    def lower(self) -> ExactNumber:
        """Returns lower clipping bound."""
        return self._lower

    @property
    def measure_column(self) -> str:
        """Returns name of the column to be summed."""
        return self._measure_column

    @property
    def sum_column(self) -> str:
        """Returns name of the output column containing sums."""
        return self._sum_column

    @property
    def input_domain(self) -> SparkGroupedDataFrameDomain:
        """Returns input domain."""
        return cast(SparkGroupedDataFrameDomain, self._input_domain)

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        sensitivity = max(abs(self.upper), abs(self.lower))
        return ExactNumber(d_in * sensitivity)

    def __call__(self, grouped_dataframe: GroupedDataFrame) -> DataFrame:
        """Returns DataFrame containing sum of specified column for each group."""
        # pylint: disable=no-member
        lower_ceil = self.lower.to_float(round_up=True)
        upper_floor = (
            lower_ceil
            if self.lower == self.upper  # TODO(#1023)
            else self.upper.to_float(round_up=False)
        )

        return grouped_dataframe.agg(
            func=sf.sum(
                sf.array_min(
                    sf.array(
                        [
                            sf.array_max(
                                sf.array(
                                    [sf.lit(lower_ceil), sf.col(self.measure_column)]
                                )
                            ),
                            sf.lit(upper_floor),
                        ]
                    )
                )
            ).alias(self.sum_column),
            fill_value=0,
        )


@overload
def create_count_aggregation(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance],
    count_column: Optional[str],
) -> Count:
    ...


@overload
def create_count_aggregation(
    input_domain: SparkGroupedDataFrameDomain,
    input_metric: Union[SumOf, RootSumOfSquared],
    count_column: Optional[str],
) -> CountGrouped:
    ...


def create_count_aggregation(
    input_domain, input_metric, count_column=None
):  # pylint: disable=missing-type-doc, missing-return-type-doc
    """Returns a :class:`~.Count` or :class:`~.CountGrouped` transformation.

    Args:
        input_domain: Domain of input DataFrames or GroupedDataFrames.
        input_metric: Distance metric on inputs.
        count_column: If `input_domain` is a SparkGroupedDataFrameDomain, this is the
            name of the output count column.
    """
    if isinstance(input_domain, SparkDataFrameDomain):
        return Count(input_domain=input_domain, input_metric=input_metric)
    else:
        if not isinstance(input_domain, SparkGroupedDataFrameDomain):
            raise ValueError(
                "Input domain must be SparkDataFrameDomain or"
                " SparkGroupedDataFrameDomain."
            )
        return CountGrouped(
            input_domain=input_domain,
            input_metric=input_metric,
            count_column=count_column,
        )


@overload
def create_count_distinct_aggregation(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance],
    count_column: Optional[str],
) -> Count:
    ...


@overload
def create_count_distinct_aggregation(
    input_domain: SparkGroupedDataFrameDomain,
    input_metric: Union[SumOf, RootSumOfSquared],
    count_column: Optional[str],
) -> CountGrouped:
    ...


def create_count_distinct_aggregation(
    input_domain, input_metric, count_column=None
):  # pylint: disable=missing-type-doc, missing-return-type-doc, line-too-long
    """Returns a :class:`~.CountDistinct` or :class:`~.CountDistinctGrouped` transformation.

    Args:
        input_domain: Domain of input DataFrames or GroupedDataFrames.
        input_metric: Distance metric on inputs.
        count_column: If `input_domain` is a SparkGroupedDataFrameDomain, this is the
            name of the output count column.
    """
    if isinstance(input_domain, SparkDataFrameDomain):
        return CountDistinct(input_domain=input_domain, input_metric=input_metric)
    else:
        if not isinstance(input_domain, SparkGroupedDataFrameDomain):
            raise ValueError(
                "Input domain must be SparkDataFrameDomain or"
                " SparkGroupedDataFrameDomain."
            )
        return CountDistinctGrouped(
            input_domain=input_domain,
            input_metric=input_metric,
            count_column=count_column,
        )


@overload
def create_sum_aggregation(
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance],
    measure_column: str,
    lower: ExactNumberInput,
    upper: ExactNumberInput,
    sum_column: Optional[str],
) -> Sum:
    ...


@overload
def create_sum_aggregation(
    input_domain: SparkGroupedDataFrameDomain,
    input_metric: Union[SumOf, RootSumOfSquared],
    measure_column: str,
    lower: ExactNumberInput,
    upper: ExactNumberInput,
    sum_column: Optional[str],
) -> SumGrouped:
    ...


def create_sum_aggregation(
    input_domain, input_metric, measure_column, lower, upper, sum_column=None
):  # pylint: disable=missing-type-doc, missing-return-type-doc
    """Returns a :class:`~.Sum` or :class:`~.SumGrouped` transformation.

    Args:
        input_domain: Domain of input DataFrames or GroupedDataFrames.
        input_metric: Distance metric on inputs.
            name of the output sum column.
        measure_column: Column to be summed.
        lower: Lower clipping bound for measure column.
        upper: Upper clipping bound for measure column.
        sum_column: If `input_domain` is a SparkGroupedDataFrameDomain, this is the
            column name to be used for sums in the DataFrame output by the
            measurement. If None, this column will be named “sum(<measure_column>)”.
    """
    if isinstance(input_domain, SparkDataFrameDomain):
        return Sum(
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column=measure_column,
            upper=upper,
            lower=lower,
        )
    else:
        if not isinstance(input_domain, SparkGroupedDataFrameDomain):
            raise ValueError(
                "Input Domain must be SparkDataFrameDomain or"
                " SparkGroupedDataFrameDomain"
            )
        return SumGrouped(
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column=measure_column,
            upper=upper,
            lower=lower,
            sum_column=sum_column,
        )
