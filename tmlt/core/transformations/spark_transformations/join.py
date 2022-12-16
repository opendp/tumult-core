"""Transformations for joining Spark DataFrames."""
# TODO(#1320): Add links to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from dataclasses import replace
from enum import Enum
from functools import reduce
from typing import Any, Dict, List, Optional, Union, cast

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from typeguard import typechecked

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
)
from tmlt.core.metrics import (
    AddRemoveKeys,
    DictMetric,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.truncation import drop_large_groups, truncate_large_groups


class PublicJoin(Transformation):
    """Join a Spark DataFrame with a public Pandas DataFrame.

    Examples:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import SparkStringColumnDescriptor
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2"],
            ...         }
            ...     )
            ... )
            >>> spark_dataframe_with_null = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", None, "a3"],
            ...             "B": ["b1", "b1", "b2", "b2"],
            ...         }
            ...     )
            ... )

        Natural join:

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> # Create example public dataframe
        >>> public_dataframe = spark.createDataFrame(
        ...     pd.DataFrame(
        ...         {
        ...             "B": ["b1", "b2", "b2"],
        ...             "C": ["c1", "c2", "c3"],
        ...         }
        ...     )
        ... )
        >>> # Create the transformation
        >>> natural_join = PublicJoin(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     public_df=public_dataframe,
        ...     metric=SymmetricDifference(),
        ... )
        >>> # Apply transformation to data
        >>> joined_spark_dataframe = natural_join(spark_dataframe)
        >>> print_sdf(joined_spark_dataframe)
            B   A   C
        0  b1  a1  c1
        1  b1  a2  c1
        2  b2  a3  c2
        3  b2  a3  c2
        4  b2  a3  c3
        5  b2  a3  c3

        Join with some common columns excluded from join:

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> # Create example public dataframe
        >>> public_dataframe = spark.createDataFrame(
        ...     pd.DataFrame(
        ...         {
        ...             "A": ["a1", "a1", "a2"],
        ...             "B": ["b1", "b1", "b2"],
        ...         }
        ...     )
        ... )
        >>> # Create the transformation
        >>> public_join = PublicJoin(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     public_df=public_dataframe,
        ...     metric=SymmetricDifference(),
        ...     join_cols=["A"],
        ... )
        >>> # Apply transformation to data
        >>> joined_spark_dataframe = public_join(spark_dataframe)
        >>> print_sdf(joined_spark_dataframe)
            A B_left B_right
        0  a1     b1      b1
        1  a1     b1      b1
        2  a2     b1      b2

        Join on nulls

        >>> # Example input
        >>> print_sdf(spark_dataframe_with_null)
              A   B
        0    a1  b1
        1    a2  b1
        2    a3  b2
        3  None  b2
        >>> # Create example public dataframe
        >>> public_dataframe = spark.createDataFrame(
        ...     pd.DataFrame(
        ...         {
        ...             "A": ["a1", "a2", None],
        ...             "C": ["c1", "c2", "c3"],
        ...         }
        ...     )
        ... )
        >>> # Create the transformation
        >>> join_transformation = PublicJoin(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     public_df=public_dataframe,
        ...     metric=SymmetricDifference(),
        ...     join_on_nulls=True,
        ... )
        >>> # Apply transformation to data
        >>> joined_spark_dataframe = join_transformation(spark_dataframe_with_null)
        >>> print_sdf(joined_spark_dataframe)
              A   B   C
        0    a1  b1  c1
        1    a2  b1  c2
        2  None  b2  c3

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          (matches input metric)

        >>> public_join.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> public_join.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B_left': SparkStringColumnDescriptor(allow_null=False), 'B_right': SparkStringColumnDescriptor(allow_null=True)})
        >>> public_join.input_metric
        SymmetricDifference()
        >>> public_join.output_metric
        SymmetricDifference()

        Stability Guarantee:
            For

            - SymmetricDifference()
            - IfGroupedBy(column, SumOf(SymmetricDifference()))
            - IfGroupedBy(column, RootSumOfSquared(SymmetricDifference()))

            :class:`~.PublicJoin`'s :meth:`~.stability_function` returns the `d_in`
            times the maximum count of any combination of values in the join columns of
            `public_df`.

            >>> # Both example transformations had a stability of 2
            >>> natural_join.join_cols
            ['B']
            >>> natural_join.public_df.toPandas()
                B   C
            0  b1  c1
            1  b2  c2
            2  b2  c3
            >>> # Notice that 'b2' occurs twice
            >>> natural_join.stability_function(1)
            2
            >>> natural_join.stability_function(2)
            4

            For

            - IfGroupedBy(column, SymmetricDifference())

            :class:`~.PublicJoin`'s :meth:`~.stability_function` returns `d_in`

            >>> PublicJoin(
            ...     input_domain=SparkDataFrameDomain(
            ...         {
            ...             "A": SparkStringColumnDescriptor(),
            ...             "B": SparkStringColumnDescriptor(),
            ...         }
            ...     ),
            ...     public_df=public_dataframe,
            ...     metric=IfGroupedBy("A", SymmetricDifference()),
            ... ).stability_function(2)
            2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        metric: Union[SymmetricDifference, IfGroupedBy],
        public_df: DataFrame,
        public_df_domain: Optional[SparkDataFrameDomain] = None,
        join_cols: Optional[List[str]] = None,
        join_on_nulls: bool = False,
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames.
            metric: Metric for input/output Spark DataFrames.
            public_df: A Spark DataFrame to join with.
            public_df_domain: Domain of public DataFrame to join with. If this domain
                indicates that a float column does not allow nans (or infs), all rows
                in `public_df` containing a nan (or an inf) in that column will be
                dropped. If None, domain is inferred from the schema of `public_df` and
                any float column will be marked as allowing inf and nan values.
            join_cols: Names of columns to join on. If None, a natural join is
                performed.
            join_on_nulls: If True, null values on corresponding join columns of the
                public and private DataFrames will be considered to be equal.
        """
        if isinstance(metric, IfGroupedBy):
            if metric.inner_metric not in (
                SymmetricDifference(),
                SumOf(SymmetricDifference()),
                RootSumOfSquared(SymmetricDifference()),
            ):
                raise ValueError(
                    "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                    "SumOf(SymmetricDifference()), or "
                    "RootSumOfSquared(SymmetricDifference())"
                )

        common_cols = set(input_domain.schema) & set(public_df.columns)
        if not join_cols:
            if not common_cols:
                raise ValueError("Can not join: No common columns.")
            join_cols = sorted(common_cols, key=list(input_domain.schema).index)
        else:
            join_cols = join_cols.copy()

        if not set(join_cols) <= set(common_cols):
            raise ValueError("Join columns must be common to both DataFrames.")

        if public_df_domain:
            if public_df.schema != public_df_domain.spark_schema:
                raise ValueError(
                    "public_df's Spark schema does not match public_df_domain"
                )
            for col, descriptor in public_df_domain.schema.items():
                if isinstance(descriptor, SparkFloatColumnDescriptor):
                    if not descriptor.allow_inf:
                        public_df = public_df.filter(
                            ~public_df[col].isin([float("inf"), -float("inf")])
                        )
                    if not descriptor.allow_nan:
                        public_df = public_df.filter(~sf.isnan(public_df[col]))

        else:
            public_df_domain = SparkDataFrameDomain.from_spark_schema(public_df.schema)
        for col in join_cols:
            if input_domain[col].data_type != public_df_domain[col].data_type:
                raise ValueError(
                    "Join columns must have identical types on both "
                    f"DataFrames. {input_domain[col].data_type} and "
                    f"{public_df_domain[col].data_type} are incompatible."
                )

        join_cols_schema = {col: input_domain[col] for col in join_cols}
        overlapping_cols = common_cols - set(join_cols)
        left_schema = {
            col + ("_left" if col in overlapping_cols else ""): input_domain[col]
            for col in input_domain.schema
            if col not in join_cols
        }
        right_schema = {
            col + ("_right" if col in overlapping_cols else ""): public_df_domain[col]
            for col in public_df_domain.schema
            if col not in join_cols
        }
        output_domain = SparkDataFrameDomain(
            {**join_cols_schema, **left_schema, **right_schema}
        )
        if isinstance(metric, IfGroupedBy) and metric.column in overlapping_cols:
            raise ValueError(
                f"IfGroupedBy column {metric.column} is an overlapping"
                " column but not a join key."
            )
        for col in overlapping_cols:
            public_df = public_df.withColumnRenamed(col, f"{col}_right")

        public_df_join_columns = public_df.select(*join_cols)
        if not join_on_nulls:
            public_df_join_columns = public_df_join_columns.dropna()
        if (
            isinstance(metric, IfGroupedBy)
            and metric.inner_metric == SymmetricDifference()
        ):
            self._join_stability = 1
        else:
            self._join_stability = max(
                public_df_join_columns.groupby(*join_cols)
                .count()
                .select("count")
                .toPandas()["count"]
                .to_list(),
                default=0,
            )

        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=output_domain,
            output_metric=metric,
        )
        self._join_on_nulls = join_on_nulls
        self._overlapping_cols = overlapping_cols
        self._public_df = public_df
        self._public_df = public_df
        self._join_cols = join_cols

    @property
    def join_cols(self) -> List[str]:
        """Returns list of columns to be joined on."""
        return self._join_cols.copy()

    @property
    def public_df(self) -> DataFrame:
        """Returns Pandas DataFrame being joined with."""
        return self._public_df

    @property
    def stability(self) -> int:
        """Returns stability of public join.

        The stability is the maximum count of any combination of values in the join
        columns.
        """
        return self._join_stability

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial (add link?) for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in) * self.stability

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Perform public join.

        Args:
            sdf: Private DataFrame to join public DataFrame with.
        """
        output_columns_order = list(
            (cast(SparkDataFrameDomain, self.output_domain)).schema
        )
        for col in self._overlapping_cols:
            sdf = sdf.withColumnRenamed(col, f"{col}_left")
        if not self._join_on_nulls:
            return sdf.join(self.public_df, on=self.join_cols, how="inner").select(
                output_columns_order
            )
        joined_df = sdf.join(
            self.public_df,
            on=reduce(
                lambda exp, col: exp & sdf[col].eqNullSafe(self.public_df[col]),
                self.join_cols,
                sf.lit(True),  # pylint: disable=no-member
            ),
        )
        for col in self.join_cols:
            joined_df = joined_df.drop(self.public_df[col])
        return joined_df.select(output_columns_order)


class TruncationStrategy(Enum):
    """Enumerating truncation strategies for PrivateJoin.

    See :meth:`~.PrivateJoin.stability_function` for the stability of each strategy.
    """

    TRUNCATE = 1
    """Use :func:`~.truncate_large_groups`."""
    DROP = 2
    """Use :func:`~.drop_large_groups`."""


class PrivateJoin(Transformation):
    r"""Join two private SparkDataFrames.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> left_spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a1", "a1", "a1", "a2"],
            ...             "B": ["b1", "b1", "b1", "b2", "b2", "b1"],
            ...             "X": [2, 3, 5, -1, 4, -5],
            ...         }
            ...     )
            ... )
            >>> right_spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "B": ["b1", "b2", "b2"],
            ...             "C": ["c1", "c2", "c3"],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(left_spark_dataframe)
            A   B  X
        0  a1  b1  2
        1  a1  b1  3
        2  a1  b1  5
        3  a1  b2 -1
        4  a1  b2  4
        5  a2  b1 -5
        >>> print_sdf(right_spark_dataframe)
            B   C
        0  b1  c1
        1  b2  c2
        2  b2  c3
        >>> # Create transformation
        >>> left_domain = SparkDataFrameDomain(
        ...     {
        ...         "A": SparkStringColumnDescriptor(),
        ...         "B": SparkStringColumnDescriptor(),
        ...         "X": SparkIntegerColumnDescriptor(),
        ...     },
        ... )
        >>> assert left_spark_dataframe in left_domain
        >>> right_domain = SparkDataFrameDomain(
        ...     {
        ...         "B": SparkStringColumnDescriptor(),
        ...         "C": SparkStringColumnDescriptor(),
        ...     },
        ... )
        >>> assert right_spark_dataframe in right_domain
        >>> private_join = PrivateJoin(
        ...     input_domain=DictDomain(
        ...         {
        ...             "left": left_domain,
        ...             "right": right_domain,
        ...         }
        ...     ),
        ...     left_key="left",
        ...     right_key="right",
        ...     left_truncation_strategy=TruncationStrategy.TRUNCATE,
        ...     left_truncation_threshold=2,
        ...     right_truncation_strategy=TruncationStrategy.TRUNCATE,
        ...     right_truncation_threshold=2,
        ... )
        >>> input_dictionary = {
        ...     "left": left_spark_dataframe,
        ...     "right": right_spark_dataframe
        ... }
        >>> # Apply transformation to data
        >>> joined_dataframe = private_join(input_dictionary)
        >>> print_sdf(joined_dataframe)
            B   A  X   C
        0  b1  a1  5  c1
        1  b1  a2 -5  c1
        2  b2  a1 -1  c2
        3  b2  a1 -1  c3
        4  b2  a1  4  c2
        5  b2  a1  4  c3

    .. Note:
        This join works similarly to :class:`~.PublicJoin`, see it for more examples.

    Transformation Contract:
        * Input domain - :class:`~.DictDomain` containing two SparkDataFrame domains.
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.DictMetric` with :class:`~.SymmetricDifference` for
          each input.
        * Output metric - :class:`~.SymmetricDifference`

        >>> private_join.input_domain
        DictDomain(key_to_domain={'left': SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)}), 'right': SparkDataFrameDomain(schema={'B': SparkStringColumnDescriptor(allow_null=False), 'C': SparkStringColumnDescriptor(allow_null=False)})})
        >>> private_join.output_domain
        SparkDataFrameDomain(schema={'B': SparkStringColumnDescriptor(allow_null=False), 'A': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64), 'C': SparkStringColumnDescriptor(allow_null=False)})
        >>> private_join.input_metric
        DictMetric(key_to_metric={'left': SymmetricDifference(), 'right': SymmetricDifference()})
        >>> private_join.output_metric
        SymmetricDifference()

        Stability Guarantee:
            Let :math:`T_l` and :math:`T_r` be the left and right truncation strategies
            with stabilities :math:`s_l` and :math:`s_r` and thresholds :math:`\tau_l`
            and :math:`\tau_r`.

            :class:`~.PublicJoin`'s :meth:`~.stability_function` returns

            .. math::

                \tau_l \cdot s_r \cdot (df_{r1} \Delta df_{r2}) +
                \tau_r \cdot s_l \cdot (df_{l1} \Delta df_{l2})

            where:

            * :math:`df_{r1} \Delta df_{r2}` is `d_in[self.right]`
            * :math:`df_{l1} \Delta df_{l2}` is `d_in[self.left]`

            - TruncationStrategy.DROP has a stability equal to the truncation
              threshold (This is because adding a row can cause a number of rows equal
              to the truncation threshold to be dropped).
            - TruncationStrategy.TRUNCATE has a stability of 2 (This is because
              adding a new row can not only add a new row to the output, it also can
              displace another row)

            >>> # TRUNCATE has a stability of 2
            >>> s_r = s_l = private_join.truncation_strategy_stability(
            ...     TruncationStrategy.TRUNCATE, 1
            ... )
            >>> tau_r = tau_l = 2
            >>> tau_l * s_r * 1 + tau_r * s_l * 1
            8
            >>> private_join.stability_function({"left": 1, "right": 1})
            8
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        left_key: Any,
        right_key: Any,
        left_truncation_strategy: TruncationStrategy,
        right_truncation_strategy: TruncationStrategy,
        left_truncation_threshold: int,
        right_truncation_threshold: int,
        join_cols: Optional[List[str]] = None,
        join_on_nulls: bool = False,
    ):
        r"""Constructor.

        The following conditions are checked:

            - `input_domain` is a DictDomain with 2
              :class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain`\ s.
            - `left` and `right` are the two keys in the input domain.
            - `join_cols` is not empty, when provided or computed (if None).
            - Columns in `join_cols` are common to both tables.
            - Columns in `join_cols` have matching column types in both tables.

        Args:
            input_domain: Domain of input dictionaries (with exactly two keys).
            left_key: Key for the left DataFrame.
            right_key: Key for the right DataFrame.
            left_truncation_strategy: :class:`~.TruncationStrategy` to use for
                truncating the left DataFrame.
            right_truncation_strategy:  :class:`~.TruncationStrategy` to use for
                truncating the right DataFrame.
            left_truncation_threshold: The maximum number of rows to allow for each
                combination of values of `join_cols` in the left DataFrame.
            right_truncation_threshold: The maximum number of rows to allow for each
                combination of values of `join_cols` in the right DataFrame.
            join_cols: Columns to perform join on. If None, or empty, natural join is
                computed.
            join_on_nulls: If True, null values on corresponding join columns of
                both dataframes will be considered to be equal.
        """
        if input_domain.length != 2:
            raise ValueError("Input domain must be a DictDomain with 2 keys.")
        if left_key == right_key:
            raise ValueError("Left and right keys must be distinct.")
        if left_key not in input_domain.key_to_domain:
            raise ValueError(f"Invalid key: Key '{left_key}' not in input domain.")
        if right_key not in input_domain.key_to_domain:
            raise ValueError(f"Invalid key: Key '{right_key}' not in input domain.")

        left_domain, right_domain = input_domain[left_key], input_domain[right_key]
        if not isinstance(left_domain, SparkDataFrameDomain) or not isinstance(
            right_domain, SparkDataFrameDomain
        ):
            raise ValueError("Input domain must be SparkDataFrameDomin for both keys.")

        common_cols = set(left_domain.schema) & set(right_domain.schema)
        if not join_cols:
            if not common_cols:
                raise ValueError("Can not join: No common columns.")
            join_cols = sorted(common_cols, key=list(left_domain.schema).index)
        else:
            join_cols = join_cols.copy()

        join_cols_schema = {}
        for key in join_cols:
            if left_domain[key] != right_domain[key]:
                raise ValueError(
                    "Left and right DataFrame domains have mismatching types on"
                    f" join column {key}."
                )
            if join_on_nulls:
                join_cols_schema[key] = left_domain[key]
            else:
                join_cols_schema[key] = replace(left_domain[key], allow_null=False)
        overlapping_cols = common_cols - set(join_cols)
        all_input_cols = set(left_domain.schema) | set(right_domain.schema)
        for col in overlapping_cols:
            if f"{col}_left" in all_input_cols or f"{col}_right" in all_input_cols:
                raise ValueError(
                    f"Join would rename overlapping column '{col}' to an existing"
                    " column name."
                )

        left_schema = {
            col + ("_left" if col in overlapping_cols else ""): left_domain[col]
            for col in left_domain.schema
            if col not in join_cols
        }
        right_schema = {
            col + ("_right" if col in overlapping_cols else ""): right_domain[col]
            for col in right_domain.schema
            if col not in join_cols
        }

        output_domain = SparkDataFrameDomain(
            {**join_cols_schema, **left_schema, **right_schema}
        )

        super().__init__(
            input_domain=input_domain,
            input_metric=DictMetric(
                {left_key: SymmetricDifference(), right_key: SymmetricDifference()}
            ),
            output_domain=output_domain,
            output_metric=SymmetricDifference(),
        )
        self._left_key = left_key
        self._right_key = right_key
        self._left_truncation_strategy = left_truncation_strategy
        self._right_truncation_strategy = right_truncation_strategy
        self._left_truncation_threshold = left_truncation_threshold
        self._right_truncation_threshold = right_truncation_threshold
        self._join_cols = join_cols
        self._overlapping_cols = set(common_cols) - set(join_cols)
        self._join_on_nulls = join_on_nulls

    @property
    def left_key(self) -> Any:
        """Returns key to left DataFrame."""
        return self._left_key

    @property
    def right_key(self) -> Any:
        """Returns key to right DataFrame."""
        return self._right_key

    @property
    def left_truncation_strategy(self) -> TruncationStrategy:
        """Returns TruncationStrategy for truncating the left DataFrame."""
        return self._left_truncation_strategy

    @property
    def right_truncation_strategy(self) -> TruncationStrategy:
        """Returns TruncationStrategy for truncating the right DataFrame."""
        return self._right_truncation_strategy

    @property
    def left_truncation_threshold(self) -> int:
        """Returns the threshold for truncating the left DataFrame."""
        return self._left_truncation_threshold

    @property
    def right_truncation_threshold(self) -> int:
        """Returns the threshold for truncating the right DataFrame."""
        return self._right_truncation_threshold

    @property
    def join_cols(self) -> List[str]:
        """Returns list of column names to join on."""
        return self._join_cols.copy()

    @property
    def join_on_nulls(self) -> bool:
        """Returns whether to consider null equal to null."""
        return self._join_on_nulls

    @staticmethod
    def truncation_strategy_stability(
        truncation_strategy: TruncationStrategy, threshold: int
    ) -> int:
        """Returns the stability for the given truncation strategy."""
        return {TruncationStrategy.TRUNCATE: 2, TruncationStrategy.DROP: threshold}[
            truncation_strategy
        ]

    @typechecked
    def stability_function(self, d_in: Dict[Any, ExactNumberInput]) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        tau_l = self.left_truncation_threshold
        tau_r = self.right_truncation_threshold
        s_l = self.truncation_strategy_stability(self.left_truncation_strategy, tau_l)
        s_r = self.truncation_strategy_stability(self.right_truncation_strategy, tau_r)
        d_in_l = ExactNumber(d_in[self.left_key])
        d_in_r = ExactNumber(d_in[self.right_key])
        return tau_l * s_r * d_in_r + tau_r * s_l * d_in_l

    def __call__(self, dfs: Dict[Any, DataFrame]) -> DataFrame:
        """Perform join."""

        def truncate(df: DataFrame, strategy: TruncationStrategy, threshold: int):
            if strategy == TruncationStrategy.TRUNCATE:
                return truncate_large_groups(df, self.join_cols, threshold)
            elif strategy == TruncationStrategy.DROP:
                return drop_large_groups(df, self.join_cols, threshold)
            else:
                raise AssertionError("Unsupported TruncationStrategy")

        left = truncate(
            dfs[self.left_key],
            self.left_truncation_strategy,
            self.left_truncation_threshold,
        )
        right = truncate(
            dfs[self.right_key],
            self.right_truncation_strategy,
            self.right_truncation_threshold,
        )

        for col in self._overlapping_cols:
            left = left.withColumnRenamed(col, f"{col}_left")
            right = right.withColumnRenamed(col, f"{col}_right")

        if not self._join_on_nulls:
            right = right.dropna()
            return left.join(right, on=self.join_cols, how="inner")

        joined_df = left.join(
            right, on=[left[col].eqNullSafe(right[col]) for col in self.join_cols]
        )
        for col in self.join_cols:
            joined_df = joined_df.drop(right[col])
        output_columns_order = list(
            (cast(SparkDataFrameDomain, self.output_domain)).schema
        )
        return joined_df.select(output_columns_order)


class PrivateJoinOnKey(Transformation):
    r"""Join two private SparkDataFrames including a key column.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkDataFrameDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> left_spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a1", "a1", "a1", "a2"],
            ...             "B": ["b1", "b1", "b1", "b2", "b2", "b1"],
            ...             "X": [2, 3, 5, -1, 4, -5],
            ...         }
            ...     )
            ... )
            >>> right_spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "B": ["b1", "b2", "b2"],
            ...             "C": ["c1", "c2", "c3"],
            ...         }
            ...     )
            ... )
            >>> # This input dataframe is not involved in the join but will be included in the output
            >>> ignored_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "B": ["b1", "b2", "b2"],
            ...             "D": ["d1", "d1", "d2"],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(left_spark_dataframe)
            A   B  X
        0  a1  b1  2
        1  a1  b1  3
        2  a1  b1  5
        3  a1  b2 -1
        4  a1  b2  4
        5  a2  b1 -5
        >>> print_sdf(right_spark_dataframe)
            B   C
        0  b1  c1
        1  b2  c2
        2  b2  c3
        >>> print_sdf(ignored_dataframe)
            B   D
        0  b1  d1
        1  b2  d1
        2  b2  d2
        >>> # Create transformation
        >>> left_domain = SparkDataFrameDomain(
        ...     {
        ...         "A": SparkStringColumnDescriptor(),
        ...         "B": SparkStringColumnDescriptor(),
        ...         "X": SparkIntegerColumnDescriptor(),
        ...     },
        ... )
        >>> assert left_spark_dataframe in left_domain
        >>> right_domain = SparkDataFrameDomain(
        ...     {
        ...         "B": SparkStringColumnDescriptor(),
        ...         "C": SparkStringColumnDescriptor(),
        ...     },
        ... )
        >>> assert right_spark_dataframe in right_domain
        >>> ignored_domain = SparkDataFrameDomain(
        ...     {
        ...         "B": SparkStringColumnDescriptor(),
        ...         "D": SparkStringColumnDescriptor(),
        ...     },
        ... )
        >>> assert ignored_dataframe in ignored_domain
        >>> private_join = PrivateJoinOnKey(
        ...     input_domain=DictDomain(
        ...         {
        ...             "left": left_domain,
        ...             "right": right_domain,
        ...             "ignored": ignored_domain,
        ...         }
        ...     ),
        ...     input_metric=AddRemoveKeys(
        ...         {
        ...            "left": "B",
        ...            "right": "B",
        ...            "ignored": "B",
        ...         }
        ...     ),
        ...     left_key="left",
        ...     right_key="right",
        ...     new_key="joined",
        ... )
        >>> input_dictionary = {
        ...     "left": left_spark_dataframe,
        ...     "right": right_spark_dataframe,
        ...     "ignored": ignored_dataframe,
        ... }
        >>> # Apply transformation to data
        >>> output_dictionary = private_join(input_dictionary)
        >>> assert left_spark_dataframe is output_dictionary["left"]
        >>> assert right_spark_dataframe is output_dictionary["right"]
        >>> assert ignored_dataframe is output_dictionary["ignored"]
        >>> joined_dataframe = output_dictionary["joined"]
        >>> print_sdf(joined_dataframe)
            B   A  X   C
        0  b1  a1  2  c1
        1  b1  a1  3  c1
        2  b1  a1  5  c1
        3  b1  a2 -5  c1
        4  b2  a1 -1  c2
        5  b2  a1 -1  c3
        6  b2  a1  4  c2
        7  b2  a1  4  c3

    .. Note:
        This join works similarly to :class:`~.PublicJoin`, see it for more examples.

    .. Note:
        Unlike :class:`~.PrivateJoin`, this join allows for other dataframes to be present in the input dictionary, and
        will output a dictionary containing all of the input dataframes along with the joined dataframe.
        This is because of the stability analysis for AddRemoveKeys. See :mod:`~.add_remove_keys` for more details.

    Transformation Contract:
        * Input domain - :class:`~.DictDomain` containing two or more SparkDataFrame domains.
        * Output domain - The same as the input :class:`~.DictDomain` with the addition of a new
          :class:`~.SparkDataFrameDomain` for the joined table.
        * Input metric - :class:`~.AddRemoveKeys`
        * Output metric - :class:`~.AddRemoveKeys`

    >>> private_join.input_domain
    DictDomain(key_to_domain={'left': SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)}), 'right': SparkDataFrameDomain(schema={'B': SparkStringColumnDescriptor(allow_null=False), 'C': SparkStringColumnDescriptor(allow_null=False)}), 'ignored': SparkDataFrameDomain(schema={'B': SparkStringColumnDescriptor(allow_null=False), 'D': SparkStringColumnDescriptor(allow_null=False)})})
    >>> private_join.output_domain
    DictDomain(key_to_domain={'left': SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)}), 'right': SparkDataFrameDomain(schema={'B': SparkStringColumnDescriptor(allow_null=False), 'C': SparkStringColumnDescriptor(allow_null=False)}), 'ignored': SparkDataFrameDomain(schema={'B': SparkStringColumnDescriptor(allow_null=False), 'D': SparkStringColumnDescriptor(allow_null=False)}), 'joined': SparkDataFrameDomain(schema={'B': SparkStringColumnDescriptor(allow_null=False), 'A': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64), 'C': SparkStringColumnDescriptor(allow_null=False)})})
    >>> private_join.input_metric
    AddRemoveKeys(df_to_key_column={'left': 'B', 'right': 'B', 'ignored': 'B'})
    >>> private_join.output_metric
    AddRemoveKeys(df_to_key_column={'left': 'B', 'right': 'B', 'ignored': 'B', 'joined': 'B'})

    Stability Guarantee:
        :class:`~.PrivateJoinOnKey`'s :meth:`~.stability_function` returns `d_in`

        >>> private_join.stability_function(1)
        1
        >>> private_join.stability_function(2)
        2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        left_key: Any,
        right_key: Any,
        new_key: Any,
        join_cols: Optional[List[str]] = None,
        join_on_nulls: bool = False,
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input dictionaries. Must contain `left_key` and `right_key`,
                but may also contain other keys.
            input_metric: AddRemoveKeys metric for the input dictionaries. The left and right dataframes
                must use the same key column.
            left_key: Key for the left DataFrame.
            right_key: Key for the right DataFrame.
            new_key: Key for the output DataFrame.
            join_cols: Columns to perform join on. If None, or empty, natural join is
                computed.
            join_on_nulls: If True, null values on corresponding join columns of
                both dataframes will be considered to be equal.
        """
        if left_key == right_key:
            raise ValueError("Left and right keys must be distinct.")
        if left_key not in input_domain.key_to_domain:
            raise ValueError(f"Invalid key: Key '{left_key}' not in input domain.")
        if right_key not in input_domain.key_to_domain:
            raise ValueError(f"Invalid key: Key '{right_key}' not in input domain.")

        left_domain, right_domain = input_domain[left_key], input_domain[right_key]
        if not isinstance(left_domain, SparkDataFrameDomain) or not isinstance(
            right_domain, SparkDataFrameDomain
        ):
            raise ValueError("Input domain must be SparkDataFrameDomin for both keys.")

        common_cols = set(left_domain.schema) & set(right_domain.schema)
        if not join_cols:
            if not common_cols:
                raise ValueError("Can not join: No common columns.")
            join_cols = sorted(common_cols, key=list(left_domain.schema).index)
        else:
            join_cols = join_cols.copy()

        join_cols_schema = {}
        for key in join_cols:
            if left_domain[key] != right_domain[key]:
                raise ValueError(
                    "Left and right DataFrame domains have mismatching types on"
                    f" join column {key}."
                )
            if join_on_nulls:
                join_cols_schema[key] = left_domain[key]
            else:
                join_cols_schema[key] = replace(left_domain[key], allow_null=False)
        overlapping_cols = common_cols - set(join_cols)
        all_input_cols = set(left_domain.schema) | set(right_domain.schema)
        for col in overlapping_cols:
            if f"{col}_left" in all_input_cols or f"{col}_right" in all_input_cols:
                raise ValueError(
                    f"Join would rename overlapping column '{col}' to an existing"
                    " column name."
                )

        left_schema = {
            col + ("_left" if col in overlapping_cols else ""): left_domain[col]
            for col in left_domain.schema
            if col not in join_cols
        }
        right_schema = {
            col + ("_right" if col in overlapping_cols else ""): right_domain[col]
            for col in right_domain.schema
            if col not in join_cols
        }

        new_df_domain = SparkDataFrameDomain(
            {**join_cols_schema, **left_schema, **right_schema}
        )
        output_domain = DictDomain(
            {**input_domain.key_to_domain, new_key: new_df_domain}
        )

        if left_key not in input_metric.df_to_key_column:
            raise ValueError(f"Invalid key: Key '{left_key}' not in input metric.")
        if right_key not in input_metric.df_to_key_column:
            raise ValueError(f"Invalid key: Key '{right_key}' not in input metric.")
        if (
            input_metric.df_to_key_column[left_key]
            != input_metric.df_to_key_column[right_key]
        ):
            raise ValueError("Left and right keys must have the same key column.")
        key_column = input_metric.df_to_key_column[left_key]
        if key_column not in join_cols:
            raise ValueError("Key column must be joined on.")

        output_metric = AddRemoveKeys(
            {**input_metric.df_to_key_column, new_key: key_column}
        )

        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )
        self._left_key = left_key
        self._right_key = right_key
        self._new_key = new_key
        self._join_cols = join_cols
        self._overlapping_cols = overlapping_cols
        self._join_on_nulls = join_on_nulls

    @property
    def left_key(self) -> Any:
        """Returns key to left DataFrame."""
        return self._left_key

    @property
    def right_key(self) -> Any:
        """Returns key to right DataFrame."""
        return self._right_key

    @property
    def new_key(self) -> Any:
        """Returns key to output DataFrame."""
        return self._new_key

    @property
    def join_cols(self) -> List[str]:
        """Returns list of column names to join on."""
        return self._join_cols.copy()

    @property
    def join_on_nulls(self) -> bool:
        """Returns whether to consider null equal to null."""
        return self._join_on_nulls

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, dfs: Dict[Any, DataFrame]):
        """Perform join."""
        left = dfs[self.left_key]
        right = dfs[self.right_key]
        for col in self._overlapping_cols:
            left = left.withColumnRenamed(col, f"{col}_left")
            right = right.withColumnRenamed(col, f"{col}_right")

        if not self._join_on_nulls:
            right = right.dropna()
            joined_df = left.join(right, on=self.join_cols, how="inner")
        else:
            joined_df = left.join(
                right, on=[left[col].eqNullSafe(right[col]) for col in self.join_cols]
            )
            for col in self.join_cols:
                joined_df = joined_df.drop(right[col])
        output_columns_order = list(
            (
                cast(
                    SparkDataFrameDomain,
                    cast(DictDomain, self.output_domain).key_to_domain[self.new_key],
                )
            ).schema
        )
        new_dfs = dfs.copy()
        new_dfs[self.new_key] = joined_df.select(output_columns_order)
        return new_dfs
