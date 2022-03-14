"""Transformations for joining Spark DataFrames."""
# TODO(#1320): Add links to privacy and stability tutorial

# <placeholder: boilerplate>

from abc import abstractmethod
from functools import reduce
from typing import Any, Dict, List, Optional, Union, cast

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from typeguard import typechecked

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
)
from tmlt.core.metrics import DictMetric, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import get_nonconflicting_string


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
            input_domain: Domain of the input SparkDataFrames.
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
                public and private dataframes will be considered to be equal.
        """
        if isinstance(metric, IfGroupedBy):
            if not isinstance(metric.inner_metric.inner_metric, SymmetricDifference):
                raise ValueError(
                    "Inner metric for IfGroupedBy metric must be SymmetricDifference."
                )
            if metric.column not in input_domain.schema:
                raise ValueError(
                    f"Invalid IfGroupedBy metric: {metric.column} not in input domain."
                )

        common_cols = set(input_domain.schema) & set(public_df.columns)
        if not join_cols:
            if not common_cols:
                raise ValueError("Can not join: No common columns.")
            join_cols = sorted(common_cols, key=list(input_domain.schema).index)

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


class Truncation(Transformation):
    """Transforms a Spark DataFrame so that each group has at most `threshold` rows."""

    @typechecked
    def __init__(self, domain: SparkDataFrameDomain, keys: List[str], threshold: int):
        """Constructor.

        Args:
            domain: Domain of input DataFrames.
            keys: Keys to truncate on. (See `threshold` hold below)
            threshold: Truncation threshold. Truncation is performed on rows with
                (tuple) value `v` (for the given keys) only if the multiplicity of `v`
                in input DataFrame is strictly larger than threshold value.
        """
        if not keys:
            raise ValueError("No key provided for truncation.")
        if len(keys) != len(set(keys)):
            raise ValueError("Truncation keys must be distinct.")
        missing_keys = set(keys) - set(domain.schema)
        if missing_keys:
            raise ValueError(f"Truncation keys not in domain: {missing_keys}")
        if threshold <= 0:
            raise ValueError("Truncation threshold must be a positive integer.")
        super().__init__(
            input_domain=domain,
            input_metric=SymmetricDifference(),
            output_domain=domain,
            output_metric=SymmetricDifference(),
        )
        self._threshold = threshold
        self._keys = keys

    @property
    def keys(self) -> List[str]:
        """Returns truncation keys."""
        return self._keys.copy()

    @property
    def threshold(self) -> int:
        """Returns truncation threshold."""
        return self._threshold

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial (add link?) for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in) * self.stability

    @property
    @abstractmethod
    def stability(self) -> int:
        """Returns stability of the truncation."""
        ...


class HashTopKTruncation(Truncation):
    """Order by output of a hash function and keep top K rows for each group.

    For a given threshold T, this transformation performs the following steps:
        - Each row in the input DataFrame is hashed to produce a new int column.
        - For each group (identified by values in the key columns), rows in the group
            are sorted by the hash column.
        - For groups containing more than T rows, all but the first T rows are dropped.

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
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a1", "a1", "a1", "a2"],
            ...             "B": ["b1", "b1", "b1", "b2", "b2", "b1"],
            ...             "X": [2, 3, 5, -1, 4, -5],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B  X
        0  a1  b1  2
        1  a1  b1  3
        2  a1  b1  5
        3  a1  b2 -1
        4  a1  b2  4
        5  a2  b1 -5
        >>> # Create the transformation
        >>> truncate = HashTopKTruncation(
        ...     domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     keys=["A", "B"],
        ...     threshold=2,
        ... )
        >>> # Apply transformation to data
        >>> truncated_spark_dataframe = truncate(spark_dataframe)
        >>> print_sdf(truncated_spark_dataframe)
            A   B  X
        0  a1  b1  2
        1  a1  b1  5
        2  a1  b2 -1
        3  a1  b2  4
        4  a2  b1 -5

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference`
        * Output metric - :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> truncate.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> truncate.input_metric
        SymmetricDifference()
        >>> truncate.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.Truncation`'s :meth:`~.stability_function` returns the `d_in`
            times 2. (This is because a row that is added or removed can not only become
            included in the top K, but can also displace another row from being included
            in the top K.)

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # pylint: disable=line-too-long

    def __call__(self, df: DataFrame) -> DataFrame:
        """Perform the transformation."""
        index_col_name = get_nonconflicting_string(df.columns)
        hash_col_name = get_nonconflicting_string(df.columns + [index_col_name])
        shuffled_partitions = Window.partitionBy(*self.keys).orderBy(
            hash_col_name, *df.columns
        )
        return (
            df.withColumn(
                hash_col_name, sf.hash(*df.columns)  # pylint: disable=no-member
            )
            .withColumn(
                index_col_name,
                sf.row_number().over(shuffled_partitions),  # pylint: disable=no-member
            )
            .filter(f"{index_col_name}<={self.threshold}")
            .drop(index_col_name, hash_col_name)
        )

    @property
    def stability(self) -> int:
        """Returns stability of the truncation.

        The stability is 2, because a row that is added or removed can not only become
        included in the top K, but can also displace another row from being included
        in the top K.
        """
        return 2


class DropAllTruncation(Truncation):
    """Drop all records having key value with multiplicity greater than threshold.

    Args:
        domain: Domain of input DataFrames.
        keys: Keys to truncate on. (See `threshold` hold below)
        threshold: Truncation threshold. Truncation is performed on rows with
            (tuple) value `v` (for the given keys) only if the multiplicity of `v`
            in input DataFrame is strictly larger than threshold value.

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
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a1", "a1", "a1", "a1", "a2"],
            ...             "B": ["b1", "b1", "b1", "b2", "b2", "b1"],
            ...             "X": [2, 3, 5, -1, 4, -5],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B  X
        0  a1  b1  2
        1  a1  b1  3
        2  a1  b1  5
        3  a1  b2 -1
        4  a1  b2  4
        5  a2  b1 -5
        >>> # Create the transformation
        >>> truncate = DropAllTruncation(
        ...     domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...             "X": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     keys=["A", "B"],
        ...     threshold=2,
        ... )
        >>> # Apply transformation to data
        >>> truncated_spark_dataframe = truncate(spark_dataframe)
        >>> print_sdf(truncated_spark_dataframe)
            A   B  X
        0  a1  b2 -1
        1  a1  b2  4
        2  a2  b1 -5

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference`
        * Output metric - :class:`~.SymmetricDifference`

        >>> truncate.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> truncate.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'X': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> truncate.input_metric
        SymmetricDifference()
        >>> truncate.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.Truncation`'s :meth:`~.stability_function` returns the `d_in` times
            the specified `threshold`.

            >>> truncate.stability_function(1)
            2
            >>> truncate.stability_function(2)
            4
    """  # pylint: disable=line-too-long

    def __call__(self, df: DataFrame) -> DataFrame:
        """Perform the transformation."""
        count_col_name = get_nonconflicting_string(df.columns)
        partitions = Window.partitionBy(*self.keys)
        return (
            df.withColumn(
                count_col_name,
                sf.count(sf.lit(1)).over(partitions),  # pylint: disable=no-member
            )
            .filter(f"{count_col_name}<={self.threshold}")
            .drop(count_col_name)
        )

    @property
    def stability(self) -> int:
        """Returns stability of the truncation.

        The stability is self.threshold, because a row that is added or removed can
        cause all rows with the same key to be dropped/not dropped.
        """
        return self.threshold


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
        ...     left="left",
        ...     right="right",
        ...     left_truncator=HashTopKTruncation(
        ...         domain=left_domain,
        ...         keys=["B"],
        ...         threshold=2,
        ...     ),
        ...     right_truncator=HashTopKTruncation(
        ...         domain=right_domain,
        ...         keys=["B"],
        ...         threshold=2,
        ...     ),
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
            Let :math:`T_l` and :math:`T_r` be the left and right truncation operators with
            stabilities :math:`s_l` and :math:`s_r` and thresholds :math:`\tau_l` and
            :math:`\tau_r`.

            :class:`~.PublicJoin`'s :meth:`~.stability_function` returns

            .. math::

                \tau_l \cdot s_r \cdot (df_{r1} \Delta df_{r2}) +
                \tau_r \cdot s_l \cdot (df_{l1} \Delta df_{l2})

            where:

            * :math:`df_{r1} \Delta df_{r2}` is `d_in[self.right]`
            * :math:`df_{l1} \Delta df_{l2}` is `d_in[self.left]`

            >>> # Both example transformations had a stability of 2
            >>> s_r = s_l = tau_r = tau_l = 2
            >>> tau_l * s_r * 1 + tau_r * s_r * 1
            8
            >>> private_join.stability_function({"left": 1, "right": 1})
            8
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        left: Any,
        right: Any,
        left_truncator: Truncation,
        right_truncator: Truncation,
        join_cols: Optional[List[str]] = None,
    ):
        r"""Constructor.

        The following conditions are checked:

            - `input_domain` is a DictDomain with 2
              :class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain`\ s.
            - `left` and `right` are the two keys in the input domain.
            - `left_truncator` and `right_truncator` have the same domains as
              the corresponding entries for `left` and `right` in the
              `input_domain`.
            - `left_truncator` and `right_truncator` both operate on `join_cols`.
            - `join_cols` is not empty, when provided or computed (if None).
            - Columns in `join_cols` are common to both tables.
            - Columns in `join_cols` have matching column types in both tables.

        Args:
            input_domain: Domain of input dictionaries (with exactly two keys).
            left: Key for left DataFrame.
            right: Key for right DataFrame.
            left_truncator: Truncation transformation for truncating left DataFrame.
            right_truncator: Truncation transformation for truncating right DataFrame.
            join_cols: Columns to perform join on. If None, or empty, natural join is
                computed.
        """
        if input_domain.length != 2:
            raise ValueError("Input domain must be a DictDomain with 2 keys.")
        if left == right:
            raise ValueError("Left and right keys must be distinct.")
        if left not in input_domain.key_to_domain:
            raise ValueError(f"Invalid key: Key '{left}' not in input domain.")
        if right not in input_domain.key_to_domain:
            raise ValueError(f"Invalid key: Key '{right}' not in input domain.")

        left_domain, right_domain = input_domain[left], input_domain[right]
        if not isinstance(left_domain, SparkDataFrameDomain) or not isinstance(
            right_domain, SparkDataFrameDomain
        ):
            raise ValueError("Input domain must be SparkDataFrameDomin for both keys.")
        if left_truncator.input_domain != left_domain:
            raise ValueError("Input domain for left_truncator does not match left key.")
        if right_truncator.input_domain != right_domain:
            raise ValueError(
                "Input domain for right_truncator does not match right key."
            )

        common_cols = set(left_domain.schema) & set(right_domain.schema)
        if not join_cols:
            if not common_cols:
                raise ValueError("Can not join: No common columns.")
            join_cols = sorted(common_cols, key=list(left_domain.schema).index)

        if not left_truncator.keys == join_cols == right_truncator.keys:
            raise ValueError("Truncation keys must match join columns.")

        join_cols_schema = {}
        for key in join_cols:
            if left_domain[key] != right_domain[key]:
                raise ValueError(
                    "Left and right DataFrame domains have mismatching types on"
                    f" join column {key}."
                )
            join_cols_schema[key] = left_domain[key]
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
                {left: SymmetricDifference(), right: SymmetricDifference()}
            ),
            output_domain=output_domain,
            output_metric=SymmetricDifference(),
        )
        self._left = left
        self._left_truncator = left_truncator
        self._right = right
        self._right_truncator = right_truncator
        self._join_cols = join_cols
        self._overlapping_cols = set(common_cols) - set(join_cols)

    @property
    def left(self) -> Any:
        """Returns key to left DataFrame."""
        return self._left

    @property
    def right(self) -> Any:
        """Returns key to right DataFrame."""
        return self._right

    @property
    def left_truncator(self) -> Truncation:
        """Returns Truncation transformation for truncating left DataFrame."""
        return self._left_truncator

    @property
    def right_truncator(self) -> Truncation:
        """Returns Truncation transformation for truncating right DataFrame."""
        return self._right_truncator

    @property
    def join_cols(self) -> List[str]:
        """Returns list of column names to join on."""
        return self._join_cols.copy()

    @typechecked
    def stability_function(self, d_in: Dict[str, ExactNumberInput]) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        left_d_mid = self.left_truncator.stability_function(d_in[self.left])
        right_d_mid = self.right_truncator.stability_function(d_in[self.right])
        return (
            self.left_truncator.threshold * right_d_mid
            + self.right_truncator.threshold * left_d_mid
        )

    def __call__(self, dfs: Dict[Any, DataFrame]) -> DataFrame:
        """Perform join."""
        left = self._left_truncator(dfs[self._left])
        right = self._right_truncator(dfs[self._right])

        for col in self._overlapping_cols:
            # Assumes {col}_left, {col}_right not already taken.
            left = left.withColumnRenamed(col, f"{col}_left")
            right = right.withColumnRenamed(col, f"{col}_right")

        return left.join(right, on=self._join_cols, how="inner")
