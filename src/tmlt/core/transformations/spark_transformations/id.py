"""Add a column containing a unique id for each row in a Spark DataFrame."""


# TODO(#1320): Add link to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window
from typeguard import typechecked

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import get_nonconflicting_string


class AddUniqueColumn(Transformation):
    """Adds a column containing a unique ID for each row.

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
        >>> add_unique_column = AddUniqueColumn(
        ...     input_domain=SparkDataFrameDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True),
        ...         }
        ...     ),
        ...     column="ID",
        ... )
        >>> # Apply transformation to data
        >>> output_dataframe = add_unique_column(spark_dataframe)
        >>> output_dataframe.sort("A").show(truncate=False)
        +----+----+--------------------------------+
        |A   |B   |ID                              |
        +----+----+--------------------------------+
        |null|NaN |5B6E756C6C2C224E614E222C2231225D|
        |a1  |0.1 |5B226131222C22302E31222C2231225D|
        |a2  |null|5B226132222C6E756C6C2C2231225D  |
        +----+----+--------------------------------+
        <BLANKLINE>

        Transformation Contract:
            * Input domain - :class:`~.SparkDataFrameDomain`
            * Output domain - :class:`~.SparkDataFrameDomain`
            * Input metric - :class:`~.SymmetricDifference`
            * Output metric - :class:`~.IfGroupedBy` over :class:`~.SymmetricDifference`

            >>> add_unique_column.input_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True, allow_null=False, size=64)})
            >>> add_unique_column.output_domain
            SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True, allow_null=False, size=64), 'ID': SparkStringColumnDescriptor(allow_null=False)})
            >>> add_unique_column.input_metric
            SymmetricDifference()
            >>> add_unique_column.output_metric
            IfGroupedBy(column='ID', inner_metric=SymmetricDifference())

            Stability Guarantee:
                :class:`~.AddUniqueColumn`'s :meth:`~.stability_function` returns `d_in`.

                >>> add_unique_column.stability_function(1)
                1
                >>> add_unique_column.stability_function(2)
                2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(self, input_domain: SparkDataFrameDomain, column: str):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrames.
            column: Name of the id column.
        """
        if column in input_domain.schema:
            raise ValueError(f"Column name ({column}) already exists.")

        super().__init__(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            output_domain=SparkDataFrameDomain(
                {**input_domain.schema, column: SparkStringColumnDescriptor()}
            ),
            output_metric=IfGroupedBy(column, SymmetricDifference()),
        )
        self._column = column

    @property
    def column(self) -> str:
        """Returns name of ID column to add."""
        return self._column

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
        """Returns DataFrame with ID column added."""
        # pylint: disable=no-member
        shuffled_partitions = Window.partitionBy(*sdf.columns).orderBy(sdf.columns[0])
        rank_column = get_nonconflicting_string(sdf.columns)

        sdf = sdf.withColumn(rank_column, sf.row_number().over(shuffled_partitions))

        return sdf.withColumn(
            self.column, sf.hex(sf.to_json(sf.array(*sdf.columns)).cast("string"))
        ).drop(rank_column)
