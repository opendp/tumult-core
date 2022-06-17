"""Measurements on Spark DataFrames."""
# TODO(#1320): Add link to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0

import uuid
from abc import abstractmethod
from threading import Lock
from typing import Any, Union, cast

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from typeguard import typechecked

# cleanup is imported just so its cleanup function runs at exit
import tmlt.core.utils.cleanup  # pylint: disable=unused-import
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkGroupedDataFrameDomain,
    SparkStringColumnDescriptor,
    convert_pandas_domain,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.pandas_measurements.dataframe import Aggregate
from tmlt.core.measurements.pandas_measurements.series import AddNoiseToSeries
from tmlt.core.metrics import OnColumn, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.utils.configuration import Config
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.misc import get_nonconflicting_string

_materialization_lock = Lock()


class SparkMeasurement(Measurement):
    """Base class that materializes output DataFrames before returning."""

    @abstractmethod
    def call(self, val: Any) -> DataFrame:
        """Performs measurement.

        Warning:
            Spark recomputes the output of this method (adding different noise
            each time) on every call to collect.
        """

    def __call__(self, val: Any) -> DataFrame:
        """Performs measurement and returns a DataFrame with additional protections.

        See :ref:`pseudo-side-channel-mitigations` for more details on the specific
        mitigations we apply here.
        """
        return _get_sanitized_df(self.call(val))


class AddNoiseToColumn(SparkMeasurement):
    """Adds noise to a single aggregated column of a Spark DataFrame.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.measurements.noise_mechanisms import (
            ...     AddLaplaceNoise,
            ... )
            >>> from tmlt.core.measurements.pandas_measurements.series import (
            ...     AddNoiseToSeries,
            ... )
            >>> from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
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
            ...             "A": ["a1", "a1", "a2", "a2"],
            ...             "B": ["b1", "b2", "b1", "b2"],
            ...             "count": [3, 2, 1, 0],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B  count
        0  a1  b1      3
        1  a1  b2      2
        2  a2  b1      1
        3  a2  b2      0
        >>> # Create a measurement that can add noise to a pd.Series
        >>> add_laplace_noise = AddLaplaceNoise(
        ...     scale="0.5",
        ...     input_domain=NumpyIntegerDomain(),
        ... )
        >>> # Create a measurement that can add noise to a Spark DataFrame
        >>> add_laplace_noise_to_column = AddNoiseToColumn(
        ...     input_domain=SparkDataFrameDomain(
        ...         schema={
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...             "count": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     measurement=AddNoiseToSeries(add_laplace_noise),
        ...     measure_column="count",
        ... )
        >>> # Apply measurement to data
        >>> noisy_spark_dataframe = add_laplace_noise_to_column(spark_dataframe)
        >>> print_sdf(noisy_spark_dataframe) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            A   B   count
        0  a1  b1 ...
        1  a1  b2 ...
        2  a2  b1 ...
        3  a2  b2 ...

    Measurement Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output type - Spark DataFrame
        * Input metric - :class:`~.OnColumn` with metric
          `SumOf(SymmetricDifference())` (for :class:`~.PureDP`) or
          `RootSumOfSquared(SymmetricDifference())` (for :class:`~.RhoZCDP`) on each
          column.
        * Output measure - :class:`~.PureDP` or :class:`~.RhoZCDP`

        >>> add_laplace_noise_to_column.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'count': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> add_laplace_noise_to_column.input_metric
        OnColumn(column='count', metric=SumOf(inner_metric=AbsoluteDifference()))
        >>> add_laplace_noise_to_column.output_measure
        PureDP()

        Privacy Guarantee:
            :class:`~.AddNoiseToColumn`'s :meth:`~.privacy_function` returns the output of
            privacy function on the :class:`~.AddNoiseToSeries` measurement.

            >>> add_laplace_noise_to_column.privacy_function(1)
            2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        measurement: AddNoiseToSeries,
        measure_column: str,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input spark DataFrames.
            measurement: :class:`~.AddNoiseToSeries` measurement for adding noise to
                `measure_column`.
            measure_column: Name of column to add noise to.

        Note:
            The input metric of this measurement is derived from the `measure_column`
            and the input metric of the `measurement` to be applied. In particular, the
            input metric of this measurement is `measurement.input_metric` on the
            specified `measure_column`.
        """
        measure_column_domain = input_domain[measure_column].to_numpy_domain()
        if measure_column_domain != measurement.input_domain.element_domain:
            raise ValueError(
                f"{measure_column} has domain {measure_column_domain}, which is"
                " incompatible with measurement's input domain"
                f" {measurement.input_domain.element_domain}"
            )
        assert isinstance(measurement.input_metric, (SumOf, RootSumOfSquared))
        super().__init__(
            input_domain=input_domain,
            input_metric=OnColumn(measure_column, measurement.input_metric),
            output_measure=measurement.output_measure,
            is_interactive=False,
        )
        self._measure_column = measure_column
        self._measurement = measurement

    @property
    def measure_column(self) -> str:
        """Returns the name of the column to add noise to."""
        return self._measure_column

    @property
    def measurement(self) -> AddNoiseToSeries:
        """Returns the :class:`~.AddNoiseToSeries` measurement to apply to measure column."""
        return self._measurement

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If the :meth:`~.Measurement.privacy_function` of the
                :class:`~.AddNoiseToSeries` measurement raises :class:`NotImplementedError`.
        """
        self.input_metric.validate(d_in)
        return self.measurement.privacy_function(d_in)

    def call(self, sdf: DataFrame) -> DataFrame:
        """Applies measurement to measure column."""
        udf = sf.pandas_udf(
            self.measurement, self.measurement.output_type, sf.PandasUDFType.SCALAR
        ).asNondeterministic()
        sdf = sdf.withColumn(self.measure_column, udf(sdf[self.measure_column]))
        return sdf


class ApplyInPandas(SparkMeasurement):
    """Applies a pandas dataframe aggregation to each group in a GroupedDataFrame."""

    @typechecked
    def __init__(
        self,
        input_domain: SparkGroupedDataFrameDomain,
        input_metric: Union[SumOf, RootSumOfSquared],
        aggregation_function: Aggregate,
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input GroupedDataFrames.
            input_metric: Distance metric on inputs. It must one of
                :class:`~.SumOf` or :class:`~.RootSumOfSquared` with
                inner metric :class:`~.SymmetricDifference`.
            aggregation_function: An Aggregation measurement to be applied to each
                group. The input domain of this measurement must be a
                :class:`~.PandasDataFrameDomain` corresponding to a subset of the
                non-grouping columns in the `input_domain`.
        """
        if input_metric.inner_metric != SymmetricDifference():
            raise ValueError(
                "Input metric must be SumOf(SymmetricDifference()) or"
                " RootSumOfSquared(SymmetricDifference())"
            )

        # Check that the input domain is compatible with the aggregation
        # function's input domain.
        available_columns = set(input_domain.schema) - set(
            input_domain.group_keys.columns
        )
        needed_columns = set(aggregation_function.input_domain.schema)
        if not needed_columns <= available_columns:
            raise ValueError(
                "The aggregation function needs unexpected columns: "
                f"{sorted(needed_columns - available_columns)}"
            )
        for column in needed_columns:
            if input_domain[column].allow_null and not isinstance(
                input_domain[column], SparkStringColumnDescriptor
            ):
                raise ValueError(
                    f"Column ({column}) in the input domain is a"
                    " numeric nullable column, which is not supported by ApplyInPandas"
                )

        if SparkDataFrameDomain(
            convert_pandas_domain(aggregation_function.input_domain)
        ) != SparkDataFrameDomain(
            {column: input_domain[column] for column in needed_columns}
        ):
            raise ValueError(
                "The input domain is not compatible with the input domain of the "
                "aggregation function."
            )

        self._aggregation_function = aggregation_function

        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=aggregation_function.output_measure,
            is_interactive=False,
        )

    @property
    def aggregation_function(self) -> Aggregate:
        """Returns the aggregation function."""
        return self._aggregation_function

    @property
    def input_domain(self) -> SparkGroupedDataFrameDomain:
        """Returns input domain."""
        return cast(SparkGroupedDataFrameDomain, super().input_domain)

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If self.aggregation_function.privacy_function(d_in)
                raises :class:`NotImplementedError`.
        """
        return self.aggregation_function.privacy_function(d_in)

    def call(self, grouped_dataframe: GroupedDataFrame) -> DataFrame:
        """Returns DataFrame obtained by applying pandas aggregation to each group."""
        return grouped_dataframe.select(
            grouped_dataframe.groupby_columns
            + list(self.aggregation_function.input_domain.schema)
        ).apply_in_pandas(
            aggregation_function=self.aggregation_function,
            aggregation_output_schema=self.aggregation_function.output_schema,
        )


def _get_sanitized_df(sdf: DataFrame) -> DataFrame:
    """Returns a randomly repartitioned and materialized DataFrame.

    See :ref:`pseudo-side-channel-mitigations` for more details on the specific
    mitigations we apply here.
    """
    # pylint: disable=no-name-in-module
    partitioning_column = get_nonconflicting_string(sdf.columns)
    # repartitioning by a column of random numbers ensures that the content
    # of partitions of the output DataFrame is determined randomly.
    # for each row, its partition number (the partition index that the row is
    # distributed to) is determined as: `hash(partitioning_column) % num_partitions`
    return _get_materialized_df(
        sdf.withColumn(partitioning_column, sf.rand())
        .repartition(partitioning_column)
        .drop(partitioning_column)
        .sortWithinPartitions(*sdf.columns),
        table_name=f"table_{uuid.uuid4().hex}",
    )


def _get_materialized_df(sdf: DataFrame, table_name: str) -> DataFrame:
    """Returns a new DataFrame constructed after materializing.

    Args:
        sdf: DataFrame to be materialized.
        table_name: Name to be used to refer to the table.
            If a table with `table_name` already exists, an error is raised.
    """
    col_names = sdf.columns
    # The following is necessary because saving in parquet format requires that column
    # names do not contain any of these characters in " ,;{}()\\n\\t=".
    sdf = sdf.toDF(*[str(i) for i in range(len(col_names))])
    with _materialization_lock:
        spark = SparkSession.builder.getOrCreate()
        last_database = spark.catalog.currentDatabase()
        spark.sql(f"CREATE DATABASE IF NOT EXISTS `{Config.temp_db_name()}`;")
        spark.catalog.setCurrentDatabase(Config.temp_db_name())
        sdf.write.saveAsTable(table_name)
        materialized_df = spark.read.table(table_name).toDF(*col_names)
        spark.catalog.setCurrentDatabase(last_database)
        return materialized_df
