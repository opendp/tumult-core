"""Measurements on Spark DataFrames."""
# TODO(#1320): Add link to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import uuid
from abc import abstractmethod
from threading import Lock
from typing import Any, Optional, Tuple, Union, cast

import sympy as sp
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from typeguard import typechecked

# cleanup is imported just so its cleanup function runs at exit
import tmlt.core.utils.cleanup  # pylint: disable=unused-import
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    convert_pandas_domain,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import AddGeometricNoise
from tmlt.core.measurements.pandas_measurements.dataframe import Aggregate
from tmlt.core.measurements.pandas_measurements.series import AddNoiseToSeries
from tmlt.core.measures import ApproxDP
from tmlt.core.metrics import OnColumn, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.utils.configuration import Config
from tmlt.core.utils.distributions import double_sided_geometric_cmf_exact
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.misc import get_nonconflicting_string
from tmlt.core.utils.validation import validate_exact_number

# pylint: disable=no-member

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
        # TODO(#2107): Fix typing once pd.Series is a usable type
        udf = sf.pandas_udf(  # type: ignore
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


class GeometricPartitionSelection(SparkMeasurement):
    r"""Discovers the distinct rows in a DataFrame, suppressing infrequent rows.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1"] + ["a2"] * 100,
            ...             "B": ["b1"] + ["b2"] * 100,
            ...         }
            ...     )
            ... )
            >>> noisy_spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a2"],
            ...             "B": ["b2"],
            ...             "count": [106],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
              A   B
        0    a1  b1
        1    a2  b2
        2    a2  b2
        3    a2  b2
        4    a2  b2
        ..   ..  ..
        96   a2  b2
        97   a2  b2
        98   a2  b2
        99   a2  b2
        100  a2  b2
        <BLANKLINE>
        [101 rows x 2 columns]
        >>> measurement = GeometricPartitionSelection(
        ...     input_domain=SparkDataFrameDomain(
        ...         schema={
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         },
        ...     ),
        ...     threshold=50,
        ...     alpha=1,
        ... )
        >>> noisy_spark_dataframe = measurement(spark_dataframe) # doctest: +SKIP
        >>> print_sdf(noisy_spark_dataframe)  # doctest: +NORMALIZE_WHITESPACE
            A   B  count
        0  a2  b2    106

    Measurement Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output type - Spark DataFrame
        * Input metric - :class:`~.SymmetricDifference`
        * Output measure - :class:`~.ApproxDP`

        >>> measurement.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> measurement.input_metric
        SymmetricDifference()
        >>> measurement.output_measure
        ApproxDP()

        Privacy Guarantee:
            For :math:`d_{in} = 0`, returns :math:`(0, 0)`

            For :math:`d_{in} = 1`, returns
            :math:`(1/\alpha, 1 - CDF_{\alpha}[\tau - 2])`

            For :math:`d_{in} > 1`, returns
            :math:`(d_{in} \cdot \epsilon, d_{in} \cdot e^{d_{in} \cdot \epsilon} \cdot \delta)`

            where:

            * :math:`\alpha` is :attr:`~.alpha`
            * :math:`\tau` is :attr:`~.threshold`
            * :math:`\epsilon` is the first element returned for the :math:`d_{in} = 1`
              case
            * :math:`\delta` is the second element returned for the :math:`d_{in} = 1`
              case
            * :math:`CDF_{\alpha}` is :func:`~.double_sided_geometric_cmf_exact`

            >>> epsilon, delta = measurement.privacy_function(1)
            >>> epsilon
            1
            >>> delta.to_float(round_up=True)
            3.8328565409781243e-22
            >>> epsilon, delta = measurement.privacy_function(2)
            >>> epsilon
            2
            >>> delta.to_float(round_up=True)
            5.664238400088129e-21
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        threshold: int,
        alpha: ExactNumberInput,
        count_column: Optional[str] = None,
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames. Input cannot contain
                floating point columns.
            threshold: The minimum threshold for the noisy count to have to be released.
                Can be nonpositive, but must be integral.
            alpha: The noise scale parameter for Geometric noise. See
                :class:`~.AddGeometricNoise` for more information.
            count_column: Column name for output group counts. If None, output column
                will be named "count".
        """
        if any(
            isinstance(column_descriptor, SparkFloatColumnDescriptor)
            for column_descriptor in input_domain.schema.values()
        ):
            raise ValueError("Input domain cannot contain any float columns.")
        try:
            validate_exact_number(
                value=alpha,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid alpha: {e}")
        if count_column is None:
            count_column = "count"
        if count_column in set(input_domain.schema):
            raise ValueError(
                f"Invalid count column name: ({count_column}) column already exists"
            )
        self._alpha = ExactNumber(alpha)
        self._threshold = threshold
        self._count_column = count_column
        super().__init__(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            output_measure=ApproxDP(),
            is_interactive=False,
        )

    @property
    def alpha(self) -> ExactNumber:
        """Returns the noise scale."""
        return self._alpha

    @property
    def threshold(self) -> int:
        """Returns the minimum noisy count to include row."""
        return self._threshold

    @property
    def count_column(self) -> str:
        """Returns the count column name."""
        return self._count_column

    @typechecked
    def privacy_function(
        self, d_in: ExactNumberInput
    ) -> Tuple[ExactNumber, ExactNumber]:
        """Returns the smallest d_out satisfied by the measurement.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        d_in = ExactNumber(d_in)
        if d_in == 0:
            return ExactNumber(0), ExactNumber(0)
        if self.alpha == 0:
            return ExactNumber(float("inf")), ExactNumber(0)
        if d_in < 1:
            raise NotImplementedError()
        base_epsilon = 1 / self.alpha
        base_delta = 1 - double_sided_geometric_cmf_exact(
            self.threshold - 2, self.alpha
        )
        if d_in == 1:
            return base_epsilon, base_delta
        return (
            d_in * base_epsilon,
            min(
                ExactNumber(1),
                d_in * ExactNumber(sp.E) ** (d_in * base_epsilon) * base_delta,
            ),
        )

    def call(self, sdf: DataFrame) -> DataFrame:
        """Return the noisy counts for common rows."""
        count_df = sdf.groupBy(sdf.columns).agg(sf.count("*").alias(self.count_column))
        internal_measurement = AddNoiseToColumn(
            input_domain=SparkDataFrameDomain(
                schema={
                    **cast(SparkDataFrameDomain, self.input_domain).schema,
                    self.count_column: SparkIntegerColumnDescriptor(),
                }
            ),
            measurement=AddNoiseToSeries(AddGeometricNoise(self.alpha)),
            measure_column=self.count_column,
        )
        noisy_count_df = internal_measurement(count_df)
        return noisy_count_df.filter(sf.col(self.count_column) >= self.threshold)


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
