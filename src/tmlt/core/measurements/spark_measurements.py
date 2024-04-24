# pylint: disable=line-too-long
"""Measurements on Spark DataFrames.

See `the architecture guide <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
for more information.
"""
# pylint: enable=line-too-long

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import uuid
from abc import abstractmethod
from itertools import accumulate
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import sympy as sp
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame, SparkSession  # pylint: disable=unused-import
from pyspark.sql import functions as sf
from typeguard import typechecked

# cleanup is imported just so its cleanup function runs at exit
import tmlt.core.utils.cleanup  # pylint: disable=unused-import
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    convert_pandas_domain,
)
from tmlt.core.exceptions import (
    DomainMismatchError,
    UnsupportedDomainError,
    UnsupportedMetricError,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import AddGeometricNoise
from tmlt.core.measurements.pandas_measurements.dataframe import Aggregate
from tmlt.core.measurements.pandas_measurements.series import AddNoiseToSeries
from tmlt.core.measures import ApproxDP, PureDP
from tmlt.core.metrics import OnColumn, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.utils.distributions import double_sided_geometric_cmf_exact
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.misc import get_materialized_df, get_nonconflicting_string
from tmlt.core.utils.validation import validate_exact_number

# pylint: disable=no-member


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
            raise DomainMismatchError(
                (measure_column_domain, measurement.input_domain.element_domain),
                (
                    f"{measure_column} has domain {measure_column_domain}, which is"
                    " incompatible with measurement's input domain"
                    f" {measurement.input_domain.element_domain}"
                ),
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

        See `the architecture guide <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If the :meth:`~.Measurement.privacy_function` of the
                :class:`~.AddNoiseToSeries` measurement raises :class:`NotImplementedError`.
        """
        self.input_metric.validate(d_in)
        return self.measurement.privacy_function(d_in)

    def call(self, val: DataFrame) -> DataFrame:
        """Applies measurement to measure column."""
        # TODO(#2107): Fix typing once pd.Series is a usable type
        sdf = val
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
            raise UnsupportedMetricError(
                input_metric,
                (
                    "Input metric must be SumOf(SymmetricDifference()) or"
                    " RootSumOfSquared(SymmetricDifference())"
                ),
            )

        # Check that the input domain is compatible with the aggregation
        # function's input domain.
        available_columns = set(input_domain.schema) - set(input_domain.groupby_columns)
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

        aggregation_function_domain = SparkDataFrameDomain(
            convert_pandas_domain(aggregation_function.input_domain)
        )
        input_domain_as_spark = SparkDataFrameDomain(
            {column: input_domain[column] for column in needed_columns}
        )
        if aggregation_function_domain != input_domain_as_spark:
            raise DomainMismatchError(
                (aggregation_function_domain, input_domain_as_spark),
                (
                    "The input domain is not compatible with the input domain of the "
                    "aggregation function."
                ),
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

    # pylint: disable=line-too-long
    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        See `the architecture guide <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If self.aggregation_function.privacy_function(d_in)
                raises :class:`NotImplementedError`.
        """
        # pylint: enable=line-too-long
        return self.aggregation_function.privacy_function(d_in)

    def call(self, val: GroupedDataFrame) -> DataFrame:
        """Returns DataFrame obtained by applying pandas aggregation to each group."""
        grouped_dataframe = val
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
            raise UnsupportedDomainError(
                input_domain, "Input domain cannot contain any float columns."
            )
        try:
            validate_exact_number(
                value=alpha,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid alpha: {e}") from e
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

    # pylint: disable=line-too-long
    @typechecked
    def privacy_function(
        self, d_in: ExactNumberInput
    ) -> Tuple[ExactNumber, ExactNumber]:
        """Returns the smallest d_out satisfied by the measurement.

        See `the architecture guide <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        # pylint: enable=line-too-long
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

    def call(self, val: DataFrame) -> DataFrame:
        """Return the noisy counts for common rows."""
        sdf = val
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


class BoundSelection(Measurement):
    r"""Discovers a noisy bound based on a DataFrame Column.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1"] * 100,
            ...             "B": [10] * 10 + [20] * 20 + [30] * 30 + [40] * 40,
            ...         }
            ...     )
            ... )
            >>> min_bound, max_bound = -64, 64

        >>> # Example input
        >>> print_sdf(spark_dataframe)
             A   B
        0   a1  10
        1   a1  10
        2   a1  10
        3   a1  10
        4   a1  10
        ..  ..  ..
        95  a1  40
        96  a1  40
        97  a1  40
        98  a1  40
        99  a1  40
        <BLANKLINE>
        [100 rows x 2 columns]
        >>> measurement = BoundSelection(
        ...     input_domain=SparkDataFrameDomain(
        ...         schema={
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     bound_column="B",
        ...     alpha=1,
        ...     threshold=0.95,
        ... )
        >>> min_bound, max_bound = measurement(spark_dataframe) # doctest: +SKIP
        >>> print(f"Min: {min_bound}, Max: {max_bound}")  # doctest: +NORMALIZE_WHITESPACE
        Min: -64, Max: 64

    Measurement Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output type - Tuple[int, int] if the :attr:`~.bound_column` is int, else Tuple[float, float]
        * Input metric - :class:`~.SymmetricDifference`
        * Output measure - :class:`~.PureDP`

        >>> measurement.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> measurement.input_metric
        SymmetricDifference()
        >>> measurement.output_measure
        PureDP()

        Privacy Guarantee:
            For :math:`d_{in} = 0`, returns :math:`0`

            For :math:`d_{in} > 0`, returns :math:`(4/\alpha) * d_{in}`

            where:

            * :math:`\alpha` is :attr:`~.alpha`

            >>> measurement.privacy_function(1)
            4
            >>> measurement.privacy_function(2)
            8
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        bound_column: str,
        alpha: ExactNumberInput,
        threshold: float = 0.95,
    ):
        r"""Constructor.

        Args:
            input_domain: Domain of the input Spark DataFrames. Input must either be
                a column of floating point numbers or a column of integers.
            bound_column: Column name for finding the bounds.
                The column values must be between [-2^64 + 1, 2^64 - 1] for
                64 bit integers, [-2^32 + 1, 2^32 - 1] for 32 bit integers
                and between [-2^100, 2^100] for floats.
            alpha: The noise scale parameter for Geometric noise that will be added
                to true number of values falling between the tested bounds on each
                round of the algorithm.
                Noise with scale of :math:`\alpha / 2` will be added
                to compute the threshold.
                See :class:`~.AddGeometricNoise` for more information.
            threshold: The fraction of the total count to use as the threshold.
                This value should be between (0, 1]. By default it is set to 0.95.
        """
        if bound_column not in input_domain.schema:
            raise ValueError(
                f"Invalid bounding column name: ({bound_column}) not in schema"
            )
        column_type = input_domain.schema[bound_column]
        if isinstance(column_type, SparkFloatColumnDescriptor):
            splits = (
                [float("-inf")]
                + [-(2**i) * 2**-100 for i in range(200, -1, -1)]
                + [0]
                + [2**i * 2**-100 for i in range(201)]
                + [float("inf")]
            )
        elif isinstance(column_type, SparkIntegerColumnDescriptor):
            splits = (
                [-(2 ** (column_type.size - 1)) + 1]
                + [-(2**i) for i in range(column_type.size - 2, -1, -1)]
                + [0]
                + [2**i for i in range(column_type.size)]
            )
        else:
            raise ValueError(
                "Invalid column type, expected [SparkFloatColumnDescriptor,"
                f" SparkIntegerColumnDescriptor], got {column_type}"
            )
        self._splits = splits
        self._bound_column_type = column_type

        try:
            validate_exact_number(
                value=alpha,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid noise scale: {e}") from e
        if not 0 < threshold <= 1:
            raise ValueError(f"Invalid threshold: {threshold}. Must be in (0, 1]")
        self._bound_column = bound_column
        self._alpha = ExactNumber(alpha)
        self._threshold = threshold
        super().__init__(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            is_interactive=False,
        )

    @property
    def bound_column(self) -> str:
        """Returns the column to compute the bounds for."""
        return self._bound_column

    @property
    def bound_column_type(self) -> SparkColumnDescriptor:
        """Returns the type of the bound column."""
        return self._bound_column_type

    @property
    def splits(self) -> Union[List[float], List[int]]:
        """Returns the splits."""
        return self._splits.copy()

    @property
    def alpha(self) -> ExactNumber:
        """Returns the alpha."""
        return self._alpha

    @property
    def threshold(self) -> float:
        """Returns the threshold."""
        return self._threshold

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        See `the architecture guide <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        d_in = ExactNumber(d_in)
        if d_in == 0:
            return ExactNumber(0)
        if self._alpha == 0:
            return ExactNumber(float("inf"))
        if d_in < 1:
            raise NotImplementedError()
        return (4 / self._alpha) * d_in

    def __call__(self, sdf: DataFrame) -> Union[Tuple[float, float], Tuple[int, int]]:
        """Returns the bounds for the given column."""
        bin_col = get_nonconflicting_string(sdf.columns)
        bin_count_col = get_nonconflicting_string(sdf.columns + [bin_col])

        # Split the data into bins
        bucketizer = Bucketizer(
            splits=self._splits, inputCol=self._bound_column, outputCol=bin_col
        )
        binned: pd.DataFrame = (
            bucketizer.setHandleInvalid("keep")
            .transform(sdf)
            .groupBy(bin_col)
            .agg(sf.count(bin_col).alias(bin_count_col))
            .toPandas()
        )

        non_zero_bins = binned[bin_col].values
        # Bucketizer returns only non-zero bins, so we need to fill in the rest
        binned_counts = []
        for i in range(len(self._splits) - 1):
            if float(i) not in non_zero_bins:
                binned_counts.append(0)
            else:
                binned_counts.append(
                    binned.loc[binned[bin_col] == float(i)][bin_count_col].values[0]
                )

        center = len(binned_counts) // 2
        negative_bins = binned_counts[:center]
        negative_bins.reverse()
        positive_bins = binned_counts[center:]
        counts_by_bin = [neg + pos for neg, pos in zip(negative_bins, positive_bins)]
        if isinstance(self._bound_column_type, SparkFloatColumnDescriptor):
            # Take all except for last value to filter out the infinite bin.
            counts_by_bin = counts_by_bin[:-1]

        add_threshold_noise = AddGeometricNoise(self._alpha / 2)
        noisy_threshold = add_threshold_noise(np.int64(0))
        true_count = self._threshold * sum(counts_by_bin)

        cumulative_counts_by_bin = list(accumulate(counts_by_bin))

        # Finding the bin that contains enough counts above the threshold
        add_bin_noise = AddGeometricNoise(self._alpha)
        bounds = self._splits[len(self._splits) // 2 :]
        if isinstance(self._bound_column_type, SparkFloatColumnDescriptor):
            # Take all except for last value to filter out the infinite bound.
            bounds = bounds[:-1]

        for step, bin_count in enumerate(cumulative_counts_by_bin):
            noisy_count = add_bin_noise(np.int64(bin_count - true_count))
            upper_bound = bounds[step + 1]
            if noisy_count >= noisy_threshold:
                break
        return -upper_bound, upper_bound


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
    return get_materialized_df(
        sdf.withColumn(partitioning_column, sf.rand())
        .repartition(partitioning_column)
        .drop(partitioning_column)
        .sortWithinPartitions(*sdf.columns),
        table_name=f"table_{uuid.uuid4().hex}",
    )
