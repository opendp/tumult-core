# pylint: disable=line-too-long
"""Measurements on Spark DataFrames.

See `the architecture guide <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
for more information.
"""
# pylint: enable=line-too-long

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import uuid
from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union, cast

import sympy as sp
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType
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
from tmlt.core.exceptions import (
    DomainColumnError,
    DomainMismatchError,
    UnsupportedDomainError,
    UnsupportedMetricError,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import AddGeometricNoise
from tmlt.core.measurements.pandas_measurements.dataframe import Aggregate
from tmlt.core.measurements.pandas_measurements.series import AddNoiseToSeries
from tmlt.core.measures import ApproxDP, PureDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    OnColumn,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.distributions import double_sided_geometric_cmf_exact
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.join import join
from tmlt.core.utils.misc import get_materialized_df, get_nonconflicting_string
from tmlt.core.utils.validation import validate_exact_number


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
            >>> from pyspark.sql import SparkSession
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
        >>> print_sdf(noisy_spark_dataframe)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            A   B   count
        0  a1  b1 ...
        1  a1  b2 ...
        2  a2  b1 ...
        3  a2  b2 ...

    Measurement Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output type - Spark DataFrame
        * Input metric - :class:`~.OnColumn` with metric
          ``SumOf(SymmetricDifference())`` (for :class:`~.PureDP`) or
          ``RootSumOfSquared(SymmetricDifference())`` (for :class:`~.RhoZCDP`) on each
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
                ``measure_column``.
            measure_column: Name of column to add noise to.

        Note:
            The input metric of this measurement is derived from the ``measure_column``
            and the input metric of the ``measurement`` to be applied. In particular, the
            input metric of this measurement is ``measurement.input_metric`` on the
            specified ``measure_column``.
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
        if self.measurement.noise_measurement.adds_no_noise:
            return sdf
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
                non-grouping columns in the ``input_domain``.
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
        # needed_columns can't be substituted for
        # aggregation_function.input_domain.schema because it is a set, which can change
        # the order of columns. Order is important when checking that the domains match.
        input_domain_as_spark = SparkDataFrameDomain(
            {
                column: input_domain[column]
                for column in aggregation_function.input_domain.schema
            }
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
            >>> from pyspark.sql import SparkSession
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
        >>> noisy_spark_dataframe = measurement(spark_dataframe)  # doctest: +SKIP
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
    """  # pylint: disable=line-too-long,useless-suppression

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


class SparseVectorPrefixSums(SparkMeasurement):
    r"""Find the rank of the row causing the prefix sum to exceed the threshold.

    Example:
        ..
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "grouping": ["A"] * 10 + ["B"] * 10,
            ...             "rank": list(range(10)) + list(range(-5, 5)),
            ...             "count": [1] * 10 + [2] * 4 + [1000] + [2] * 5,
            ...         }
            ...     )
            ... )
            >>> noisy_spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "grouping": ["A", "B"],
            ...             "rank": [8, -1],
            ...         }
            ...     )
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
           grouping  rank  count
        0         A     0      1
        1         A     1      1
        2         A     2      1
        3         A     3      1
        4         A     4      1
        5         A     5      1
        6         A     6      1
        7         A     7      1
        8         A     8      1
        9         A     9      1
        10        B    -5      2
        11        B    -4      2
        12        B    -3      2
        13        B    -2      2
        14        B    -1   1000
        15        B     0      2
        16        B     1      2
        17        B     2      2
        18        B     3      2
        19        B     4      2

        >>> measurement = SparseVectorPrefixSums(
        ...     input_domain=SparkDataFrameDomain(
        ...         schema={
        ...             "grouping": SparkStringColumnDescriptor(),
        ...             "rank": SparkIntegerColumnDescriptor(),
        ...             "count": SparkIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     count_column="count",
        ...     rank_column="rank",
        ...     alpha=1,
        ...     grouping_columns=["grouping"],
        ...     threshold_fraction=0.90,
        ... )
        >>> noisy_spark_dataframe = measurement(spark_dataframe)  # doctest: +SKIP
        >>> print_sdf(noisy_spark_dataframe)  # doctest: +NORMALIZE_WHITESPACE
          grouping  rank
        0        A     8
        1        B    -1

    Measurement Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output type - Spark DataFrame
        * Input metric - :class:`~.OnColumn` (with inner metric
            ``SumOf(AbsoluteDifference())``) on the ``count_column``).
        * Output measure - :class:`~.PureDP`

        >>> measurement.input_domain
        SparkDataFrameDomain(schema={'grouping': SparkStringColumnDescriptor(allow_null=False), 'rank': SparkIntegerColumnDescriptor(allow_null=False, size=64), 'count': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> measurement.input_metric
        OnColumn(column='count', metric=SumOf(inner_metric=AbsoluteDifference()))
        >>> measurement.output_measure
        PureDP()

        Privacy Guarantee:
            For :math:`d_{in} = 0`, returns :math:`0`

            For :math:`d_{in} \ge 1`, returns
            :math:`(4 / \alpha) \cdot d_{in}`

            where:

            * :math:`\alpha` is :attr:`~.alpha`

            >>> measurement.privacy_function(1)
            4
            >>> measurement.privacy_function(2)
            8
    """  # pylint: disable=line-too-long,useless-suppression

    @typechecked
    def __init__(
        self,
        input_domain: SparkDataFrameDomain,
        count_column: str,
        rank_column: str,
        alpha: ExactNumberInput,
        grouping_columns: Optional[List[str]] = None,
        threshold_fraction: float = 0.95,
    ):
        r"""Constructor.

        Args:
            input_domain: Dataframe containing bin counts.
            count_column: Column name for the column containing the counts.
            rank_column: Column name for the column defining the ranking on rows to
                compute prefix sums.
            alpha: The noise scale parameter for Geometric noise that will be added to
                each prefix sum.  Noise with scale of :math:`\alpha / 2` will be added
                when computing the threshold.
                See :class:`~.AddGeometricNoise` for more information.
            grouping_columns: Optional list of column names defining the groups. The
                output dataframe will contain one row per group. If None, the entire
                input dataframe is treated as a single group.
            threshold_fraction: The fraction of the total count to use as the threshold.
                This value should be between (0, 1]. By default it is set to 0.95.
        """
        if grouping_columns is None:
            grouping_columns = []

        if count_column not in input_domain.schema:
            raise DomainColumnError(
                input_domain,
                count_column,
                f"Column '{count_column}' is not in the input schema.",
            )
        if rank_column not in input_domain.schema:
            raise DomainColumnError(
                input_domain,
                rank_column,
                f"Column '{rank_column}' is not in the input schema.",
            )
        for column in grouping_columns:
            if column not in input_domain.schema:
                raise DomainColumnError(
                    input_domain,
                    column,
                    f"Column '{column}' is not in the input schema.",
                )
            if column in (count_column, rank_column):
                raise ValueError(
                    "Grouping columns cannot contain the count or rank columns."
                )

        self._count_column = count_column
        self._rank_column = rank_column
        self.grouping_columns = grouping_columns

        try:
            validate_exact_number(
                value=alpha,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid noise scale: {e}") from e
        if not 0 < threshold_fraction <= 1:
            raise ValueError(
                f"Invalid threshold fraction: {threshold_fraction}. Must be in (0, 1]."
            )
        self._alpha = ExactNumber(alpha)
        self._threshold_fraction = threshold_fraction
        super().__init__(
            input_domain=input_domain,
            input_metric=OnColumn(count_column, SumOf(AbsoluteDifference())),
            output_measure=PureDP(),
            is_interactive=False,
        )

    @property
    def alpha(self) -> ExactNumber:
        """Returns the alpha."""
        return self._alpha

    @property
    def threshold_fraction(self) -> float:
        """Returns the threshold."""
        return self._threshold_fraction

    @property
    def count_column(self) -> str:
        """Returns the count column."""
        return self._count_column

    @property
    def rank_column(self) -> str:
        """Returns the rank column."""
        return self._rank_column

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        See `the architecture guide
        <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_ for more
        information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        d_in = ExactNumber(d_in)
        if d_in == 0:
            return ExactNumber(0)
        if self.alpha == 0:
            return ExactNumber(float("inf"))
        if d_in < 1:
            raise NotImplementedError()
        return (4 / self.alpha) * d_in

    def call(self, val: DataFrame) -> DataFrame:
        """Return row causing prefix sum to exceed the threshold."""
        df = val
        threshold_column = get_nonconflicting_string(df.columns)

        window_spec = Window.partitionBy(*self.grouping_columns).orderBy(
            self.rank_column
        )
        df = df.withColumn(
            self.count_column, sf.sum(self.count_column).over(window_spec)
        )

        add_threshold_noise = sf.pandas_udf(
            AddNoiseToSeries(AddGeometricNoise(self.alpha / 2)),
            IntegerType(),
            sf.PandasUDFType.SCALAR,
        ).asNondeterministic()
        thresholds = (
            df.groupBy(*self.grouping_columns)
            .agg(sf.max(self.count_column).alias("total_count"))
            .withColumn(
                threshold_column,
                add_threshold_noise(
                    (sf.col("total_count") * sf.lit(self.threshold_fraction)).cast(
                        "int"
                    )
                ),
            )
            .drop("total_count")
        )

        add_bin_noise = sf.pandas_udf(
            AddNoiseToSeries(AddGeometricNoise(self.alpha)),
            IntegerType(),
            sf.PandasUDFType.SCALAR,
        ).asNondeterministic()
        df = df.withColumn(self.count_column, add_bin_noise(sf.col(self.count_column)))

        if len(self.grouping_columns) == 0:
            df = df.crossJoin(thresholds)
        else:
            df = join(df, thresholds, self.grouping_columns, nulls_are_equal=True)

        max_rank_column = get_nonconflicting_string(df.columns)
        df = df.withColumn(
            max_rank_column,
            sf.max(self.rank_column).over(Window.partitionBy(*self.grouping_columns)),
        )

        row_number = get_nonconflicting_string(df.columns)
        df = (
            df.filter(
                (sf.col(self.count_column) >= sf.col(threshold_column))
                | (sf.col(max_rank_column) == sf.col(self.rank_column)),
            )
            .withColumn(
                row_number,
                sf.row_number().over(window_spec),
            )
            .filter(sf.col(row_number) == 1)
        )

        return df.select(self.grouping_columns + [self.rank_column])


def _get_sanitized_df(sdf: DataFrame) -> DataFrame:
    """Returns a randomly repartitioned and materialized DataFrame.

    See :ref:`pseudo-side-channel-mitigations` for more details on the specific
    mitigations we apply here.
    """
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
