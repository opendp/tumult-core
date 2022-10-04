"""Module containing metrics used for constructing measurements and transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-member

from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
from typing import Any, Dict, Iterable, List, Tuple, Union, cast

import numpy as np  # pylint: disable=unused-import
import pandas as pd
import sympy as sp
from pyspark.sql import functions as sf
from pyspark.sql.session import SparkSession
from typeguard import typechecked

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.validation import validate_exact_number


class Metric(ABC):
    """Base class for input/output metrics."""

    @abstractmethod
    def validate(self, value: Any):
        """Raises an error if `value` not a valid distance.

        Args:
            value: A distance between two datasets under this metric.
        """
        ...

    @abstractmethod
    def compare(self, value1: Any, value2: Any) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        ...

    @abstractmethod
    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain."""
        ...

    @abstractmethod
    def distance(self, value1: Any, value2: Any, domain: Domain) -> Any:
        """Returns the metric distance between two elements of a supported domain."""
        ...

    def _validate_distance_arguments(
        self, value1: Any, value2: Any, domain: Domain
    ) -> None:
        """Raise an exception if the arguments to a distance method aren't valid."""
        if not self.supports_domain(domain):
            raise ValueError(f"{repr(self)} does not support domain {repr(domain)}.")
        if value1 not in domain:
            raise OutOfDomainError(f"The first argument is not in domain {domain}.")
        if value2 not in domain:
            raise OutOfDomainError(f"The second argument is not in domain {domain}.")

    def __eq__(self, other: Any) -> bool:
        """Return True if both metrics are equal."""
        return repr(self) == repr(other)

    def __hash__(self):
        """Returns hash value."""
        return hash(repr(self))

    def __repr__(self) -> str:
        """Returns string representation."""
        return f"{self.__class__.__name__}()"


class NullMetric(Metric):
    """Metric for use when distance is undefined."""

    def validate(self, value: Any):
        """Raises an error if `value` not a valid distance.

        This method is not implemented.

        Args:
            value: A distance between two datasets under this metric.
        """
        raise NotImplementedError()

    def compare(self, value1: Any, value2: Any) -> bool:
        """Returns True if `value1` is less than or equal to `value2`.

        This method is not implemented.

        Args:
            value1: A distance between two datasets under this metric.
            value2: A distance between two datasets under this metric.
        """
        raise NotImplementedError()

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        return True

    def distance(self, value1: Any, value2: Any, domain: Domain) -> Any:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        raise NotImplementedError()


class ExactNumberMetric(Metric):
    """A metric whose distances are exact numbers."""

    @abstractmethod
    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        ...


class AbsoluteDifference(ExactNumberMetric):
    """The absolute value of the difference of two values.

    Example:
        >>> AbsoluteDifference().distance(
        ...     np.int64(20), np.int64(82), NumpyIntegerDomain()
        ... )
        62
        >>> # 1.2 is first converted to rational 5404319552844595/4503599627370496
        >>> AbsoluteDifference().distance(
        ...     np.float64(1.2), np.float64(1.0), NumpyFloatDomain()
        ... )
        900719925474099/4503599627370496
    """

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a nonnegative real or infinite

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            validate_exact_number(
                value=value,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid value for metric AbsoluteDifference: {e}")

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        self.validate(value1)
        self.validate(value2)
        return ExactNumber(value1) <= ExactNumber(value2)

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        if isinstance(domain, NumpyIntegerDomain):
            return True
        if isinstance(domain, NumpyFloatDomain):
            return not domain.allow_inf and not domain.allow_nan
        return False

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        if isinstance(domain, NumpyIntegerDomain):
            distance = ExactNumber(int(abs(value1 - value2)))
        else:
            assert isinstance(domain, NumpyFloatDomain)
            distance = ExactNumber(
                abs(
                    sp.Rational(*float.as_integer_ratio(value1))
                    - sp.Rational(*float.as_integer_ratio(value2))
                )
            )
        self.validate(distance)
        return distance


class SymmetricDifference(ExactNumberMetric):
    """The number of elements that are in only one of two sets.

    This metric is compatible with spark dataframes, pandas dataframes, and pandas
    series. It ignores ordering and, in the case of pandas, indices. That is, it treats
    each collection as a multiset of items. For non-grouped data, it treats each record
    as an item. For grouped data there are a few cases:

    - The distance between two groups with the same multi-set of records is 0
    - The distance between two groups where exactly one is empty is 1
    - The distance between two groups with different records (where neither is empty) is
      2

    Examples:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkColumnsDescriptor,
        ...     SparkIntegerColumnDescriptor,
        ... )
        >>> spark = SparkSession.builder.getOrCreate()
        >>> domain = SparkDataFrameDomain(
        ...     {
        ...         "A": SparkIntegerColumnDescriptor(),
        ...         "B": SparkIntegerColumnDescriptor(),
        ...     }
        ... )
        >>> df1 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 1, 1, 2, 3], "B": [2, 2, 2, 4, 3]})
        ... )
        >>> df2 = spark.createDataFrame(pd.DataFrame({"A": [1, 2, 1], "B": [2, 4, 1]}))
        >>> SymmetricDifference().distance(df1, df2, domain)
        4
        >>> group_keys = spark.createDataFrame(pd.DataFrame({"B": [1, 2, 4]}))
        >>> domain = SparkGroupedDataFrameDomain(
        ...     {
        ...         "A": SparkIntegerColumnDescriptor(),
        ...         "B": SparkIntegerColumnDescriptor(),
        ...     },
        ...     group_keys,
        ... )
        >>> grouped_df1 = GroupedDataFrame(df1, group_keys)
        >>> grouped_df2 = GroupedDataFrame(df2, group_keys)
        >>> SymmetricDifference().distance(grouped_df1, grouped_df2, domain)
        3
    """

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a nonnegative integer or infinity

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            validate_exact_number(
                value=value,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid value for metric SymmetricDifference: {e}")

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        self.validate(value1)
        self.validate(value2)
        return ExactNumber(value1) <= ExactNumber(value2)

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        return isinstance(
            domain,
            (
                SparkDataFrameDomain,
                PandasDataFrameDomain,
                PandasSeriesDomain,
                SparkGroupedDataFrameDomain,
            ),
        )

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        if isinstance(domain, SparkDataFrameDomain):
            distance = ExactNumber(
                value1.exceptAll(value2).count() + value2.exceptAll(value1).count()
            )
            self.validate(distance)
            return distance
        elif isinstance(domain, PandasDataFrameDomain):
            s1 = Counter(map(tuple, value1.values))
            s2 = Counter(map(tuple, value2.values))
            distance = ExactNumber(sum((s1 - s2).values()) + sum((s2 - s1).values()))
            self.validate(distance)
            return distance
        elif isinstance(domain, PandasSeriesDomain):
            s1 = Counter(value1)
            s2 = Counter(value2)
            distance = ExactNumber(sum((s1 - s2).values()) + sum((s2 - s1).values()))
            self.validate(distance)
            return distance
        else:
            assert isinstance(domain, SparkGroupedDataFrameDomain)
            groups1 = value1.get_groups()
            groups2 = value2.get_groups()
            group_domain = domain.get_group_domain()

            # If this fails, one of the grouped dataframes isn't part of the domain, and
            # something should have failed earlier.
            assert set(groups1.keys()) == set(groups2.keys())
            distance = ExactNumber(0)
            for key in groups1:
                df1 = groups1[key]
                df2 = groups2[key]
                if self.distance(df1, df2, group_domain) > 0:
                    if len(df1.head(1)) == 0 or len(df2.head(1)) == 0:
                        distance += 1
                    else:
                        distance += 2
            return distance


class HammingDistance(ExactNumberMetric):
    """The number of elements that are different between two sets of the same size.

    This metric is compatible with spark dataframes, pandas dataframes, and pandas
    series. It ignores ordering and, in the case of pandas, indices. That is, it treats
    each collection as a multiset of records.

    If the sets are not the same size, the distance is infinity.

    Example:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import SparkColumnsDescriptor
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkIntegerColumnDescriptor,
        ... )
        >>> spark = SparkSession.builder.getOrCreate()
        >>> domain = SparkDataFrameDomain(
        ...     {
        ...         "A": SparkIntegerColumnDescriptor(),
        ...         "B": SparkIntegerColumnDescriptor(),
        ...     }
        ... )
        >>> df1 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 1, 1, 3], "B": [2, 2, 2, 4]})
        ... )
        >>> df2 = spark.createDataFrame(pd.DataFrame({"A": [1, 2], "B": [2, 4]}))
        >>> HammingDistance().distance(df1, df2, domain)
        oo
        >>> df3 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 2, 3, 1], "B": [2, 4, 4, 2]})
        ... )
        >>> HammingDistance().distance(df1, df3, domain)
        1
    """

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a nonnegative and integer or infinity

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            validate_exact_number(
                value=value,
                allow_nonintegral=False,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid value for metric HammingDistance: {e}")

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        self.validate(value1)
        self.validate(value2)
        return ExactNumber(value1) <= ExactNumber(value2)

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        return isinstance(
            domain, (SparkDataFrameDomain, PandasDataFrameDomain, PandasSeriesDomain)
        )

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        if isinstance(domain, SparkDataFrameDomain):
            if value1.count() != value2.count():
                return sp.oo
            distance = ExactNumber(value1.exceptAll(value2).count())
            self.validate(distance)
            return distance
        elif isinstance(domain, PandasDataFrameDomain):
            s1 = Counter(map(tuple, value1.values))
            s2 = Counter(map(tuple, value2.values))
            if sum(s1.values()) != sum(s2.values()):
                return ExactNumber(sp.oo)
            distance = ExactNumber(sum((s1 - s2).values()))
            self.validate(distance)
            return distance
        else:
            assert isinstance(domain, PandasSeriesDomain)
            s1 = Counter(value1)
            s2 = Counter(value2)
            if sum(s1.values()) != sum(s2.values()):
                return ExactNumber(sp.oo)
            distance = ExactNumber(sum((s1 - s2).values()))
            self.validate(distance)
            return distance


class AggregationMetric(ExactNumberMetric):
    """Distances resulting from aggregating distances of its components.

    Components may be elements of a series, groups of a grouped dataframe, or elements
    of a list. This metric is parameterized by an `inner_metric` that is used to compute
    the distances of the components. See :class:`SumOf` or :class`RootSumOfSquared` for
    example usage.
    """

    @typechecked
    def __init__(
        self,
        inner_metric: Union[
            AbsoluteDifference, SymmetricDifference, HammingDistance, "IfGroupedBy"
        ],
    ):
        """Constructor.

        Args:
            inner_metric: Metric to be applied to the components.
        """
        self._inner_metric = inner_metric

    @property
    def inner_metric(
        self,
    ) -> Union[AbsoluteDifference, SymmetricDifference, HammingDistance, "IfGroupedBy"]:
        """Returns metric to be used for summing."""
        return self._inner_metric

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`.

        Args:
            value1: A distance between two datasets under this metric.
            value2: A distance between two datasets under this metric.
        """
        self.validate(value1)
        self.validate(value2)
        return ExactNumber(value1) <= ExactNumber(value2)

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        if not isinstance(self.inner_metric, ExactNumberMetric):
            return False

        if isinstance(domain, SparkGroupedDataFrameDomain):
            return self.inner_metric.supports_domain(domain.get_group_domain())
        if isinstance(domain, PandasSeriesDomain):
            return self.inner_metric.supports_domain(domain.element_domain)
        if isinstance(domain, ListDomain):
            return self.inner_metric.supports_domain(domain.element_domain)
        return False

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        if isinstance(domain, SparkGroupedDataFrameDomain):
            # help mypy
            assert isinstance(value1, GroupedDataFrame)
            assert isinstance(value2, GroupedDataFrame)

            groups1 = value1.get_groups()
            groups2 = value2.get_groups()

            # If this fails, one of the grouped dataframes isn't part of the domain, and
            # something should have failed earlier.
            assert set(groups1.keys()) == set(groups2.keys())
            group_domain = domain.get_group_domain()
            distance = self._aggregate(
                [
                    self.inner_metric.distance(groups1[key], groups2[key], group_domain)
                    for key in groups1.keys()
                ]
            )
        elif isinstance(domain, PandasSeriesDomain):
            # help mypy
            assert isinstance(value1, pd.Series)
            assert isinstance(value2, pd.Series)
            # mypy refused to typecheck without introducing a new variable despite the
            # fact that it knows domain is an instance of PandasSeriesDomain. This
            # could be related to its use inside a lambda.
            typed_domain = cast(PandasSeriesDomain, domain)

            if value1.size != value2.size:
                return ExactNumber(sp.oo)

            distance = self._aggregate(
                map(
                    lambda x: self.inner_metric.distance(
                        x[0], x[1], typed_domain.element_domain
                    ),
                    # Omitting to_numpy zips correctly but converts to python
                    # ints/floats instead of keeping numpy ints/floats.
                    zip(value1.to_numpy(), value2.to_numpy()),
                )
            )

        else:
            assert isinstance(domain, ListDomain)
            # help mypy
            assert isinstance(value1, list)
            assert isinstance(value2, list)
            # mypy refused to typecheck without introducing a new variable despite the
            # fact that it knows domain is a an instance of PandasSeriesDomain. This
            # could be related to its use inside a lambda.
            typed_domain2 = cast(ListDomain, domain)

            if len(value1) != len(value2):
                return ExactNumber(sp.oo)

            distance = self._aggregate(
                map(
                    lambda x: self.inner_metric.distance(
                        x[0], x[1], typed_domain2.element_domain
                    ),
                    zip(value1, value2),
                )
            )

        self.validate(distance)
        return distance

    def __repr__(self) -> str:
        """Returns string representation."""
        return f"{self.__class__.__name__}(inner_metric={self.inner_metric})"

    @abstractmethod
    def _aggregate(self, distances: Iterable[ExactNumber]) -> ExactNumber:
        """Aggregate the component distances.

        Args:
            distances: The list of distances to aggregate.
        """
        ...


class SumOf(AggregationMetric):
    """Distances resulting from summing distances of its components.

    These components may be elements of a series, groups of a grouped dataframe, or
    elements of a list. This metric is parameterized by an `inner_metric` that is used
    to compute the distances of the components.

    Example:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import SparkColumnsDescriptor
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkIntegerColumnDescriptor,
        ... )
        >>> spark = SparkSession.builder.getOrCreate()

        >>> # Symmetric difference on SparkGroupedDataFrame
        >>> group_keys = spark.createDataFrame(pd.DataFrame({"A": [1, 2]}))
        >>> domain = SparkGroupedDataFrameDomain(
        ...     {
        ...         "A": SparkIntegerColumnDescriptor(),
        ...         "B": SparkIntegerColumnDescriptor(),
        ...     },
        ...     group_keys,
        ... )
        >>> df1 = GroupedDataFrame(
        ...     spark.createDataFrame(
        ...         pd.DataFrame({"A": [1, 1, 2, 3], "B": [1, 1, 2, 4]})
        ...     ),
        ...     group_keys,
        ... )
        >>> df2 = GroupedDataFrame(
        ...     spark.createDataFrame(
        ...         pd.DataFrame({"A": [1, 2, 2, 3], "B": [1, 3, 4, 5]})
        ...     ),
        ...     group_keys,
        ... )
        >>> SumOf(SymmetricDifference()).distance(df1, df2, domain)
        4
        >>> # Using HammingDistance gives a distance of infinity since the groups
        >>> # are different sizes, despite the fact that the two dataframes are the
        >>> # same size.
        >>> SumOf(HammingDistance()).distance(df1, df2, domain)
        oo

        >>> # Absolute difference on pandas series first converts the floats to
        >>> # rationals, then exactly computes the distance.
        >>> domain = PandasSeriesDomain(NumpyFloatDomain())
        >>> series1 = pd.Series([1.2, 0.8])
        >>> series2 = pd.Series([0.3, 1.4])
        >>> SumOf(AbsoluteDifference()).distance(series1, series2, domain)
        27021597764222973/18014398509481984
    """

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a a valid distance for :attr:`~.inner_metric`

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            self.inner_metric.validate(value)
        except ValueError as e:
            raise ValueError(f"Invalid metric value for SumOf metric: {e}")

    def _aggregate(self, distances: Iterable[ExactNumber]) -> ExactNumber:
        """Aggregate the component distances.

        Args:
            distances: The list of distances to aggregate.
        """
        return sum(distances, ExactNumber(0))


class RootSumOfSquared(AggregationMetric):
    """The square root of the sum of the squares of component distances.

    These components may be elements of a series, groups of a grouped dataframe, or
    elements of a list. This metric is parameterized by an `inner_metric` that is used
    to compute the distances of the components.

    Example:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import SparkColumnsDescriptor
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkIntegerColumnDescriptor,
        ... )
        >>> spark = SparkSession.builder.getOrCreate()

        >>> # Symmetric difference on SparkGroupedDataFrame
        >>> group_keys = spark.createDataFrame(pd.DataFrame({"A": [1, 2]}))
        >>> domain = SparkGroupedDataFrameDomain(
        ...     {
        ...         "A": SparkIntegerColumnDescriptor(),
        ...         "B": SparkIntegerColumnDescriptor(),
        ...     },
        ...     group_keys,
        ... )
        >>> df1 = GroupedDataFrame(
        ...     spark.createDataFrame(
        ...         pd.DataFrame({"A": [1, 1, 2, 3], "B": [1, 1, 2, 4]})
        ...     ),
        ...     group_keys,
        ... )
        >>> df2 = GroupedDataFrame(
        ...     spark.createDataFrame(
        ...         pd.DataFrame({"A": [1, 2, 2, 3], "B": [1, 3, 4, 5]})
        ...     ),
        ...     group_keys,
        ... )
        >>> RootSumOfSquared(SymmetricDifference()).distance(df1, df2, domain)
        sqrt(10)
        >>> # Using HammingDistance gives a distance of infinity since the groups
        >>> # are different sizes, despite the fact that the two dataframes are the
        >>> # same size.
        >>> RootSumOfSquared(HammingDistance()).distance(df1, df2, domain)
        oo
    """

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a nonnegative real or infinity

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            validate_exact_number(
                value=value,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid value for metric RootSumOfSquared: {e}")

    def _aggregate(self, distances: Iterable[ExactNumber]) -> ExactNumber:
        """Aggregate the component distances.

        Args:
            distances: The list of distances to aggregate.
        """
        return ExactNumber(
            sp.sqrt(sum((d ** 2 for d in distances), ExactNumber(0)).expr)
        )


class OnColumn(ExactNumberMetric):
    """The value of a metric applied to a single column treated as a vector.

    Example:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkIntegerColumnDescriptor,
        ... )

        >>> spark = SparkSession.builder.getOrCreate()
        >>> domain = SparkDataFrameDomain(
        ...     {
        ...         "A": SparkIntegerColumnDescriptor(),
        ...         "B": SparkIntegerColumnDescriptor(),
        ...     }
        ... )
        >>> value1 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 23], "B": [3, 1]})
        ... )
        >>> value2 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [2, 20], "B": [1, 8]})
        ... )
        >>> OnColumn("A", SumOf(AbsoluteDifference())).distance(value1, value2, domain)
        4
        >>> OnColumn("B", RootSumOfSquared(AbsoluteDifference())).distance(
        ...     value1, value2, domain
        ... )
        sqrt(53)
    """

    @typechecked
    def __init__(self, column: str, metric: Union[SumOf, RootSumOfSquared]):
        """Constructor.

        Args:
            column: The column to apply the metric to.
            metric: The metric to apply.
        """
        self._column = column
        self._metric = metric

    @property
    def column(self) -> str:
        """Return the column to apply the metric to."""
        return self._column

    @property
    def metric(self) -> Union[SumOf, RootSumOfSquared]:
        """Return the metric to apply."""
        return self._metric

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a a valid distance for :attr:`~.metric`

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            self.metric.validate(value)
        except ValueError as e:
            raise ValueError(f"Invalid value for OnColumn metric on {self.column}: {e}")

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        return self.metric.compare(value1, value2)

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        return (
            isinstance(domain, SparkDataFrameDomain)
            and self.column in domain.schema
            and self.metric.supports_domain(
                PandasSeriesDomain(domain[self.column].to_numpy_domain())
            )
        )

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        # help mypy
        assert isinstance(domain, SparkDataFrameDomain)

        distance = self.metric.distance(
            value1.select(self.column).toPandas().loc[:, self.column],
            value2.select(self.column).toPandas().loc[:, self.column],
            PandasSeriesDomain(domain[self.column].to_numpy_domain()),
        )
        self.validate(distance)
        return distance

    def __repr__(self) -> str:
        """Returns string representation."""
        return (
            f"{self.__class__.__name__}(column={repr(self.column)},"
            f" metric={self.metric})"
        )


class OnColumns(Metric):
    """A tuple containing the values of multiple OnColumn metrics.

    Example:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkIntegerColumnDescriptor,
        ... )

        >>> spark = SparkSession.builder.getOrCreate()
        >>> domain = SparkDataFrameDomain(
        ...     {
        ...         "A": SparkIntegerColumnDescriptor(),
        ...         "B": SparkIntegerColumnDescriptor(),
        ...     }
        ... )
        >>> metric = OnColumns(
        ...     [
        ...         OnColumn("A", SumOf(AbsoluteDifference())),
        ...         OnColumn("B", RootSumOfSquared(AbsoluteDifference())),
        ...     ]
        ... )
        >>> value1 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 23], "B": [3, 1]})
        ... )
        >>> value2 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [2, 20], "B": [1, 8]})
        ... )
        >>> metric.distance(value1, value2, domain)
        (4, sqrt(53))
    """

    @typechecked
    def __init__(self, on_columns: List[OnColumn]):
        """Constructor.

        Args:
            on_columns: The OnColumn metrics to apply.
        """
        self._on_columns = on_columns

    @property
    def on_columns(self) -> List[OnColumn]:
        """Return the OnColumn metrics to apply."""
        return self._on_columns

    def validate(self, value: Tuple[ExactNumberInput, ...]):
        """Raises an error if `value` not a valid distance.

        * `value` must be a tuple with one value for each metric in :attr:`~.on_columns`
        * each value must be a valid distance for the corresponding metric

        Args:
            value: A distance between two datasets under this metric.
        """
        if not (isinstance(value, tuple) and len(value) == len(self.on_columns)):
            raise ValueError(
                f"Expecting a tuple of length {len(self._on_columns)}. Not {value}"
            )
        for column_value, on_column in zip(value, self.on_columns):
            try:
                on_column.validate(column_value)
            except ValueError as e:
                raise ValueError(f"Invalid value for OnColumns metric: {e}")

    def compare(
        self, value1: Tuple[ExactNumberInput, ...], value2: Tuple[ExactNumberInput, ...]
    ) -> bool:
        """Returns True if `value1` is less than or equal to `value2`.

        Args:
            value1: A distance between two datasets under this metric.
            value2: A distance between two datasets under this metric.
        """
        self.validate(value1)
        self.validate(value2)
        return all(
            metric.compare(element1, element2)
            for metric, element1, element2 in zip(self.on_columns, value1, value2)
        )

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        return isinstance(domain, SparkDataFrameDomain) and all(
            (on_column.supports_domain(domain) for on_column in self.on_columns)
        )

    def distance(
        self, value1: Any, value2: Any, domain: Domain
    ) -> Tuple[ExactNumber, ...]:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        # help mypy
        assert isinstance(domain, SparkDataFrameDomain)

        distance = tuple(
            column.distance(value1, value2, domain) for column in self.on_columns
        )
        self.validate(distance)
        return distance

    def __repr__(self) -> str:
        """Returns string representation."""
        return (
            f"{self.__class__.__name__}"
            "(on_columns=["
            f"{', '.join([repr(on_column) for on_column in self.on_columns])}])"
        )


class IfGroupedBy(ExactNumberMetric):
    """Distance between two DataFrames that shall be grouped by a given attribute.

    This metric is an upper bound on the distance for any fixed set of grouping keys.
    This assumes that the distance between two empty groups is zero, and the inner
    metric must satisfy this property.

    The grouping column cannot contain floating point values.

    Examples:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkIntegerColumnDescriptor,
        ... )

        >>> spark = SparkSession.builder.getOrCreate()
        >>> domain = SparkDataFrameDomain(
        ...     {
        ...         "A": SparkIntegerColumnDescriptor(),
        ...         "B": SparkIntegerColumnDescriptor(),
        ...         "C": SparkIntegerColumnDescriptor(),
        ...     },
        ... )
        >>> metric = IfGroupedBy("C", RootSumOfSquared(SymmetricDifference()))
        >>> value1 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 1, 3], "B": [2, 1, 4], "C": [1, 1, 2]}),
        ... )
        >>> value2 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [2, 1], "B": [1, 1], "C": [1, 1]})
        ... )
        >>> metric.distance(value1, value2, domain)
        sqrt(5)
        >>> metric = IfGroupedBy("C", SymmetricDifference())
        >>> value1 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 1, 3], "B": [2, 1, 4], "C": [1, 1, 2]}),
        ... )
        >>> value2 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 1], "B": [2, 1], "C": [1, 1]})
        ... )
        >>> metric.distance(value1, value2, domain)
        1
    """

    @typechecked
    def __init__(
        self,
        column: str,
        inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference],
    ):
        """Constructor.

        Args:
            column: Column that the DataFrame shall be grouped by.
            inner_metric: Metric to be applied to corresponding groups in
                the DataFrame.
        """
        self._column = column
        self._inner_metric = inner_metric

    @property
    def column(self) -> str:
        """Column that DataFrame shall be grouped by."""
        return self._column

    @property
    def inner_metric(self) -> Union[SumOf, RootSumOfSquared, SymmetricDifference]:
        """Metric to be applied for corresponding groups."""
        return self._inner_metric

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a a valid distance for :attr:`~.inner_metric`

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            self.inner_metric.validate(value)
        except ValueError as e:
            raise ValueError(f"Invalid value for IfGroupedBy metric: {e}")

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`.

        Args:
            value1: A distance between two datasets under this metric.
            value2: A distance between two datasets under this metric.
        """
        self.validate(value1)
        self.validate(value2)
        return self.inner_metric.compare(value1, value2)

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        if not (
            isinstance(domain, SparkDataFrameDomain) and self.column in domain.schema
        ):
            return False
        # We create a grouped data frame domain that would result from grouping on
        # column so that we can check the grouped data frame domain against the inner
        # metric. We don't know the value of the groupby keys, so we use an empty
        # dataframe. Whether the inner metric supports the domain shouldn't depend on
        # these values.
        grouped_df_domain = SparkGroupedDataFrameDomain(
            domain.schema,
            SparkSession.builder.getOrCreate().createDataFrame(
                pd.DataFrame({self.column: []}),
                schema=domain.project([self.column]).spark_schema,
            ),
        )
        return self.inner_metric.supports_domain(grouped_df_domain)

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        # help mypy
        assert isinstance(domain, SparkDataFrameDomain)

        groupby_keys = (
            value1.select(self.column).union(value2.select(self.column)).distinct()
        )
        # Constructing a GroupedDataFrame with empty rows but nonempty columns is not
        # allowed.  So, we hardcode the distance to be zero when there are no groups.
        if groupby_keys.count() == 0:
            return ExactNumber(0)
        distance = self._inner_metric.distance(
            GroupedDataFrame(value1, groupby_keys),
            GroupedDataFrame(value2, groupby_keys),
            SparkGroupedDataFrameDomain(domain.schema, groupby_keys),
        )
        self.validate(distance)
        return distance

    def __repr__(self) -> str:
        """Returns string representation."""
        return (
            f"{self.__class__.__name__}("
            f"column='{self.column}', inner_metric={self.inner_metric})"
        )


class DictMetric(Metric):
    """Distance between two dictionaries with identical sets of keys.

    Example:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkIntegerColumnDescriptor,
        ... )

        >>> spark = SparkSession.builder.getOrCreate()
        >>> metric = DictMetric(
        ...     {"x": AbsoluteDifference(), "y": SymmetricDifference()}
        ... )
        >>> domain = DictDomain(
        ...     {
        ...         "x": NumpyIntegerDomain(),
        ...         "y": SparkDataFrameDomain(
        ...             {
        ...                 "A": SparkIntegerColumnDescriptor(),
        ...                 "B": SparkIntegerColumnDescriptor(),
        ...             }
        ...         ),
        ...     }
        ... )
        >>> df1 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 1, 3], "B": [2, 1, 4]})
        ... )
        >>> df2 = spark.createDataFrame(pd.DataFrame({"A": [2, 1], "B": [1, 1]}))
        >>> value1 = {"x": np.int64(1), "y": df1}
        >>> value2 = {"x": np.int64(10), "y": df2}
        >>> metric.distance(value1, value2, domain)
        {'x': 9, 'y': 3}
    """

    @typechecked
    def __init__(self, key_to_metric: Dict[Any, Metric]):
        """Constructor.

        Args:
            key_to_metric: Mapping from dictionary key to metric.
        """
        self._key_to_metric = key_to_metric

    @property
    def key_to_metric(self) -> Dict[Any, Metric]:
        """Returns mapping from keys to metrics."""
        return self._key_to_metric

    def validate(self, value: Dict[Any, Any]):
        """Raises an error if `value` not a valid distance.

        * `value` must be a dictionary with the same keys as :attr:`~.key_to_metric`
        * each value in the dictionary must be a valid distance under the corresponding
          metric

        Args:
            value: A distance between two datasets under this metric.
        """
        if not isinstance(value, dict):
            raise ValueError("DictMetric value must be a python dictionary.")
        if set(self._key_to_metric) != set(value):
            raise ValueError(
                f"Invalid DictMetric value: Expected keys: {set(self._key_to_metric)}"
                f" not: {set(value)}"
            )
        for key, metric_value in value.items():
            try:
                self._key_to_metric[key].validate(metric_value)
            except ValueError as e:
                raise ValueError(f"Invalid value for DictMetric: {e}")

    def compare(self, value1: Dict[Any, Any], value2: Dict[Any, Any]) -> bool:
        """Returns True if `value1` is less than or equal to `value2`.

        Args:
            value1: A distance between two datasets under this metric.
            value2: A distance between two datasets under this metric.
        """
        self.validate(value1)
        self.validate(value2)
        return all(
            metric.compare(value1[key], value2[key])
            for key, metric in self.key_to_metric.items()
        )

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        return (
            isinstance(domain, DictDomain)
            and set(self.key_to_metric.keys()) == set(domain.key_to_domain.keys())
            and all(
                (
                    self.key_to_metric[k].supports_domain(domain[k])
                    for k in self.key_to_metric
                )
            )
        )

    def distance(self, value1: Any, value2: Any, domain: Domain) -> Dict[Any, Any]:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        # help mypy
        assert isinstance(domain, DictDomain)

        distance = {
            k: m.distance(value1[k], value2[k], domain[k])
            for k, m in self.key_to_metric.items()
        }
        self.validate(distance)
        return distance

    def __getitem__(self, key: Any) -> Metric:
        """Returns metric associated with given key."""
        return self.key_to_metric[key]

    def __len__(self) -> int:
        """Returns number of keys in the metric."""
        return len(self.key_to_metric)

    def __repr__(self) -> str:
        """Returns string representation."""
        sorted_key_to_metric = {
            key: self[key] for key in sorted(self.key_to_metric, key=str)
        }
        return f"{self.__class__.__name__}(key_to_metric={sorted_key_to_metric})"


class AddRemoveKeys(Metric):
    """The number of keys that dictionaries of dataframe differ by.

    This metric can be thought of as a extension of :class:`IfGroupedBy` with inner
    metric :class:`~SymmetricDifference`, except it is applied to a dictionary of
    dataframes, instead of a single dataframe.

    `AddRemoveKeys(X)` can be described in the following way:

    Sum over each key that appears in the key column in either
    neighbor, where the key column for dataframe df is given by `X[df]`.

    - 0 if both neighbors "match" for `X[df] = key`
    - 1 if only one neighbor has records for `X[df] = key`
    - 2 if both neighbor have records for `X[df] = key`, but they don't "match"

    The key column cannot containg floating point values, and all dataframes must have
    the same type for the key column. The key columns for the different dataframes may
    have different names.

    Examples:
        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> from tmlt.core.domains.spark_domains import (
        ...     SparkIntegerColumnDescriptor,
        ...     SparkStringColumnDescriptor,
        ... )
        >>> spark = SparkSession.builder.getOrCreate()
        >>> domain = DictDomain(
        ...     {
        ...         1: SparkDataFrameDomain(
        ...             {
        ...                 "A": SparkIntegerColumnDescriptor(),
        ...                 "B": SparkIntegerColumnDescriptor(),
        ...             },
        ...         ),
        ...         2: SparkDataFrameDomain(
        ...             {
        ...                 "C": SparkIntegerColumnDescriptor(),
        ...                 "D": SparkStringColumnDescriptor(),
        ...             },
        ...         ),
        ...     }
        ... )
        >>> metric = AddRemoveKeys({1: "A", 2: "C"})
        >>> # key=1 matches, key=2 is only in value1, key=3 is only in value2, key=4
        >>> # differs
        >>> value1 = {
        ...     1: spark.createDataFrame(
        ...             pd.DataFrame(
        ...             {
        ...                 "A": [1, 1, 2],
        ...                 "B": [1, 1, 1],
        ...             }
        ...         )
        ...     ),
        ...     2: spark.createDataFrame(
        ...         pd.DataFrame(
        ...             {
        ...                 "C": [1, 4],
        ...                 "D": ["1", "1"],
        ...             }
        ...         )
        ...     )
        ... }
        >>> value2 = {
        ...     1: spark.createDataFrame(
        ...             pd.DataFrame(
        ...             {
        ...                 "A": [1, 1, 3],
        ...                 "B": [1, 1, 1],
        ...             }
        ...         )
        ...     ),
        ...     2: spark.createDataFrame(
        ...         pd.DataFrame(
        ...             {
        ...                 "C": [1, 4],
        ...                 "D": ["1", "2"],
        ...             }
        ...         )
        ...     )
        ... }
        >>> metric.distance(value1, value2, domain)
        4
    """

    @typechecked
    def __init__(self, df_to_key_column: Dict[Any, str]):
        """Constructor.

        Args:
            df_to_key_column: A dictionary mapping dataframe names to the name of the
                key column in that dataframe.
        """
        self._df_to_key_column = df_to_key_column.copy()

    @property
    def df_to_key_column(self) -> Dict[Any, str]:
        """Returns the key column."""
        return self._df_to_key_column.copy()

    @typechecked
    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a nonnegative real or infinite

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            validate_exact_number(
                value=value,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid value for metric AbsoluteDifference: {e}")

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        self.validate(value1)
        self.validate(value2)
        return ExactNumber(value1) <= ExactNumber(value2)

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        if isinstance(domain, DictDomain):
            column_descriptor = None
            if set(domain.key_to_domain).symmetric_difference(
                set(self.df_to_key_column)
            ):
                return False
            for key, element_domain in domain.key_to_domain.items():
                id_column = self.df_to_key_column[key]
                if not isinstance(element_domain, SparkDataFrameDomain):
                    return False
                if id_column not in element_domain.schema:
                    return False
                if isinstance(
                    element_domain.schema[id_column], SparkFloatColumnDescriptor
                ):
                    return False
                if column_descriptor is None:
                    column_descriptor = element_domain.schema[id_column]
                else:
                    if element_domain.schema[id_column] != column_descriptor:
                        return False
            return True
        return False

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        assert isinstance(domain, DictDomain)
        keys_in_value1_elements = {}
        keys_in_value2_elements = {}
        for dict_key in domain.key_to_domain:
            keys_in_value1_elements[dict_key] = set(
                value1[dict_key]
                .select(self.df_to_key_column[dict_key])
                .distinct()
                .rdd.map(lambda x: x[0])
                .collect()
            )
            keys_in_value2_elements[dict_key] = set(
                value2[dict_key]
                .select(self.df_to_key_column[dict_key])
                .distinct()
                .rdd.map(lambda x: x[0])
                .collect()
            )
        value1_keys = reduce(lambda x, y: x | y, keys_in_value1_elements.values())
        value2_keys = reduce(lambda x, y: x | y, keys_in_value2_elements.values())
        added_keys = value2_keys - value1_keys
        removed_keys = value1_keys - value2_keys

        # keys which may have changed
        for key in value1_keys & value2_keys:
            for dict_key in domain.key_to_domain:
                df1 = value1[dict_key].filter(
                    sf.col(self.df_to_key_column[dict_key]).eqNullSafe(key)
                )
                df2 = value2[dict_key].filter(
                    sf.col(self.df_to_key_column[dict_key]).eqNullSafe(key)
                )
                if (
                    SymmetricDifference().distance(
                        df1, df2, domain.key_to_domain[dict_key]
                    )
                    > 0
                ):
                    added_keys.add(key)
                    removed_keys.add(key)
                    break
        distance = ExactNumber(len(added_keys) + len(removed_keys))
        self.validate(distance)
        return distance

    def __repr__(self) -> str:
        """Returns string representation."""
        return (
            f"{self.__class__.__name__}(df_to_key_column={repr(self.df_to_key_column)})"
        )
