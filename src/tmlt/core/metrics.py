"""Module containing metrics used for constructing measurements and transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import numpy as np  # noqa: F401 -- needed for doctests
import pandas as pd
import sympy as sp
from pyspark.sql import functions as sf
from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import (
    PandasDataFrameDomain,
    PandasFloatColumnDescriptor,
    PandasGroupedTableDomain,
    PandasSeriesDomain,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
)
from tmlt.core.exceptions import OutOfDomainError, UnsupportedCombinationError
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.format import Formattable, format_labeled_siblings, format_siblings
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.misc import ConciseFrozenSet
from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable, concat_rows
from tmlt.core.utils.pandas_grouping import distinct_rows, group_indices, row_keys
from tmlt.core.utils.validation import validate_exact_number


class Metric(Formattable, ABC):
    """Base class for input/output metrics."""

    @abstractmethod
    def validate(self, value: Any) -> None:
        """Raises an error if ``value`` not a valid distance.

        Args:
            value: A distance between two datasets under this metric.
        """

    @abstractmethod
    def compare(self, value1: Any, value2: Any) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``."""

    @abstractmethod
    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain."""

    @abstractmethod
    def distance(self, value1: Any, value2: Any, domain: Domain) -> Any:
        """Returns the metric distance between two elements of a supported domain."""

    def _validate_distance_arguments(
        self, value1: Any, value2: Any, domain: Domain
    ) -> None:
        """Raise an exception if the arguments to a distance method aren't valid."""
        if not self.supports_domain(domain):
            raise UnsupportedCombinationError(
                (self, domain), f"{repr(self)} does not support domain {repr(domain)}."
            )
        try:
            domain.validate(value1)
        except OutOfDomainError as exception:
            raise OutOfDomainError(
                domain, value1, "The first argument is not in the domain"
            ) from exception
        try:
            domain.validate(value2)
        except OutOfDomainError as exception:
            raise OutOfDomainError(
                domain, value2, "The second argument is not in the domain"
            ) from exception

    def __eq__(self, other: Any) -> bool:
        """Return True if both metrics are equal."""
        return repr(self) == repr(other)

    def __hash__(self) -> int:
        """Returns hash value."""
        return hash(repr(self))

    def __repr__(self) -> str:
        """Returns string representation."""
        return f"{self.__class__.__name__}()"


class NullMetric(Metric):
    """Metric for use when distance is undefined."""

    def validate(self, value: Any) -> None:
        """Raises an error if ``value`` not a valid distance.

        This method is not implemented.

        Args:
            value: A distance between two datasets under this metric.
        """
        raise NotImplementedError()

    def compare(self, value1: Any, value2: Any) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``.

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

    def validate(self, value: ExactNumberInput) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a nonnegative real or infinite

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
            raise ValueError(f"Invalid value for metric AbsoluteDifference: {e}") from e

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``."""
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
    as an item.

    Both descriptions of a pandas DataFrame are supported:
    :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`, the counterpart of
    :class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain`, compares rows the
    way Spark does (see :mod:`tmlt.core.utils.pandas_grouping`), so its distances
    agree with the Spark ones; the older
    :class:`~tmlt.core.domains.pandas_domains.PandasDataFrameDomain` compares them
    with Python tuple equality, under which no two NaNs are ever the same row.

    For grouped data there are a few cases:

    - If the group keys are different, the distance is infinity
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
        ...     ["B"],
        ... )
        >>> grouped_df1 = GroupedDataFrame(df1, group_keys)
        >>> grouped_df2 = GroupedDataFrame(df2, group_keys)
        >>> SymmetricDifference().distance(grouped_df1, grouped_df2, domain)
        3
    """

    def validate(self, value: ExactNumberInput) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a nonnegative integer or infinity

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
            raise ValueError(
                f"Invalid value for metric SymmetricDifference: {e}"
            ) from e

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``."""
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
                PandasTableDomain,
                SparkGroupedDataFrameDomain,
                PandasGroupedTableDomain,
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
        elif isinstance(domain, PandasTableDomain):
            # Rows are compared by their
            # :func:`~tmlt.core.utils.pandas_grouping.row_keys`, not by the
            # ``Counter(map(tuple, value.values))`` of the
            # PandasDataFrameDomain branch below, which is wrong in two ways
            # for a table of mixed dtypes: ``.values`` builds one homogeneous
            # array, so an int64 column beside a float64 one is upcast to
            # float64 and integers that differ only past 2**53 become one
            # value; and a tuple holding a NaN never equals itself, so a
            # NaN-bearing row counts as different from its own copy. Row keys
            # are per-column and null-safe, and put two rows in one group
            # exactly when Spark does, which is what makes this branch's
            # distances equal to the SparkDataFrameDomain branch's.
            table_counts1 = Counter(row_keys(value1))
            table_counts2 = Counter(row_keys(value2))
            distance = ExactNumber(
                sum((table_counts1 - table_counts2).values())
                + sum((table_counts2 - table_counts1).values())
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
            assert isinstance(
                domain, (PandasGroupedTableDomain, SparkGroupedDataFrameDomain)
            )
            # The group keys must agree exactly, and each group whose contents
            # differ contributes 2 -- or 1 when one side of the group is empty.
            # That is the whole algorithm for either backend; only asking a
            # group whether it is empty differs. A pandas group is a frame,
            # which len() answers for; a Spark group is a plan, and taking one
            # row is what asks without counting the rest.
            is_empty: Callable[[Any], bool] = (
                (lambda group: len(group) == 0)
                if isinstance(domain, PandasGroupedTableDomain)
                else (lambda group: len(group.head(1)) == 0)
            )
            groups1 = value1.get_groups()
            groups2 = value2.get_groups()
            if groups1.keys() != groups2.keys():
                return ExactNumber(sp.oo)
            group_domain = domain.get_group_domain()
            distance = ExactNumber(0)
            for key, group1 in groups1.items():
                group2 = groups2[key]
                if self.distance(group1, group2, group_domain) > 0:
                    distance += 1 if is_empty(group1) or is_empty(group2) else 2
            return distance


class HammingDistance(ExactNumberMetric):
    """The number of elements that are different between two sets of the same size.

    This metric is compatible with spark dataframes, pandas dataframes, and pandas
    series. It ignores ordering and, in the case of pandas, indices. That is, it treats
    each collection as a multiset of records.

    As for :class:`SymmetricDifference`, both descriptions of a pandas DataFrame are
    supported, and only :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`
    compares rows the way Spark does.

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

    def validate(self, value: ExactNumberInput) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a nonnegative and integer or infinity

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
            raise ValueError(f"Invalid value for metric HammingDistance: {e}") from e

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``."""
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
                PandasTableDomain,
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
            if value1.count() != value2.count():
                return sp.oo
            distance = ExactNumber(value1.exceptAll(value2).count())
            self.validate(distance)
            return distance
        elif isinstance(domain, PandasTableDomain):
            # Rows are compared by their
            # :func:`~tmlt.core.utils.pandas_grouping.row_keys`; see the same
            # branch of SymmetricDifference.distance for why the
            # PandasDataFrameDomain branch's row tuples are not reused. The
            # size mismatch gives the infinity the SparkDataFrameDomain branch
            # above returns as a bare sympy value; ExactNumber(sp.oo) compares
            # equal to it and matches this method's return type.
            table_counts1 = Counter(row_keys(value1))
            table_counts2 = Counter(row_keys(value2))
            if sum(table_counts1.values()) != sum(table_counts2.values()):
                return ExactNumber(sp.oo)
            distance = ExactNumber(sum((table_counts1 - table_counts2).values()))
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

    Components may be elements of a series, groups of a grouped dataframe, or
    elements of a list. This metric is parameterized by an ``inner_metric`` that
    is used to compute the distances of the components. See :class:`SumOf` or
    :class`RootSumOfSquared` for example usage.

    If the values are grouped dataframes, the groups must be the same for both values,
    or the distance is infinity.

    If the values are pandas series or lists, they must be the same size, or the
    distance is infinity. The index of the series is ignored.
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
        """Returns True if ``value1`` is less than or equal to ``value2``.

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

        if isinstance(domain, (SparkGroupedDataFrameDomain, PandasGroupedTableDomain)):
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
        if isinstance(domain, (SparkGroupedDataFrameDomain, PandasGroupedTableDomain)):
            # A grouped dataframe and a grouped table both hand over a mapping
            # from group key to that group's rows, and the distance is the
            # aggregate of the inner metric over the groups either way. The
            # annotations are what tell mypy the two mappings have one type:
            # narrowing the *values* with isinstance would leave their key
            # types to be joined, which they cannot be.
            groups1: Dict[Any, Any] = value1.get_groups()
            groups2: Dict[Any, Any] = value2.get_groups()
            if groups1.keys() != groups2.keys():
                return ExactNumber(sp.oo)
            group_domain = domain.get_group_domain()
            distance = self._aggregate(
                [
                    self.inner_metric.distance(groups1[key], groups2[key], group_domain)
                    for key in groups1
                ]
            )
        elif isinstance(domain, PandasSeriesDomain):
            # help mypy
            assert isinstance(value1, pd.Series)
            assert isinstance(value2, pd.Series)

            if value1.size != value2.size:
                return ExactNumber(sp.oo)

            distance = self._aggregate(
                map(
                    lambda x: self.inner_metric.distance(
                        x[0], x[1], domain.element_domain
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

            if len(value1) != len(value2):
                return ExactNumber(sp.oo)

            distance = self._aggregate(
                map(
                    lambda x: self.inner_metric.distance(
                        x[0], x[1], domain.element_domain
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


class SumOf(AggregationMetric):
    """Distances resulting from summing distances of its components.

    These components may be elements of a series, groups of a grouped dataframe, or
    elements of a list. This metric is parameterized by an ``inner_metric`` that is used
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
        ...     ["A"],
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

    def validate(self, value: ExactNumberInput) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a a valid distance for :attr:`~.inner_metric`

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            self.inner_metric.validate(value)
        except ValueError as e:
            raise ValueError(f"Invalid metric value for SumOf metric: {e}") from e

    def _aggregate(self, distances: Iterable[ExactNumber]) -> ExactNumber:
        """Aggregate the component distances.

        Args:
            distances: The list of distances to aggregate.
        """
        return sum(distances, ExactNumber(0))


class RootSumOfSquared(AggregationMetric):
    """The square root of the sum of the squares of component distances.

    These components may be elements of a series, groups of a grouped dataframe, or
    elements of a list. This metric is parameterized by an ``inner_metric`` that is used
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
        ...     ["A"],
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

    def validate(self, value: ExactNumberInput) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a nonnegative real or infinity

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
            raise ValueError(f"Invalid value for metric RootSumOfSquared: {e}") from e

    def _aggregate(self, distances: Iterable[ExactNumber]) -> ExactNumber:
        """Aggregate the component distances.

        Args:
            distances: The list of distances to aggregate.
        """
        return ExactNumber(sp.sqrt(sum((d**2 for d in distances), ExactNumber(0)).expr))


class OnColumn(ExactNumberMetric):
    """The value of a metric applied to a single column treated as a vector.

    The domain may describe a Spark table
    (:class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain`) or a pandas one
    (:class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`); either way the
    column is handed to :attr:`~.metric` as a :class:`pandas.Series` belonging to the
    :class:`~tmlt.core.domains.pandas_domains.PandasSeriesDomain` of the column
    descriptor's numpy domain. A column whose descriptor has no numpy domain -- a
    nullable integer, a date, or a timestamp, on either backend -- is therefore not
    supported, and :meth:`supports_domain` raises rather than returning False for one.

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

    def validate(self, value: ExactNumberInput) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a valid distance for :attr:`~.metric`

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            self.metric.validate(value)
        except ValueError as e:
            raise ValueError(
                f"Invalid value for OnColumn metric on {self.column}: {e}"
            ) from e

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``."""
        return self.metric.compare(value1, value2)

    def supports_domain(self, domain: Domain) -> bool:
        """Return True if the metric is implemented for the passed domain.

        Args:
            domain: The domain to check against.
        """
        return (
            isinstance(domain, (SparkDataFrameDomain, PandasTableDomain))
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
        if isinstance(domain, PandasTableDomain):
            # The Spark branch below hands the inner metric the column as a
            # pandas Series with a fresh 0..n-1 index; reading the column out
            # of a pandas frame gives the same Series, and reindexing it gives
            # it the same index. The domain ignores an index, but
            # PandasSeriesDomain.validate reads elements by *label*, so a frame
            # carrying some other index -- a slice of a larger one, say --
            # would not validate at all. Reindexing copies nothing the caller
            # holds: the returned Series is a new object over the same values.
            distance = self.metric.distance(
                value1[self.column].reset_index(drop=True),
                value2[self.column].reset_index(drop=True),
                PandasSeriesDomain(domain[self.column].to_numpy_domain()),
            )
            self.validate(distance)
            return distance
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
    def __init__(self, on_columns: Sequence[OnColumn]):
        """Constructor.

        Args:
            on_columns: The OnColumn metrics to apply.
        """
        self._on_columns = list(on_columns)

    @property
    def on_columns(self) -> List[OnColumn]:
        """Return the OnColumn metrics to apply."""
        return self._on_columns

    def validate(self, value: Tuple[ExactNumberInput, ...]) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a tuple with one value for each metric
          in :attr:`~.on_columns`
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
                raise ValueError(f"Invalid value for OnColumns metric: {e}") from e

    def compare(
        self, value1: Tuple[ExactNumberInput, ...], value2: Tuple[ExactNumberInput, ...]
    ) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``.

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
        return isinstance(domain, (SparkDataFrameDomain, PandasTableDomain)) and all(
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
        assert isinstance(domain, (SparkDataFrameDomain, PandasTableDomain))

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

    def _format_children(self) -> str:
        """Render the per-column metrics as sibling children."""
        return format_siblings(self.on_columns)


class IfGroupedBy(ExactNumberMetric):
    """Distance between two DataFrames that shall be grouped by a given set of columns.

    This metric is an upper bound on the distance for any fixed set of grouping keys.
    This assumes that the distance between two empty groups is zero, and the inner
    metric must satisfy this property.

    The grouping columns cannot contain floating point values.

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
        >>> metric = IfGroupedBy(["C"], RootSumOfSquared(SymmetricDifference()))
        >>> value1 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [1, 1, 3], "B": [2, 1, 4], "C": [1, 1, 2]}),
        ... )
        >>> value2 = spark.createDataFrame(
        ...     pd.DataFrame({"A": [2, 1], "B": [1, 1], "C": [1, 1]})
        ... )
        >>> metric.distance(value1, value2, domain)
        sqrt(5)
        >>> metric = IfGroupedBy(["C"], SymmetricDifference())
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
        columns: Collection[str],
        inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference],
    ):
        """Constructor.

        Args:
            columns: Columns that the DataFrame shall be grouped by.
            inner_metric: Metric to be applied to corresponding groups in
                the DataFrame.
        """
        if not columns:
            raise ValueError(
                "Cannot instantiate an IfGroupedBy with empty columns, but got: "
                f"{columns}"
            )
        if isinstance(columns, str):
            raise ValueError(
                f"IfGroupedBy columns cannot be a single string, but got: {columns}"
            )
        duplicate_columns = [
            column for column, count in Counter(columns).items() if count > 1
        ]
        if duplicate_columns:
            raise ValueError(
                "IfGroupedBy cannot have duplicate grouping columns, but these "
                f"appeared multiple times: {duplicate_columns}"
            )
        self._columns = ConciseFrozenSet(columns)
        self._inner_metric = inner_metric

    @property
    def columns(self) -> frozenset[str]:
        """Column that DataFrame shall be grouped by."""
        return self._columns

    @property
    def inner_metric(self) -> Union[SumOf, RootSumOfSquared, SymmetricDifference]:
        """Metric to be applied for corresponding groups."""
        return self._inner_metric

    def validate(self, value: ExactNumberInput) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a valid distance for :attr:`~.inner_metric`

        Args:
            value: A distance between two datasets under this metric.
        """
        try:
            self.inner_metric.validate(value)
        except ValueError as e:
            raise ValueError(f"Invalid value for IfGroupedBy metric: {e}") from e

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``.

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
        if isinstance(domain, PandasTableDomain):
            for column in self.columns:
                if column not in domain.schema:
                    return False
            grouped_table_domain = PandasGroupedTableDomain(domain.schema, self.columns)
            return self.inner_metric.supports_domain(grouped_table_domain)
        if not isinstance(domain, SparkDataFrameDomain):
            return False
        for column in self.columns:
            if column not in domain.schema:
                return False
        grouped_df_domain = SparkGroupedDataFrameDomain(domain.schema, self.columns)
        return self.inner_metric.supports_domain(grouped_df_domain)

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.
        """
        self._validate_distance_arguments(value1, value2, domain)
        if isinstance(domain, PandasTableDomain):
            return self._pandas_distance(value1, value2, domain)
        # help mypy
        assert isinstance(domain, SparkDataFrameDomain)

        ordered_groupby_columns = self._ordered_groupby_columns(domain)

        groupby_keys = (
            value1.select(ordered_groupby_columns)
            .union(value2.select(ordered_groupby_columns))
            .distinct()
        )
        # Constructing a GroupedDataFrame with empty rows but nonempty columns is not
        # allowed, so the distance is hardcoded to zero when there are no groups.
        if groupby_keys.count() == 0:
            return ExactNumber(0)
        distance = self._inner_metric.distance(
            GroupedDataFrame(value1, groupby_keys),
            GroupedDataFrame(value2, groupby_keys),
            SparkGroupedDataFrameDomain(domain.schema, self.columns),
        )
        self.validate(distance)
        return distance

    def _ordered_groupby_columns(
        self, domain: Union[SparkDataFrameDomain, PandasTableDomain]
    ) -> List[str]:
        """Returns the columns grouped by, in the domain's column order.

        :attr:`columns` is a set, so it has an order of its own that has nothing
        to do with the table's. Both backends' branches of :meth:`distance` need
        the table's, since that is the order the group keys they select come out
        in.

        Args:
            domain: The domain of the tables being compared, of either backend.
        """
        return [column for column in domain.schema if column in self.columns]

    def _pandas_distance(
        self, value1: pd.DataFrame, value2: pd.DataFrame, domain: PandasTableDomain
    ) -> ExactNumber:
        """Returns the distance between two pandas tables.

        This is the pandas branch of :meth:`distance`, split out so that the two
        backends' branches can be read side by side. It mirrors the Spark branch
        step for step: the group keys are every key appearing in either table,
        with :func:`~tmlt.core.utils.pandas_grouping.distinct_rows` taking the
        place of ``union().distinct()`` so that a null key and a NaN key stay
        distinct, and the same hardcoded zero covers the case where there are no
        keys at all.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: The domain the two tables belong to.
        """
        ordered_groupby_columns = self._ordered_groupby_columns(domain)

        key_frames = [value[ordered_groupby_columns] for value in (value1, value2)]
        # Concatenating only the non-empty frames keeps the dtypes of the
        # groupby columns, which pandas is entitled to change when an empty
        # frame takes part in a concatenation.
        nonempty_key_frames = [frame for frame in key_frames if len(frame) > 0]
        groupby_keys = distinct_rows(concat_rows(nonempty_key_frames or key_frames[:1]))
        # The Spark branch hardcodes the distance to zero when there are no
        # groups; this branch does the same, so that two tables that are both
        # empty are at distance zero under either backend.
        if len(groupby_keys) == 0:
            return ExactNumber(0)
        distance = self._inner_metric.distance(
            PandasGroupedTable(value1, groupby_keys),
            PandasGroupedTable(value2, groupby_keys),
            PandasGroupedTableDomain(domain.schema, self.columns),
        )
        self.validate(distance)
        return distance

    def __repr__(self) -> str:
        """Returns string representation."""
        return (
            f"{self.__class__.__name__}("
            f"columns={self.columns}, inner_metric={self.inner_metric})"
        )

    def __eq__(self, other: Any) -> bool:
        """We want to make sure column order is ignored when computing equality."""
        if type(other) is not IfGroupedBy:  # pylint: disable=C0123
            return False
        if self.inner_metric != other.inner_metric:
            return False
        if self.columns != other.columns:
            return False
        return True


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

    FORMAT_EXCLUDED_ATTRS = Metric.FORMAT_EXCLUDED_ATTRS | {"key_to_metric"}
    """Attributes hidden from the formatted output. @nodoc"""

    @typechecked
    def __init__(self, key_to_metric: Mapping[Any, Metric]):
        """Constructor.

        Args:
            key_to_metric: Mapping from dictionary key to metric.
        """
        self._key_to_metric: Dict[Any, Metric] = dict(key_to_metric.items())

    @property
    def key_to_metric(self) -> Dict[Any, Metric]:
        """Returns mapping from keys to metrics."""
        return self._key_to_metric.copy()

    def validate(self, value: Dict[Any, Any]) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a dictionary with the same keys as :attr:`~.key_to_metric`
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
                raise ValueError(f"Invalid value for DictMetric: {e}") from e

    def compare(self, value1: Dict[Any, Any], value2: Dict[Any, Any]) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``.

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

    def _format_children(self) -> str:
        """Render keyed metrics as labeled sibling children."""
        if not self._key_to_metric:
            return ""
        return format_labeled_siblings(
            (str(key), self[key]) for key in sorted(self.key_to_metric, key=str)
        )


class AddRemoveKeys(Metric):
    """The number of keys that dictionaries of dataframe differ by.

    This metric can be thought of as a extension of :class:`IfGroupedBy` with inner
    metric :class:`~SymmetricDifference`, except it is applied to a dictionary of
    dataframes, instead of a single dataframe.

    ``AddRemoveKeys(X)`` can be described in the following way:

    Sum over each key that appears in the key column in either
    neighbor, where the key column for dataframe df is given by ``X[df]``.

    - 0 if both neighbors "match" for ``X[df] = key``
    - 1 if only one neighbor has records for ``X[df] = key``
    - 2 if both neighbor have records for ``X[df] = key``, but they don't "match"

    The key column cannot contain floating point values, and all dataframes must have
    the same type for the key column. The key columns for the different dataframes may
    have different names.

    The dictionary's dataframes may all be Spark ones
    (:class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain`) or all be pandas
    ones (:class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`), never a
    mixture: a key of one backend and a key of the other are values of different
    Python types that no comparison here relates, so a mixed dictionary would silently
    report every key as added and removed. :meth:`supports_domain` returns False for
    one, and :meth:`distance` says so explicitly. Under the pandas domains, keys are
    compared as :mod:`tmlt.core.utils.pandas_grouping` compares them, which is how
    Spark compares them, so the distances agree across the two backends.

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

    FORMAT_EXCLUDED_ATTRS = Metric.FORMAT_EXCLUDED_ATTRS | {"df_to_key_column"}
    """Attributes hidden from the formatted output. @nodoc"""

    @typechecked
    def __init__(self, df_to_key_column: Mapping[Any, str]):
        """Constructor.

        Args:
            df_to_key_column: A dictionary mapping dataframe names to the name of the
                key column in that dataframe.
        """
        self._df_to_key_column: Dict[Any, str] = dict(df_to_key_column.items())

    @property
    def df_to_key_column(self) -> Dict[Any, str]:
        """Returns the key column."""
        return self._df_to_key_column.copy()

    @typechecked
    def validate(self, value: ExactNumberInput) -> None:
        """Raises an error if ``value`` not a valid distance.

        * ``value`` must be a nonnegative real or infinite

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
            raise ValueError(f"Invalid value for metric AbsoluteDifference: {e}") from e

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if ``value1`` is less than or equal to ``value2``."""
        self.validate(value1)
        self.validate(value2)
        return ExactNumber(value1) <= ExactNumber(value2)

    def _element_backends(self, domain: DictDomain) -> List[str]:
        """Returns the names of the backends the element domains are of.

        Every element domain that describes a table contributes its backend's
        name, once; anything else -- which :meth:`supports_domain` rejects --
        contributes nothing. More than one name means the dictionary mixes
        backends, which is not supported; see this class's docstring.

        Args:
            domain: The dictionary domain to inspect.

        Returns:
            The backend names, sorted, so that the error message built from
            them does not depend on the dictionary's order.
        """
        backends = set()
        for element_domain in domain.key_to_domain.values():
            if isinstance(element_domain, SparkDataFrameDomain):
                backends.add("Spark")
            elif isinstance(element_domain, PandasTableDomain):
                backends.add("pandas")
        return sorted(backends)

    def _rows_with_key(
        self, df: pd.DataFrame, groups: Dict[Tuple[Any, ...], Any], key: Tuple[Any, ...]
    ) -> pd.DataFrame:
        """Returns the rows of a pandas dataframe holding one key.

        Args:
            df: The dataframe to select from.
            groups: The dataframe's groups, as
                :func:`~tmlt.core.utils.pandas_grouping.group_indices` returns
                them.
            key: The key to select, which need not appear in ``df`` at all.

        Returns:
            The rows whose key column holds ``key``, which is an empty
            dataframe when the key does not appear.
        """
        positions = groups.get(key)
        # A positional selection, so the returned frame is a copy and the
        # caller's frame cannot be reached through it.
        return df.iloc[:0] if positions is None else df.iloc[positions]

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
            if len(self._element_backends(domain)) > 1:
                return False
            for key, element_domain in domain.key_to_domain.items():
                id_column = self.df_to_key_column[key]
                if not isinstance(
                    element_domain, (SparkDataFrameDomain, PandasTableDomain)
                ):
                    return False
                if id_column not in element_domain.schema:
                    return False
                if isinstance(
                    element_domain.schema[id_column],
                    (SparkFloatColumnDescriptor, PandasFloatColumnDescriptor),
                ):
                    return False
                if column_descriptor is None:
                    column_descriptor = element_domain.schema[id_column]
                elif element_domain.schema[id_column] != column_descriptor:
                    return False
            return True
        return False

    def distance(self, value1: Any, value2: Any, domain: Domain) -> ExactNumber:
        """Return the metric distance between two elements of a supported domain.

        Args:
            value1: An element of the domain.
            value2: An element of the domain.
            domain: A domain compatible with the metric.

        Raises:
            tmlt.core.exceptions.UnsupportedCombinationError: If ``domain`` is a
                dictionary domain mixing Spark and pandas dataframes. This is
                reported before the generic unsupported-domain error, which
                would not say what is wrong with it.
        """
        # Checked before _validate_distance_arguments so that a mixed
        # dictionary is reported as such; a domain that is not a DictDomain at
        # all has no backends and falls through to that generic error.
        backends = (
            self._element_backends(domain) if isinstance(domain, DictDomain) else []
        )
        if len(backends) > 1:
            raise UnsupportedCombinationError(
                (self, domain),
                f"{repr(self)} cannot be applied to a dictionary mixing "
                f"{' and '.join(backends)} dataframes: a key of one backend "
                "is never equal to a key of the other, so every key would be "
                "counted as both added and removed. Convert the dictionary to "
                "a single backend first.",
            )
        self._validate_distance_arguments(value1, value2, domain)
        # supports_domain accepts nothing but a DictDomain, and the check above
        # has already run; this is what tells mypy so.
        assert isinstance(domain, DictDomain)

        # One algorithm over two backends. All that differs is how a dataframe's
        # distinct keys are enumerated and how the rows holding one key are
        # taken out of it, so those two are bound here and the rest is written
        # once. The keys stay opaque -- a Spark value or a pandas group key --
        # and are only ever compared with keys the same backend produced.
        if backends == ["pandas"]:

            def enumerate_keys(df: Any, key_column: str) -> Dict[Any, Any]:
                """Returns each distinct key of a frame, with its rows' positions.

                ``group_indices`` is the null-safe counterpart of the Spark
                branch's ``select(...).distinct()`` and its ``eqNullSafe`` row
                filters, in one pass: it enumerates the distinct keys and the
                positions of each key's rows, comparing values the way Spark
                compares them (see :mod:`tmlt.core.utils.pandas_grouping`). A
                plain groupby would instead put a null and a NaN in one group
                and then drop it, losing every row with a null key.

                Args:
                    df: The dataframe whose keys are wanted.
                    key_column: The column holding the keys.
                """
                return group_indices(df, [key_column])

            def rows_with_key(
                df: Any, keys: Dict[Any, Any], key_column: str, key: Any
            ) -> Any:
                """Returns the rows of a dataframe holding one key.

                Args:
                    df: The dataframe to select from.
                    keys: That dataframe's keys, as ``enumerate_keys`` gave them.
                    key_column: The column holding the keys.
                    key: The key to select, which need not appear in ``df``.
                """
                del key_column  # The positions in `keys` already name it.
                return self._rows_with_key(df, keys, key)

        else:

            def enumerate_keys(df: Any, key_column: str) -> Dict[Any, Any]:
                """Returns each distinct key of a dataframe, mapped to nothing.

                A Spark row is selected by filtering, so there is nothing to
                remember about where a key's rows are.

                Args:
                    df: The dataframe whose keys are wanted.
                    key_column: The column holding the keys.
                """
                return dict.fromkeys(
                    df.select(key_column)
                    .distinct()
                    .rdd.map(lambda row: row[0])
                    .collect()
                )

            def rows_with_key(
                df: Any, keys: Dict[Any, Any], key_column: str, key: Any
            ) -> Any:
                """Returns the rows of a dataframe holding one key.

                Args:
                    df: The dataframe to select from.
                    keys: That dataframe's keys, which this backend does not need.
                    key_column: The column holding the keys.
                    key: The key to select, which need not appear in ``df``.
                """
                del keys
                return df.filter(sf.col(key_column).eqNullSafe(key))

        keys1: Dict[Any, Dict[Any, Any]] = {}
        keys2: Dict[Any, Dict[Any, Any]] = {}
        distinct1: Dict[Any, set] = {}
        distinct2: Dict[Any, set] = {}
        for dict_key in domain.key_to_domain:
            key_column = self.df_to_key_column[dict_key]
            keys1[dict_key] = enumerate_keys(value1[dict_key], key_column)
            keys2[dict_key] = enumerate_keys(value2[dict_key], key_column)
            distinct1[dict_key] = set(keys1[dict_key])
            distinct2[dict_key] = set(keys2[dict_key])
        value1_keys = reduce(lambda x, y: x | y, distinct1.values())
        value2_keys = reduce(lambda x, y: x | y, distinct2.values())
        added_keys = value2_keys - value1_keys
        removed_keys = value1_keys - value2_keys

        # keys which may have changed
        for key in value1_keys & value2_keys:
            for dict_key in domain.key_to_domain:
                key_column = self.df_to_key_column[dict_key]
                df1 = rows_with_key(value1[dict_key], keys1[dict_key], key_column, key)
                df2 = rows_with_key(value2[dict_key], keys2[dict_key], key_column, key)
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

    def _format_children(self) -> str:
        """Render each dataframe's key column on a separate labeled line."""
        if not self._df_to_key_column:
            return ""
        labels_and_values = [
            (str(key), self.df_to_key_column[key])
            for key in sorted(self.df_to_key_column, key=str)
        ]
        width = max(len(label) for label, _ in labels_and_values) + 2
        return "\n".join(
            f"* {(label + ':').ljust(width)}{value}"
            for label, value in labels_and_values
        )
