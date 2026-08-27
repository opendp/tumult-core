"""Domains for Pandas datatypes.

This module holds two independent families of domains.

The first describes a pandas object through the domain of its *elements*:
:class:`PandasSeriesDomain` wraps a
:class:`~tmlt.core.domains.numpy_domains.NumpyDomain`, and
:class:`PandasDataFrameDomain` maps each column to one such series domain. These
are the domains used by the aggregations in
:mod:`tmlt.core.measurements.aggregations`, where every value in a column is a
numpy scalar of a single type.

The second describes a pandas DataFrame the way
:mod:`tmlt.core.domains.spark_domains` describes a Spark DataFrame: through one
:class:`PandasColumnDescriptor` per column, collected in a
:class:`PandasTableDomain`. Each descriptor carries the same information as its
Spark counterpart -- the kind of value, its size in bits where that applies, and
whether nulls, NaNs and infinities are permitted -- so that both backends can
describe the same table. Unlike Spark, pandas has several dtypes that can hold
the same values, so each descriptor additionally fixes which dtypes a column it
describes may have; those rules are given in full in the class docstrings.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from __future__ import annotations

import datetime
import math
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Collection,
    Dict,
    Generic,
    Mapping,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from typeguard import check_type, typechecked

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.numpy_domains import (
    NumpyDomain,
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.utils.format import Formattable, format_labeled_siblings
from tmlt.core.utils.misc import ConciseFrozenSet, get_fullname

if TYPE_CHECKING:
    from tmlt.core.domains.spark_domains import SparkColumnDescriptor


@dataclass(frozen=True)
class PandasSeriesDomain(Domain):
    """Domain of Pandas Series.

    Note:
        The index is always ignored when this domain type is used.
    """

    element_domain: NumpyDomain
    """Domain of elements in the Series."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.element_domain, NumpyDomain)

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for members of the domain."""
        return pd.Series

    def validate(self, value: Any) -> None:
        """Raises error if value is not a DataFrame with matching schema."""
        # NOTE: Can not assert (elem in self.element_domain for elem in value) because
        # iterating over a Series implicitly calls item() on the NumPy values
        # retrieving the corresponding python object
        super().validate(value)
        for i in range(len(value)):
            try:
                self.element_domain.validate(value[i])
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    self, value, f"Found invalid value in Series: {exception}"
                ) from exception

    @classmethod
    def from_numpy_type(cls, dtype: np.dtype) -> "PandasSeriesDomain":
        """Returns a Pandas Series from a NumPy type."""
        return PandasSeriesDomain(NumpyDomain.from_np_type(dtype))


PandasColumnsDescriptor = Dict[str, PandasSeriesDomain]
"""Mapping from column name to column domain."""

_SchemaValue = TypeVar("_SchemaValue", bound=Formattable)
"""What a schema describes a column with: a domain, or a column descriptor."""


class _PandasSchemaDomain(Domain, Generic[_SchemaValue]):
    """Base of the pandas domains that describe a table column by column.

    Every domain in this module is a mapping from column name to a description
    of that column -- a :class:`PandasSeriesDomain` for
    :class:`PandasDataFrameDomain`, a :class:`PandasColumnDescriptor` for the
    others -- and holds it the same way: in a private dict, handed out as a
    copy, compared as an ordered mapping, and rendered as labeled siblings.
    That, and the check that a frame has the schema's columns in its order, is
    what lives here. What a carrier is, and what else makes one valid, belongs
    to the subclasses.
    """

    FORMAT_EXCLUDED_ATTRS = Domain.FORMAT_EXCLUDED_ATTRS | {"schema"}
    """Attributes hidden from output when formatting this domain. @nodoc"""

    _schema: Dict[str, _SchemaValue]

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"{self.__class__.__name__}(schema={self._schema})"

    @property
    def schema(self) -> Dict[str, _SchemaValue]:
        """Returns mapping from column name to that column's description."""
        return self._schema.copy()

    def __eq__(self, other: Any) -> bool:
        """Return True if the classes are equivalent."""
        if self.__class__ != other.__class__:
            return False
        return OrderedDict(self.schema) == OrderedDict(other.schema)

    def _format_children(self) -> str:
        """Render the column schema as labeled siblings."""
        if not self._schema:
            return ""
        return format_labeled_siblings(self._schema.items())

    def _validate_columns(self, value: pd.DataFrame) -> None:
        """Raises error unless a frame's columns are the schema's, in its order.

        Args:
            value: The DataFrame to check.
        """
        value_columns = list(value.columns)
        if len(value_columns) > len(set(value_columns)):
            duplicates = set(
                col for col in value_columns if value_columns.count(col) > 1
            )
            raise OutOfDomainError(
                self, value, f"Some columns are duplicated, {sorted(duplicates)}"
            )

        schema_columns = list(self._schema)
        if value_columns != schema_columns:
            raise OutOfDomainError(
                self,
                value,
                (
                    "Columns are not as expected. DataFrame and Domain must contain"
                    " the same columns in the same order.\nDataFrame columns:"
                    f" {value_columns}\nDomain columns: {schema_columns}"
                ),
            )


class PandasDataFrameDomain(_PandasSchemaDomain[PandasSeriesDomain]):
    """Domain of Pandas DataFrames."""

    @typechecked
    def __init__(self, schema: PandasColumnsDescriptor):
        """Constructor.

        Args:
            schema: Mapping from column name to column domain.
        """
        self._schema = schema.copy()

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for the domain."""
        return pd.DataFrame

    def validate(self, value: Any) -> None:
        """Raises error if value is not a Pandas DataFrame with matching schema."""
        super().validate(value)
        self._validate_columns(value)
        for column, element_domain in self._schema.items():
            try:
                element_domain.validate(value[column])
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    self,
                    value,
                    f"Found invalid value in column '{column}': {exception}",
                ) from exception

    @classmethod
    def from_numpy_types(cls, dtypes: Dict[str, np.dtype]) -> "PandasDataFrameDomain":
        """Returns a Pandas DataFrame domain from a dictionary of NumPy types."""
        col_to_desc = {
            col: PandasSeriesDomain.from_numpy_type(dtype)
            for col, dtype in dtypes.items()
        }
        return PandasDataFrameDomain(col_to_desc)


PandasDtype = Union[np.dtype, pd.api.extensions.ExtensionDtype]
"""Either a numpy dtype or one of pandas' extension dtypes."""


def _format_dtypes(dtypes: Collection[PandasDtype]) -> str:
    """Returns a deterministic listing of ``dtypes`` for use in error messages.

    Args:
        dtypes: The dtypes to render.
    """
    names = sorted(str(dtype) for dtype in dtypes)
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} or {names[-1]}"


def _is_null(val: Any) -> bool:
    """Returns True if ``val`` is one of the values pandas treats as null.

    This is what :meth:`pandas.Series.isna` reports for an element of an object
    column: ``None``, a NaN, ``pd.NA`` and ``NaT`` are all null there. It is the
    per-value counterpart of :meth:`PandasColumnDescriptor._null_mask`, which
    answers the same question for a whole column, and the two have to agree
    about the same value or a column and its rows would validate differently.

    A NaN or a ``NaT`` in *any* of its spellings counts, which is what that
    agreement requires: ``pandas.Series.isna`` does not care whether a NaN is a
    :class:`float` or a :class:`numpy.float32` -- which is not a :class:`float`,
    where a :class:`numpy.float64` is -- nor whether a ``NaT`` is ``pd.NaT`` or
    a raw ``numpy.datetime64("NaT")``.

    This deliberately takes no view on NaNs inside a *float* column, where a NaN
    is a NaN rather than a null. That is not this function's business: the float
    descriptor answers for a NaN itself, and only asks this about values that
    are not floats at all; see :class:`PandasFloatColumnDescriptor`.

    Args:
        val: The value to check.
    """
    return (
        val is None
        or val is pd.NA
        or val is pd.NaT
        or (isinstance(val, (float, np.floating)) and math.isnan(val))
        or (isinstance(val, np.datetime64) and bool(np.isnat(val)))
    )


class PandasColumnDescriptor(Formattable, ABC):
    """Base class for describing pandas column types.

    This is the pandas counterpart of
    :class:`~tmlt.core.domains.spark_domains.SparkColumnDescriptor`, and every
    subclass corresponds to one of its subclasses; see
    :meth:`to_spark_descriptor`.

    A descriptor constrains a column in two ways. Its :attr:`accepted_dtypes`
    are the dtypes a column it describes may have -- more than one, because
    pandas can hold the same values in a numpy array or in a nullable extension
    array -- of which :attr:`pandas_dtype` is the canonical one. Its flags then
    constrain the *values*, independently of which accepted dtype the column
    uses.

    Attributes:
        allow_null: If True, null values are permitted in the domain.
    """

    FORMAT_EXCLUDED_ATTRS = Formattable.FORMAT_EXCLUDED_ATTRS | {
        "pandas_dtype",
        "accepted_dtypes",
    }
    """Attributes hidden from output when formatting this descriptor. @nodoc"""

    allow_null: bool

    @abstractmethod
    def to_numpy_domain(self) -> NumpyDomain:
        """Returns corresponding NumPy domain."""

    @abstractmethod
    def to_spark_descriptor(self) -> SparkColumnDescriptor:
        """Returns the Spark descriptor for the same values.

        Note:
            This exists so that the two backends' descriptions of a table can be
            compared, and is not itself a conversion of any data: a column
            described by the returned descriptor holds the same values as one
            described by this descriptor, but the dtype rules that make this
            descriptor pandas-specific have no Spark equivalent and are dropped.
        """

    @property
    @abstractmethod
    def pandas_dtype(self) -> PandasDtype:
        """Returns the canonical dtype for a column described by this descriptor.

        This is the dtype that operations producing such a column should give it.
        It is always one of :attr:`accepted_dtypes`, which is what validation
        accepts.
        """

    @property
    @abstractmethod
    def accepted_dtypes(self) -> frozenset[PandasDtype]:
        """Returns every dtype a column described by this descriptor may have."""

    @abstractmethod
    def valid_py_value(self, val: Any) -> bool:
        """Returns True if ``val`` is valid for the described pandas column."""

    def validate_column(self, df: pd.DataFrame, col_name: str) -> None:
        """Raises error if not all values in given DataFrame column match descriptor.

        Args:
            df: pandas DataFrame to check.
            col_name: Name of column in df to be checked.
        """
        column = self._get_column(df, col_name)
        self._validate_dtype(column)
        self._validate_values(column)

    def _get_column(self, df: pd.DataFrame, col_name: str) -> pd.Series:
        """Returns the named column of ``df``, or raises an error.

        Args:
            df: pandas DataFrame to take the column from.
            col_name: Name of the column to take.
        """
        if col_name not in df.columns:
            raise ValueError(f"'{col_name}' is not in the DataFrame")
        column = df[col_name]
        if not isinstance(column, pd.Series):
            # Selecting a duplicated column name gives a DataFrame, which has
            # no single dtype to check. PandasTableDomain rejects duplicated
            # columns before it gets here, but validate_column is public.
            raise ValueError(f"'{col_name}' is duplicated in the DataFrame")
        return column

    def _validate_dtype(self, column: pd.Series) -> None:
        """Raises error if ``column`` does not have an accepted dtype.

        Args:
            column: The column to check.
        """
        if column.dtype not in self.accepted_dtypes:
            raise ValueError(
                f"Column must have dtype {_format_dtypes(self.accepted_dtypes)}; "
                f"got {column.dtype} instead"
            )

    def _validate_values(self, column: pd.Series) -> None:
        """Raises error if any value in ``column`` is not in the domain.

        Subclasses that constrain values beyond nullability extend this.

        Args:
            column: The column to check, already known to have an accepted dtype.
        """
        if not self.allow_null and self._null_mask(column).any():
            raise ValueError("Column contains null values.")

    def _null_mask(self, column: pd.Series) -> np.ndarray:
        """Returns the mask of the null values in ``column``.

        Args:
            column: The column to check, already known to have an accepted dtype.
        """
        return column.isna().to_numpy()

    def _invalid_element_types(self, column: pd.Series) -> list[type]:
        """Returns the types of the non-null values that are not in the domain.

        The types are sorted by name, so that the resulting error messages are
        deterministic. This walks an object column's values, which the numpy and
        extension dtypes make unnecessary for every other descriptor.

        Args:
            column: The column to check, already known to have an accepted dtype.
        """
        values = column[~self._null_mask(column)]
        return sorted(
            (
                element_type
                for element_type in values.map(type).unique()
                if not self._valid_element_type(element_type)
            ),
            key=get_fullname,
        )

    def _valid_element_type(self, element_type: type) -> bool:
        """Returns True if a non-null value of ``element_type`` may appear.

        Only the descriptors over object columns, whose dtype says nothing about
        what their values are, implement this.

        Args:
            element_type: The type to check.
        """
        raise NotImplementedError()


PandasTableColumnsDescriptor = Mapping[str, PandasColumnDescriptor]
"""Mapping from column name to :class:`PandasColumnDescriptor`."""


@dataclass(frozen=True)
class PandasIntegerColumnDescriptor(PandasColumnDescriptor):
    """Describes an integer attribute in pandas.

    Accepted dtypes:
        Either the numpy dtype for the descriptor's size (``int64``, or
        ``int32`` when ``size`` is 32) or the pandas nullable extension dtype
        for it (``Int64``/``Int32``), whatever the value of ``allow_null``.
        Nullability constrains the values rather than the dtype: a numpy column
        cannot hold a null at all, so it is acceptable however ``allow_null`` is
        set, and an extension column is acceptable when ``allow_null`` is False
        exactly when it happens to hold no null.

    Example:
        ..
            >>> import pandas as pd

        >>> descriptor = PandasIntegerColumnDescriptor()
        >>> descriptor.pandas_dtype
        dtype('int64')
        >>> descriptor.validate_column(pd.DataFrame({"A": [1, 2]}), "A")
        >>> descriptor.validate_column(
        ...     pd.DataFrame({"A": pd.array([1, 2], dtype="Int64")}), "A"
        ... )
        >>> descriptor.validate_column(
        ...     pd.DataFrame({"A": pd.array([1, None], dtype="Int64")}), "A"
        ... )
        Traceback (most recent call last):
        ValueError: Column contains null values.
        >>> PandasIntegerColumnDescriptor(allow_null=True).validate_column(
        ...     pd.DataFrame({"A": pd.array([1, None], dtype="Int64")}), "A"
        ... )
    """

    SIZE_TO_DTYPE: ClassVar[Dict[int, np.dtype]] = {
        32: np.dtype("int32"),
        64: np.dtype("int64"),
    }
    """Mapping from size to the numpy dtype of that size."""

    SIZE_TO_NULLABLE_DTYPE: ClassVar[Dict[int, pd.api.extensions.ExtensionDtype]] = {
        32: pd.Int32Dtype(),
        64: pd.Int64Dtype(),
    }
    """Mapping from size to the pandas nullable extension dtype of that size."""

    SIZE_TO_MIN_MAX: ClassVar = {
        32: (-2147483648, 2147483647),
        64: (-9223372036854775808, 9223372036854775807),
    }
    """Mapping from size to tuple of minimum and maximum value allowed."""

    allow_null: bool = False
    """If True, null values are permitted in the domain."""
    size: int = 64
    """Number of bits a member of the domain occupies. Must be 32 or 64."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.allow_null, bool)
        check_type(self.size, int)
        if self.size not in [32, 64]:
            raise ValueError(f"size must be 32 or 64, not {self.size}")

    def to_numpy_domain(self) -> NumpyDomain:
        """Returns corresponding NumPy domain."""
        if self.allow_null:
            raise RuntimeError(
                "Nullable column does not have corresponding NumPy domain."
            )
        return NumpyIntegerDomain(self.size)

    def to_spark_descriptor(self) -> SparkColumnDescriptor:
        """Returns the Spark descriptor for the same values."""
        # Imported here to avoid a circular import: spark_domains imports this
        # module for PandasDataFrameDomain.
        from tmlt.core.domains.spark_domains import (  # noqa: PLC0415
            SparkIntegerColumnDescriptor,
        )

        return SparkIntegerColumnDescriptor(allow_null=self.allow_null, size=self.size)

    @property
    def pandas_dtype(self) -> PandasDtype:
        """Returns the canonical dtype for a column described by this descriptor."""
        if self.allow_null:
            return self.SIZE_TO_NULLABLE_DTYPE[self.size]
        return self.SIZE_TO_DTYPE[self.size]

    @property
    def accepted_dtypes(self) -> frozenset[PandasDtype]:
        """Returns every dtype a column described by this descriptor may have."""
        return frozenset(
            {self.SIZE_TO_DTYPE[self.size], self.SIZE_TO_NULLABLE_DTYPE[self.size]}
        )

    def valid_py_value(self, val: Any) -> bool:
        """Returns True if value is a valid python value for the descriptor.

        Note:
            Unlike
            :meth:`~tmlt.core.domains.spark_domains.SparkIntegerColumnDescriptor.valid_py_value`,
            numpy integers are accepted as well as Python ones, because indexing
            a pandas column yields numpy scalars. Booleans are rejected, even
            though :class:`bool` is a subclass of :class:`int`, because pandas
            holds them in a dtype of their own.
        """
        if isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_)):
            min_, max_ = self.SIZE_TO_MIN_MAX[self.size]
            return bool(min_ <= val <= max_)
        return self.allow_null and _is_null(val)


@dataclass(frozen=True)
class PandasFloatColumnDescriptor(PandasColumnDescriptor):
    """Describes a float attribute in pandas.

    Accepted dtypes:
        Either the numpy dtype for the descriptor's size (``float64``, or
        ``float32`` when ``size`` is 32) or the pandas nullable extension dtype
        for it (``Float64``/``Float32``), whatever the values of the flags. As
        for :class:`PandasIntegerColumnDescriptor`, the flags constrain values
        rather than dtypes.

    Nulls and NaNs:
        pandas, like Spark, distinguishes a null from a NaN, but only in an
        extension column: a numpy float column has no mask, so its NaNs are
        NaNs and it cannot represent a null at all. Validation follows that
        exactly. A NaN in a numpy column is gated by ``allow_nan``, never by
        ``allow_null``, and a numpy column is therefore never rejected for
        holding a null. In an extension column, a masked value is a null, gated
        by ``allow_null``, and an unmasked NaN -- which a
        ``pandas.arrays.FloatingArray`` can hold alongside it -- is still a NaN,
        gated by ``allow_nan``.

        Some ways of building an extension column collapse the distinction:
        ``astype("Float64")`` converts a numpy column's NaNs into nulls, for
        instance. To keep both in one column, construct a
        ``pandas.arrays.FloatingArray`` from its values and its mask.

    Example:
        ..
            >>> import numpy as np
            >>> import pandas as pd

        >>> descriptor = PandasFloatColumnDescriptor(allow_null=True)
        >>> descriptor.pandas_dtype
        Float64Dtype()
        >>> nan_and_null = pd.DataFrame(
        ...     {
        ...         "A": pd.arrays.FloatingArray(
        ...             np.array([1.0, np.nan, 2.0]),
        ...             np.array([False, False, True]),
        ...         )
        ...     }
        ... )
        >>> descriptor.validate_column(nan_and_null, "A")
        Traceback (most recent call last):
        ValueError: Column contains NaN values.
        >>> PandasFloatColumnDescriptor(
        ...     allow_nan=True, allow_null=True
        ... ).validate_column(nan_and_null, "A")
        >>> PandasFloatColumnDescriptor(allow_nan=True).validate_column(
        ...     pd.DataFrame({"A": [1.0, np.nan]}), "A"
        ... )
    """

    SIZE_TO_DTYPE: ClassVar[Dict[int, np.dtype]] = {
        32: np.dtype("float32"),
        64: np.dtype("float64"),
    }
    """Mapping from size to the numpy dtype of that size."""

    SIZE_TO_NULLABLE_DTYPE: ClassVar[Dict[int, pd.api.extensions.ExtensionDtype]] = {
        32: pd.Float32Dtype(),
        64: pd.Float64Dtype(),
    }
    """Mapping from size to the pandas nullable extension dtype of that size."""

    allow_nan: bool = False
    """If True, NaNs are permitted in the domain."""
    allow_inf: bool = False
    """If True, infs are permitted in the domain."""
    allow_null: bool = False
    """If True, null values are permitted in the domain.

    Note:
        Only a column with one of the pandas nullable extension dtypes can hold
        a null; a numpy float column's NaNs are NaNs.
    """
    size: int = 64
    """Number of bits a member of the domain occupies. Must be 32 or 64."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.allow_nan, bool)
        check_type(self.allow_inf, bool)
        check_type(self.allow_null, bool)
        check_type(self.size, int)
        if self.size not in [32, 64]:
            raise ValueError(f"size must be 32 or 64, not {self.size}")

    def to_numpy_domain(self) -> NumpyDomain:
        """Returns corresponding NumPy domain."""
        if self.allow_null:
            warnings.warn(
                "Null values in a nullable pandas column are converted to nans in"
                " NumPy",
                RuntimeWarning,
            )
        return NumpyFloatDomain(
            allow_nan=self.allow_nan, allow_inf=self.allow_inf, size=self.size
        )

    def to_spark_descriptor(self) -> SparkColumnDescriptor:
        """Returns the Spark descriptor for the same values."""
        # Imported here to avoid a circular import: spark_domains imports this
        # module for PandasDataFrameDomain.
        from tmlt.core.domains.spark_domains import (  # noqa: PLC0415
            SparkFloatColumnDescriptor,
        )

        return SparkFloatColumnDescriptor(
            allow_nan=self.allow_nan,
            allow_inf=self.allow_inf,
            allow_null=self.allow_null,
            size=self.size,
        )

    @property
    def pandas_dtype(self) -> PandasDtype:
        """Returns the canonical dtype for a column described by this descriptor."""
        if self.allow_null:
            return self.SIZE_TO_NULLABLE_DTYPE[self.size]
        return self.SIZE_TO_DTYPE[self.size]

    @property
    def accepted_dtypes(self) -> frozenset[PandasDtype]:
        """Returns every dtype a column described by this descriptor may have."""
        return frozenset(
            {self.SIZE_TO_DTYPE[self.size], self.SIZE_TO_NULLABLE_DTYPE[self.size]}
        )

    def valid_py_value(self, val: Any) -> bool:
        """Returns True if value is a valid python value for the descriptor.

        In particular, this returns True only if one of the following is true:

        - val is ``float("nan")`` and NaN is allowed.
        - val is ``float("inf")`` or ``float("-inf")``, and inf values are allowed.
        - val is a float that can be represented in ``size`` bits.
        - val is a null value and nulls are allowed in the domain. Note that
          ``float("nan")`` is a NaN rather than a null here, matching how a
          column's values are validated.
        """
        if isinstance(val, (float, np.floating)):
            if np.isinf(val):
                return self.allow_inf
            if np.isnan(val):
                return self.allow_nan
            # Not to_numpy_domain().carrier_type, which the Spark descriptor
            # uses: that warns when nulls are allowed, and a predicate should
            # not warn. A value too large for the size overflows to an infinity
            # here, which is not equal to it, so the answer is False either way.
            with np.errstate(over="ignore"):
                return bool(self.SIZE_TO_DTYPE[self.size].type(val) == val)
        return self.allow_null and _is_null(val)

    def _null_mask(self, column: pd.Series) -> np.ndarray:
        """Returns the mask of the null values in ``column``.

        Only an extension column can hold a null; a numpy column's NaNs, which
        :meth:`pandas.Series.isna` reports as null, are NaNs.

        Args:
            column: The column to check, already known to have an accepted dtype.
        """
        if isinstance(column.dtype, pd.api.extensions.ExtensionDtype):
            return column.isna().to_numpy()
        return np.zeros(len(column), dtype=bool)

    def _validate_values(self, column: pd.Series) -> None:
        """Raises error if any value in ``column`` is not in the domain.

        Args:
            column: The column to check, already known to have an accepted dtype.
        """
        super()._validate_values(column)
        # Nulls become NaNs here, which the null mask separates back out; they
        # are never infinite, so the inf check needs no such correction.
        values = column.to_numpy(dtype=np.float64, na_value=np.nan)
        if not self.allow_nan and (np.isnan(values) & ~self._null_mask(column)).any():
            raise ValueError("Column contains NaN values.")
        if not self.allow_inf and np.isinf(values).any():
            raise ValueError("Column contains infinite values.")


@dataclass(frozen=True)
class PandasStringColumnDescriptor(PandasColumnDescriptor):
    """Describes a string attribute in pandas.

    Accepted dtypes:
        The ``object`` dtype, whose non-null values must all be :class:`str` (or
        instances of a subclass of it, such as :class:`numpy.str_`). Nulls are
        whatever :meth:`pandas.Series.isna` reports: ``None``, ``float("nan")``
        and ``pd.NA``.

        pandas' own string extension dtypes are deliberately not accepted, so
        that a described column has a single representation. Convert such a
        column with ``astype(object)`` first, which leaves ``pd.NA`` in place as
        a null.

    Example:
        ..
            >>> import pandas as pd

        >>> descriptor = PandasStringColumnDescriptor()
        >>> descriptor.validate_column(pd.DataFrame({"A": ["x", "y"]}), "A")
        >>> descriptor.validate_column(pd.DataFrame({"A": ["x", None]}), "A")
        Traceback (most recent call last):
        ValueError: Column contains null values.
        >>> descriptor.validate_column(pd.DataFrame({"A": ["x", 2]}), "A")
        Traceback (most recent call last):
        ValueError: Column must contain only str values; got int instead
    """

    allow_null: bool = False
    """If True, null values are permitted in the domain."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.allow_null, bool)

    def to_numpy_domain(self) -> NumpyStringDomain:
        """Returns corresponding NumPy domain."""
        return NumpyStringDomain(allow_null=self.allow_null)

    def to_spark_descriptor(self) -> SparkColumnDescriptor:
        """Returns the Spark descriptor for the same values."""
        # Imported here to avoid a circular import: spark_domains imports this
        # module for PandasDataFrameDomain.
        from tmlt.core.domains.spark_domains import (  # noqa: PLC0415
            SparkStringColumnDescriptor,
        )

        return SparkStringColumnDescriptor(allow_null=self.allow_null)

    @property
    def pandas_dtype(self) -> PandasDtype:
        """Returns the canonical dtype for a column described by this descriptor."""
        return np.dtype(object)

    @property
    def accepted_dtypes(self) -> frozenset[PandasDtype]:
        """Returns every dtype a column described by this descriptor may have."""
        return frozenset({np.dtype(object)})

    def valid_py_value(self, val: Any) -> bool:
        """Returns True if value is a valid python value for the descriptor."""
        return isinstance(val, str) or (self.allow_null and _is_null(val))

    def _valid_element_type(self, element_type: type) -> bool:
        """Returns True if a non-null value of ``element_type`` may appear.

        Args:
            element_type: The type to check.
        """
        return issubclass(element_type, str)

    def _validate_values(self, column: pd.Series) -> None:
        """Raises error if any value in ``column`` is not in the domain.

        Args:
            column: The column to check, already known to have an accepted dtype.
        """
        super()._validate_values(column)
        invalid_types = self._invalid_element_types(column)
        if invalid_types:
            raise ValueError(
                f"Column must contain only {get_fullname(str)} values; got "
                f"{', '.join(get_fullname(t) for t in invalid_types)} instead"
            )


@dataclass(frozen=True)
class PandasDateColumnDescriptor(PandasColumnDescriptor):
    """Describes a date attribute in pandas.

    Accepted dtypes:
        The ``object`` dtype, whose non-null values must all be exactly
        :class:`datetime.date`. Nulls are whatever :meth:`pandas.Series.isna`
        reports: ``None``, ``float("nan")`` and ``pd.NA``. pandas has no dtype
        for dates, and stores a :class:`datetime.date` column as objects; note
        that constructing a Series from dates without passing
        ``dtype=object`` converts them to timestamps instead.

    Subclasses of :class:`datetime.date`:
        A :class:`datetime.datetime` -- and so a :class:`pandas.Timestamp` --
        *is* a :class:`datetime.date`, so the ``isinstance`` check that
        :class:`~tmlt.core.domains.spark_domains.SparkDateColumnDescriptor` uses
        accepts one. This descriptor rejects it. A datetime carries a time of
        day that a date column has nowhere to put, and pandas neither drops it
        nor complains: comparisons, grouping and sorting all silently treat
        ``datetime(2020, 1, 1, 12)`` as distinct from ``date(2020, 1, 1)``,
        which for a domain describing dates is a bug that has already happened.
        Convert such values with ``.map(datetime.datetime.date)`` first.

    Example:
        ..
            >>> import datetime
            >>> import pandas as pd

        >>> descriptor = PandasDateColumnDescriptor()
        >>> dates = pd.Series(
        ...     [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)], dtype=object
        ... )
        >>> descriptor.validate_column(pd.DataFrame({"A": dates}), "A")
        >>> datetimes = pd.Series([datetime.datetime(2020, 1, 1)], dtype=object)
        >>> descriptor.validate_column(  # doctest: +NORMALIZE_WHITESPACE
        ...     pd.DataFrame({"A": datetimes}), "A"
        ... )
        Traceback (most recent call last):
        ValueError: Column must contain only datetime.date values; got
        datetime.datetime instead
    """

    allow_null: bool = False
    """If True, null values are permitted in the domain."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.allow_null, bool)

    def to_numpy_domain(self) -> NumpyDomain:
        """Returns corresponding NumPy domain.

        Note:
            Date types are not supported in NumPy; this method always
            raises an exception.
        """
        raise RuntimeError("NumPy does not have support for date types.")

    def to_spark_descriptor(self) -> SparkColumnDescriptor:
        """Returns the Spark descriptor for the same values."""
        # Imported here to avoid a circular import: spark_domains imports this
        # module for PandasDataFrameDomain.
        from tmlt.core.domains.spark_domains import (  # noqa: PLC0415
            SparkDateColumnDescriptor,
        )

        return SparkDateColumnDescriptor(allow_null=self.allow_null)

    @property
    def pandas_dtype(self) -> PandasDtype:
        """Returns the canonical dtype for a column described by this descriptor."""
        return np.dtype(object)

    @property
    def accepted_dtypes(self) -> frozenset[PandasDtype]:
        """Returns every dtype a column described by this descriptor may have."""
        return frozenset({np.dtype(object)})

    def valid_py_value(self, val: Any) -> bool:
        """Returns True if the value is a valid Python value for the descriptor.

        Note:
            Only a :class:`datetime.date` itself is valid; a
            :class:`datetime.datetime`, which is one by subclassing, is not.
        """
        return type(val) is datetime.date or (self.allow_null and _is_null(val))

    def _valid_element_type(self, element_type: type) -> bool:
        """Returns True if a non-null value of ``element_type`` may appear.

        Args:
            element_type: The type to check.
        """
        return element_type is datetime.date

    def _validate_values(self, column: pd.Series) -> None:
        """Raises error if any value in ``column`` is not in the domain.

        Args:
            column: The column to check, already known to have an accepted dtype.
        """
        super()._validate_values(column)
        invalid_types = self._invalid_element_types(column)
        if invalid_types:
            raise ValueError(
                f"Column must contain only {get_fullname(datetime.date)} values; got "
                f"{', '.join(get_fullname(t) for t in invalid_types)} instead"
            )


@dataclass(frozen=True)
class PandasTimestampColumnDescriptor(PandasColumnDescriptor):
    """Describes a timestamp attribute in pandas.

    Accepted dtypes:
        A timezone-naive ``datetime64`` dtype, in any of the units pandas
        supports: ``ns`` on every supported pandas version, and also ``s``,
        ``ms`` and ``us`` on pandas 2. The canonical unit is ``ns``. Nulls are
        ``NaT``.

        A timezone-aware column is rejected, since the instants it denotes
        depend on a timezone that a naive column, and Spark's ``TimestampType``
        as Core uses it, do not carry; the error says how to convert one.

    Range of representable values:
        Between :attr:`pandas.Timestamp.min` and :attr:`pandas.Timestamp.max`,
        which is roughly the years 1678 to 2262 -- the range a ``datetime64[ns]``
        column can hold. This is narrower than Spark's ``TimestampType``, which
        covers years 1 to 9999, and it is a limit of this engine rather than a
        choice: a described column's canonical dtype is ``datetime64[ns]``, and
        a value outside the range cannot be put in one. A
        :class:`datetime.datetime` outside it is therefore not a valid value,
        and :meth:`valid_py_value` says so, rather than letting an operation
        that builds a column fail later with a ``pandas`` ``OutOfBoundsDatetime``.

    Example:
        ..
            >>> import pandas as pd

        >>> descriptor = PandasTimestampColumnDescriptor()
        >>> timestamps = pd.to_datetime(pd.Series(["2020-01-01 12:00:00"]))
        >>> descriptor.validate_column(pd.DataFrame({"A": timestamps}), "A")
        >>> aware = timestamps.dt.tz_localize("America/New_York")
        >>> descriptor.validate_column(  # doctest: +NORMALIZE_WHITESPACE
        ...     pd.DataFrame({"A": aware}), "A"
        ... )
        Traceback (most recent call last):
        ValueError: Column must be timezone-naive; got dtype
        datetime64[ns, America/New_York] instead. Convert it with
        .dt.tz_convert('UTC').dt.tz_localize(None) to get naive UTC timestamps.
    """

    UNITS: ClassVar = ("s", "ms", "us", "ns")
    """The ``datetime64`` units a described column may use."""

    MIN_MAX: ClassVar = (pd.Timestamp.min, pd.Timestamp.max)
    """The smallest and largest value a described column can hold."""

    allow_null: bool = False
    """If True, null values are permitted in the domain."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.allow_null, bool)

    def to_numpy_domain(self) -> NumpyDomain:
        """Returns corresponding NumPy domain.

        Note:
            Timestamp types are not supported in NumPy; this method always
            raises an exception.
        """
        raise RuntimeError("NumPy does not have support for timestamp types.")

    def to_spark_descriptor(self) -> SparkColumnDescriptor:
        """Returns the Spark descriptor for the same values."""
        # Imported here to avoid a circular import: spark_domains imports this
        # module for PandasDataFrameDomain.
        from tmlt.core.domains.spark_domains import (  # noqa: PLC0415
            SparkTimestampColumnDescriptor,
        )

        return SparkTimestampColumnDescriptor(allow_null=self.allow_null)

    @property
    def pandas_dtype(self) -> PandasDtype:
        """Returns the canonical dtype for a column described by this descriptor."""
        return np.dtype("datetime64[ns]")

    @property
    def accepted_dtypes(self) -> frozenset[PandasDtype]:
        """Returns every dtype a column described by this descriptor may have."""
        return frozenset(np.dtype(f"datetime64[{unit}]") for unit in self.UNITS)

    def valid_py_value(self, val: Any) -> bool:
        """Returns True if the value is a valid Python value for the descriptor.

        Note:
            A timezone-aware :class:`datetime.datetime` is not valid, matching
            the dtypes a described column may have, and neither is one outside
            :attr:`MIN_MAX`; see this class' documentation.
        """
        # A NaT is a datetime.datetime by subclassing -- NaTType derives from
        # it -- so it has to be answered before the isinstance branch below,
        # which would otherwise call it a valid naive timestamp whatever
        # allow_null is set to. PandasDateColumnDescriptor keeps a datetime out
        # of a date column with an exact-type match, for the same reason.
        if _is_null(val):
            return self.allow_null
        if isinstance(val, datetime.datetime):
            if val.tzinfo is not None:
                return False
            min_, max_ = self.MIN_MAX
            return bool(min_ <= val <= max_)
        return False

    def _validate_dtype(self, column: pd.Series) -> None:
        """Raises error if ``column`` does not have an accepted dtype.

        Args:
            column: The column to check.
        """
        if isinstance(column.dtype, pd.DatetimeTZDtype):
            raise ValueError(
                f"Column must be timezone-naive; got dtype {column.dtype} instead."
                " Convert it with .dt.tz_convert('UTC').dt.tz_localize(None) to get"
                " naive UTC timestamps."
            )
        super()._validate_dtype(column)


class _PandasDescriptorDomain(_PandasSchemaDomain[PandasColumnDescriptor]):
    """Base of the domains describing columns with a :class:`PandasColumnDescriptor`.

    They are built the same way, which is what lives here: the schema is copied
    into a dict of its own, and every value in it is checked to be a descriptor.
    """

    @typechecked
    def __init__(self, schema: PandasTableColumnsDescriptor):
        """Constructor.

        Args:
            schema: Mapping from column names to column descriptors.
        """
        self._schema = dict(schema.items())
        # TODO(#2727): Remove this check once we update typeguard to ^3.0.0
        for key, domain in self._schema.items():
            if not isinstance(domain, PandasColumnDescriptor):
                raise TypeError(
                    f"Expected domain for key '{key}' to be a "
                    f"{get_fullname(PandasColumnDescriptor)}; got "
                    f"{get_fullname(domain)} instead"
                )


class _PandasDescribedTableDomain(_PandasDescriptorDomain):
    """Base of the domains whose carrier is a table described by descriptors.

    :class:`PandasTableDomain` describes a DataFrame and
    :class:`PandasGroupedTableDomain` a grouped one; either way there is a
    column to look up and a dtype for each column, which is what lives here.
    """

    FORMAT_EXCLUDED_ATTRS = _PandasDescriptorDomain.FORMAT_EXCLUDED_ATTRS | {
        "pandas_dtypes"
    }
    """Attributes hidden from output when formatting this domain. @nodoc"""

    @property
    def pandas_dtypes(self) -> Dict[str, PandasDtype]:
        """Returns the canonical dtype of each column according to the domain.

        Note:
            There isn't a one-to-one correspondence between these dtypes and
            the domains, since the domains encode additional information --
            about nulls in an integer column, or nans and infs in a float one --
            that a dtype cannot represent, and since a column may validly have a
            dtype other than its canonical one; see
            :attr:`PandasColumnDescriptor.accepted_dtypes`.
        """
        return {col: desc.pandas_dtype for col, desc in self._schema.items()}

    def __getitem__(self, col_name: str) -> PandasColumnDescriptor:
        """Returns column descriptor for given column."""
        return self._schema[col_name]


class PandasTableDomain(_PandasDescribedTableDomain):
    """Domain of pandas DataFrames described by column descriptors.

    This is the pandas counterpart of
    :class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain`. It is
    distinct from :class:`PandasDataFrameDomain`, which describes a DataFrame
    through the numpy domain of each column's elements; see this module's
    docstring.

    Note:
        The index is ignored, as it is by the other pandas domains here: a
        DataFrame is in the domain or not regardless of what it is indexed by.

    Mutability:
        A pandas DataFrame is mutable, and a transformation that mutates its
        input in place would break every guarantee Core derives from a
        transformation's stability -- the caller's frame, which some other
        component may hold and may already have accounted for, would change
        under it. Carriers of this domain are therefore treated as immutable by
        convention: a component given one may read it, and must not write to it
        or to anything sharing its buffers, including through a view returned by
        slicing or by ``DataFrame.values``. A component that needs to modify a
        frame copies it first, and returns the copy.

        This is a contract, not something the domain can check: validation sees
        one frame at one moment, and pandas offers no way to freeze it.
    """

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for the domain."""
        return pd.DataFrame

    def validate(self, value: Any) -> None:
        """Raises error if value is not a DataFrame with matching schema."""
        super().validate(value)
        # assertion to help mypy understand the type
        assert isinstance(value, pd.DataFrame)
        self._validate_columns(value)
        for column, descriptor in self._schema.items():
            try:
                descriptor.validate_column(value, column)
            except ValueError as exception:
                raise OutOfDomainError(
                    self,
                    value,
                    f"Found invalid value in column '{column}': {exception}",
                ) from exception

    def project(self, cols: Sequence[str]) -> "PandasTableDomain":
        """Project this domain to a subset of columns.

        The column ordering of the schema is used if it differs from the input
        ordering.

        Args:
            cols: The columns to keep.
        """
        unexpected_columns = set(cols) - set(self.schema)
        if unexpected_columns:
            raise ValueError(
                f"Columns {unexpected_columns} do not exist in this schema."
            )
        return PandasTableDomain(
            {column: domain for column, domain in self.schema.items() if column in cols}
        )


class PandasGroupedTableDomain(_PandasDescribedTableDomain):
    """Domain of grouped pandas tables.

    This is the pandas counterpart of
    :class:`~tmlt.core.domains.spark_domains.SparkGroupedDataFrameDomain`: its
    carriers are :class:`~tmlt.core.utils.pandas_grouped_table.PandasGroupedTable`
    objects, whose inner table belongs to the :class:`PandasTableDomain` with
    this domain's schema and whose group keys belong to that domain projected
    onto the groupby columns.

    A floating point column cannot be grouped by, as in Spark: the group a row
    falls into would then depend on a value with no exact representation.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable

        >>> domain = PandasGroupedTableDomain(
        ...     schema={
        ...         "A": PandasStringColumnDescriptor(),
        ...         "B": PandasIntegerColumnDescriptor(),
        ...     },
        ...     groupby_columns=["A"],
        ... )
        >>> domain.get_group_domain()
        PandasTableDomain(schema={'B': PandasIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> table = PandasGroupedTable(
        ...     dataframe=pd.DataFrame({"A": ["a1", "a2"], "B": [1, 2]}),
        ...     group_keys=pd.DataFrame({"A": ["a1", "a2"]}),
        ... )
        >>> table in domain
        True
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        schema: PandasTableColumnsDescriptor,
        groupby_columns: Collection[str],
    ):
        """Constructor.

        Args:
            schema: Mapping from column name to column descriptors for all columns.
            groupby_columns: List of columns used for grouping.

        Raises:
            ValueError: If ``groupby_columns`` has duplicates, names a column
                the schema does not have, or names a floating point column.
        """
        self._groupby_columns = ConciseFrozenSet(groupby_columns)
        if len(groupby_columns) != len(self.groupby_columns):
            raise ValueError("groupby_columns contains duplicate column names.")
        invalid_groupby_columns = self.groupby_columns - set(schema)
        if invalid_groupby_columns:
            raise ValueError(
                f"Invalid groupby columns: {ConciseFrozenSet(invalid_groupby_columns)}"
            )

        for column in groupby_columns:
            if isinstance(schema[column], PandasFloatColumnDescriptor):
                raise ValueError(f"Can not group by a floating point column: {column}")

        super().__init__(schema)

    @property
    def groupby_columns(self) -> frozenset[str]:
        """Returns list of columns used for grouping."""
        return self._groupby_columns

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return (
            f"{self.__class__.__name__}(schema={self.schema},"
            f" groupby_columns={self.groupby_columns})"
        )

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for the domain."""
        # avoid circular import
        from tmlt.core.utils.pandas_grouped_table import (  # noqa: PLC0415
            PandasGroupedTable,
        )

        return PandasGroupedTable

    def validate(self, value: Any) -> None:
        """Raises error if value is not a PandasGroupedTable with matching keys."""
        # avoid circular import
        from tmlt.core.utils.pandas_grouped_table import (  # noqa: PLC0415
            PandasGroupedTable,
        )

        super().validate(value)
        assert isinstance(value, PandasGroupedTable)
        inner_df_domain = PandasTableDomain(self.schema)
        try:
            inner_df_domain.validate(value.dataframe)
        except OutOfDomainError as exception:
            raise OutOfDomainError(
                self, value, f"Invalid inner DataFrame: {exception}"
            ) from exception

        group_key_domain = PandasTableDomain(
            {
                column: desc
                for column, desc in self.schema.items()
                if column in self.groupby_columns
            }
        )
        if value.group_keys is None:
            if group_key_domain.schema:
                raise OutOfDomainError(
                    self,
                    value,
                    "Invalid group keys: expected groups, but got total aggregation",
                )
        else:
            try:
                group_key_domain.validate(value.group_keys)
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    self, value, f"Invalid group keys: {exception}"
                ) from exception

    def get_group_domain(self) -> PandasTableDomain:
        """Return the domain for one of the groups."""
        group_schema = {
            column: v
            for column, v in self.schema.items()
            if column not in self.groupby_columns
        }
        return PandasTableDomain(group_schema)

    def __eq__(self, other: Any) -> bool:
        """Return True if the schemas and group keys are identical."""
        if not super().__eq__(other):
            return False
        return self.groupby_columns == other.groupby_columns


class PandasRowDomain(_PandasDescriptorDomain):
    """Domain of pandas DataFrame rows.

    This is the pandas counterpart of
    :class:`~tmlt.core.domains.spark_domains.SparkRowDomain`, and exists for the
    same reason: it is the domain of the row transformers that
    :class:`~tmlt.core.transformations.pandas_transformations.map.Map` applies
    to a :class:`PandasTableDomain`'s frames one row at a time.

    Carrier:
        A row is a plain :class:`dict` from column name to value, where a Spark
        row is a :class:`~pyspark.sql.Row`. pandas has no row object of its own
        -- a row taken out of a DataFrame is a :class:`~pandas.Series`, which
        has one dtype for the whole row and so cannot hold a row of mixed
        types without turning every value into an object -- and a dict is what
        a user function most naturally writes.

    Nulls:
        Whatever a column's dtype uses to mark a missing value, a row's value
        for it is ``None``: never ``NaN``, ``NaT`` or ``pd.NA``. A ``NaN`` in a
        row is therefore a NaN, exactly as it is in a
        :class:`PandasFloatColumnDescriptor`'s column. The full mapping, and
        which side of it each descriptor sits on, is documented on
        :class:`~tmlt.core.transformations.pandas_transformations.map.Map`,
        which is what builds these rows.

    Note:
        Like :class:`~tmlt.core.domains.spark_domains.SparkRowDomain`, this
        domain does not implement :meth:`validate`; use
        :meth:`PandasColumnDescriptor.valid_py_value` on a row's values.
    """

    def validate(self, value: Any) -> None:
        """Raises error if value is not a row with matching schema."""
        raise NotImplementedError()

    def __contains__(self, value: Any) -> bool:
        """Returns True if value is a row with matching schema."""
        raise NotImplementedError()

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for members of PandasRowDomain."""
        return dict
