"""Unit tests for the column descriptors in :mod:`~tmlt.core.domains.pandas_domains`.

These cover :class:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor` and
its subclasses, and :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`.
The element-domain family in the same module --
:class:`~tmlt.core.domains.pandas_domains.PandasSeriesDomain` and
:class:`~tmlt.core.domains.pandas_domains.PandasDataFrameDomain` -- is covered by
``test_pandas_domains.py`` instead.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import textwrap
import time
from contextlib import nullcontext as does_not_raise
from itertools import combinations_with_replacement, product
from test.unit.backend_testing import floating_array
from test.unit.domains.abstract import DomainTests
from typing import Any, Callable, ContextManager, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from typeguard import TypeCheckError

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableColumnsDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from tmlt.core.utils.misc import get_fullname

_DATE = datetime.date(2020, 1, 1)
_TIMESTAMP = datetime.datetime(2020, 1, 1, 12, 30)


def _one_column_frame(values: Any, dtype: Any = None) -> pd.DataFrame:
    """Returns a single-column DataFrame holding ``values`` with column name "A"."""
    return pd.DataFrame({"A": pd.Series(values, dtype=dtype)})


def _nan_and_null_frame() -> pd.DataFrame:
    """Returns a frame whose single column holds a NaN and a null, distinctly.

    Only a :class:`pandas.arrays.FloatingArray` built from its values and its
    mask can hold both: every convenience constructor turns the NaN into a null.
    """
    return pd.DataFrame({"A": floating_array([1.0, np.nan, 2.0], [False, False, True])})


class TestPandasColumnDescriptors:
    r"""Tests for subclasses of class PandasColumnDescriptor.

    See subclasses of
    :class:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor`\ s.
    """

    @pytest.mark.parametrize(
        "descriptor, expected_domain",
        [
            (PandasIntegerColumnDescriptor(size=32), NumpyIntegerDomain(size=32)),
            (PandasIntegerColumnDescriptor(size=64), NumpyIntegerDomain(size=64)),
            (PandasFloatColumnDescriptor(size=32), NumpyFloatDomain(size=32)),
            (PandasFloatColumnDescriptor(size=64), NumpyFloatDomain(size=64)),
            (
                PandasFloatColumnDescriptor(size=64, allow_inf=True),
                NumpyFloatDomain(size=64, allow_inf=True),
            ),
            (
                PandasFloatColumnDescriptor(size=64, allow_nan=True),
                NumpyFloatDomain(size=64, allow_nan=True),
            ),
            (
                PandasStringColumnDescriptor(allow_null=True),
                NumpyStringDomain(allow_null=True),
            ),
            (
                PandasStringColumnDescriptor(allow_null=False),
                NumpyStringDomain(allow_null=False),
            ),
        ],
    )
    def test_to_numpy_domain(
        self, descriptor: PandasColumnDescriptor, expected_domain: Domain
    ):
        """Tests that to_numpy_domain works correctly."""
        assert descriptor.to_numpy_domain() == expected_domain

    def test_to_numpy_domain_nullable_float_warns(self):
        """A nullable float descriptor warns that nulls become nans."""
        descriptor = PandasFloatColumnDescriptor(allow_null=True)
        with pytest.warns(RuntimeWarning, match="converted to nans"):
            assert descriptor.to_numpy_domain() == NumpyFloatDomain()

    @pytest.mark.parametrize(
        "descriptor, expectation",
        [
            (
                PandasIntegerColumnDescriptor(allow_null=True),
                pytest.raises(
                    RuntimeError,
                    match="Nullable column does not have corresponding NumPy domain",
                ),
            ),
            (
                PandasDateColumnDescriptor(),
                pytest.raises(
                    RuntimeError, match="NumPy does not have support for date types"
                ),
            ),
            (
                PandasTimestampColumnDescriptor(),
                pytest.raises(
                    RuntimeError,
                    match="NumPy does not have support for timestamp types",
                ),
            ),
        ],
    )
    def test_to_numpy_domain_invalid(
        self, descriptor: PandasColumnDescriptor, expectation: ContextManager[None]
    ):
        """Tests that to_numpy_domain raises appropriate exceptions."""
        with expectation:
            descriptor.to_numpy_domain()

    @pytest.mark.parametrize(
        "descriptor, expected",
        [
            (
                PandasIntegerColumnDescriptor(allow_null=allow_null, size=size),
                SparkIntegerColumnDescriptor(allow_null=allow_null, size=size),
            )
            for allow_null, size in product([False, True], [32, 64])
        ]
        + [
            (
                PandasFloatColumnDescriptor(
                    allow_nan=allow_nan,
                    allow_inf=allow_inf,
                    allow_null=allow_null,
                    size=size,
                ),
                SparkFloatColumnDescriptor(
                    allow_nan=allow_nan,
                    allow_inf=allow_inf,
                    allow_null=allow_null,
                    size=size,
                ),
            )
            for allow_nan, allow_inf, allow_null, size in product(
                [False, True], [False, True], [False, True], [32, 64]
            )
        ]
        + [
            (
                pandas_type(allow_null=allow_null),
                spark_type(allow_null=allow_null),
            )
            for pandas_type, spark_type in [
                (PandasStringColumnDescriptor, SparkStringColumnDescriptor),
                (PandasDateColumnDescriptor, SparkDateColumnDescriptor),
                (PandasTimestampColumnDescriptor, SparkTimestampColumnDescriptor),
            ]
            for allow_null in [False, True]
        ],
    )
    def test_to_spark_descriptor(
        self, descriptor: PandasColumnDescriptor, expected: SparkColumnDescriptor
    ):
        """Every descriptor maps to the Spark descriptor for the same values."""
        actual = descriptor.to_spark_descriptor()
        assert actual == expected
        assert actual.allow_null == descriptor.allow_null

    @pytest.mark.parametrize(
        "descriptor, expected_dtype, expected_accepted",
        [
            (PandasIntegerColumnDescriptor(), "int64", {"int64", "Int64"}),
            (
                PandasIntegerColumnDescriptor(allow_null=True),
                "Int64",
                {"int64", "Int64"},
            ),
            (PandasIntegerColumnDescriptor(size=32), "int32", {"int32", "Int32"}),
            (
                PandasIntegerColumnDescriptor(allow_null=True, size=32),
                "Int32",
                {"int32", "Int32"},
            ),
            (PandasFloatColumnDescriptor(), "float64", {"float64", "Float64"}),
            (
                PandasFloatColumnDescriptor(allow_null=True),
                "Float64",
                {"float64", "Float64"},
            ),
            (PandasFloatColumnDescriptor(size=32), "float32", {"float32", "Float32"}),
            (
                PandasFloatColumnDescriptor(allow_null=True, size=32),
                "Float32",
                {"float32", "Float32"},
            ),
            (PandasStringColumnDescriptor(), "object", {"object"}),
            (PandasDateColumnDescriptor(), "object", {"object"}),
            (
                PandasTimestampColumnDescriptor(),
                "datetime64[ns]",
                {
                    "datetime64[s]",
                    "datetime64[ms]",
                    "datetime64[us]",
                    "datetime64[ns]",
                },
            ),
        ],
    )
    def test_dtypes(
        self,
        descriptor: PandasColumnDescriptor,
        expected_dtype: str,
        expected_accepted: set,
    ):
        """The canonical dtype and the accepted dtypes are as documented."""
        assert str(descriptor.pandas_dtype) == expected_dtype
        assert {str(dtype) for dtype in descriptor.accepted_dtypes} == expected_accepted
        assert descriptor.pandas_dtype in descriptor.accepted_dtypes

    @pytest.mark.parametrize(
        "descriptor, values, dtype, expectation",
        [
            # A numpy column has no null to find, whatever allow_null says.
            (PandasIntegerColumnDescriptor(), [1, 2], "int64", does_not_raise()),
            (
                PandasIntegerColumnDescriptor(allow_null=True),
                [1, 2],
                "int64",
                does_not_raise(),
            ),
            # An extension column without nulls is accepted either way.
            (PandasIntegerColumnDescriptor(), [1, 2], "Int64", does_not_raise()),
            (
                PandasIntegerColumnDescriptor(allow_null=True),
                [1, 2],
                "Int64",
                does_not_raise(),
            ),
            # An extension column with nulls is accepted only when they are.
            (
                PandasIntegerColumnDescriptor(),
                [1, None],
                "Int64",
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasIntegerColumnDescriptor(allow_null=True),
                [1, None],
                "Int64",
                does_not_raise(),
            ),
            # The same rules hold at 32 bits, and across dtype sizes.
            (
                PandasIntegerColumnDescriptor(size=32),
                [1, 2],
                "int32",
                does_not_raise(),
            ),
            (
                PandasIntegerColumnDescriptor(allow_null=True, size=32),
                [1, None],
                "Int32",
                does_not_raise(),
            ),
            (
                PandasIntegerColumnDescriptor(size=32),
                [1, 2],
                "int64",
                pytest.raises(
                    ValueError,
                    match="Column must have dtype Int32 or int32; got int64 instead",
                ),
            ),
            (
                PandasIntegerColumnDescriptor(),
                [1, 2],
                "int32",
                pytest.raises(
                    ValueError,
                    match="Column must have dtype Int64 or int64; got int32 instead",
                ),
            ),
            # A boolean column is not an integer column.
            (
                PandasIntegerColumnDescriptor(),
                [True, False],
                "bool",
                pytest.raises(
                    ValueError,
                    match="Column must have dtype Int64 or int64; got bool instead",
                ),
            ),
        ],
    )
    def test_validate_column_integer(
        self,
        descriptor: PandasColumnDescriptor,
        values: List[Any],
        dtype: str,
        expectation: ContextManager[None],
    ):
        """Integer columns are validated by the documented dtype rules."""
        with expectation:
            descriptor.validate_column(_one_column_frame(values, dtype), "A")

    @pytest.mark.parametrize(
        "descriptor, expectation",
        [
            (  # The null is found before the NaN.
                PandasFloatColumnDescriptor(),
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasFloatColumnDescriptor(allow_nan=True),
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (  # With nulls allowed, the unmasked NaN is still a NaN.
                PandasFloatColumnDescriptor(allow_null=True),
                pytest.raises(ValueError, match="Column contains NaN values"),
            ),
            (
                PandasFloatColumnDescriptor(allow_nan=True, allow_null=True),
                does_not_raise(),
            ),
        ],
    )
    def test_validate_column_float_nan_is_not_null(
        self, descriptor: PandasColumnDescriptor, expectation: ContextManager[None]
    ):
        """In an extension column, a NaN and a null are gated separately."""
        with expectation:
            descriptor.validate_column(_nan_and_null_frame(), "A")

    @pytest.mark.parametrize(
        "descriptor, values, dtype, expectation",
        [
            # A numpy column's NaN is a NaN, never a null: allowing nulls does
            # not allow it, and allowing NaNs does even when nulls are not.
            (
                PandasFloatColumnDescriptor(allow_null=True),
                [1.0, np.nan],
                "float64",
                pytest.raises(ValueError, match="Column contains NaN values"),
            ),
            (
                PandasFloatColumnDescriptor(allow_nan=True),
                [1.0, np.nan],
                "float64",
                does_not_raise(),
            ),
            (
                PandasFloatColumnDescriptor(),
                [1.0, np.nan],
                "float64",
                pytest.raises(ValueError, match="Column contains NaN values"),
            ),
            # A null in an extension column is a null.
            (
                PandasFloatColumnDescriptor(allow_nan=True),
                [1.0, None],
                "Float64",
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasFloatColumnDescriptor(allow_null=True),
                [1.0, None],
                "Float64",
                does_not_raise(),
            ),
            # Infinities are gated on their own, in either kind of column.
            (
                PandasFloatColumnDescriptor(),
                [1.0, np.inf],
                "float64",
                pytest.raises(ValueError, match="Column contains infinite values"),
            ),
            (
                PandasFloatColumnDescriptor(),
                [1.0, -np.inf],
                "float64",
                pytest.raises(ValueError, match="Column contains infinite values"),
            ),
            (
                PandasFloatColumnDescriptor(allow_inf=True),
                [1.0, np.inf, -np.inf],
                "float64",
                does_not_raise(),
            ),
            (
                PandasFloatColumnDescriptor(allow_null=True),
                [1.0, np.inf, None],
                "Float64",
                pytest.raises(ValueError, match="Column contains infinite values"),
            ),
            (
                PandasFloatColumnDescriptor(allow_inf=True, allow_null=True),
                [1.0, np.inf, None],
                "Float64",
                does_not_raise(),
            ),
            # Sizes are checked as they are for integers.
            (
                PandasFloatColumnDescriptor(size=32),
                [1.0, 2.0],
                "float32",
                does_not_raise(),
            ),
            (
                PandasFloatColumnDescriptor(size=32),
                [1.0, 2.0],
                "float64",
                pytest.raises(
                    ValueError,
                    match=(
                        "Column must have dtype Float32 or float32; got float64 instead"
                    ),
                ),
            ),
            (  # The Series constructor turns the NaN into a null; see
                # test_extension_constructor_collapses_nan_to_null.
                PandasFloatColumnDescriptor(allow_nan=True, size=32),
                [1.0, np.nan],
                "Float32",
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasFloatColumnDescriptor(allow_null=True, size=32),
                [1.0, np.nan],
                "Float32",
                does_not_raise(),
            ),
        ],
    )
    def test_validate_column_float(
        self,
        descriptor: PandasColumnDescriptor,
        values: List[Any],
        dtype: str,
        expectation: ContextManager[None],
    ):
        """Float columns are validated by the documented null/NaN/inf rules."""
        with expectation:
            descriptor.validate_column(_one_column_frame(values, dtype), "A")

    @pytest.mark.parametrize(
        "descriptor, values, expectation",
        [
            (PandasStringColumnDescriptor(), ["a", "b"], does_not_raise()),
            (
                PandasStringColumnDescriptor(),
                ["a", None],
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasStringColumnDescriptor(),
                ["a", np.nan],
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasStringColumnDescriptor(),
                ["a", pd.NA],
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasStringColumnDescriptor(allow_null=True),
                ["a", None, np.nan, pd.NA],
                does_not_raise(),
            ),
            (  # numpy strings are strings
                PandasStringColumnDescriptor(),
                [np.str_("a"), "b"],
                does_not_raise(),
            ),
            (
                PandasStringColumnDescriptor(),
                ["a", 2],
                pytest.raises(
                    ValueError,
                    match=r"Column must contain only str values; got int instead",
                ),
            ),
            (
                PandasStringColumnDescriptor(allow_null=True),
                ["a", None, 2, _DATE],
                pytest.raises(
                    ValueError,
                    match=(
                        r"Column must contain only str values; got datetime\.date,"
                        r" int instead"
                    ),
                ),
            ),
            (
                PandasStringColumnDescriptor(),
                [b"a"],
                pytest.raises(
                    ValueError,
                    match=r"Column must contain only str values; got bytes instead",
                ),
            ),
        ],
    )
    def test_validate_column_string(
        self,
        descriptor: PandasColumnDescriptor,
        values: List[Any],
        expectation: ContextManager[None],
    ):
        """String columns accept strings and the null markers pandas recognizes."""
        with expectation:
            descriptor.validate_column(_one_column_frame(values, object), "A")

    @pytest.mark.parametrize(
        "descriptor, values, expectation",
        [
            (PandasDateColumnDescriptor(), [_DATE], does_not_raise()),
            (
                PandasDateColumnDescriptor(),
                [_DATE, None],
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasDateColumnDescriptor(allow_null=True),
                [_DATE, None],
                does_not_raise(),
            ),
            (  # The deliberate strengthening: a datetime is not a date here.
                PandasDateColumnDescriptor(),
                [_DATE, _TIMESTAMP],
                pytest.raises(
                    ValueError,
                    match=(
                        r"Column must contain only datetime\.date values; got"
                        r" datetime\.datetime instead"
                    ),
                ),
            ),
            (  # ... and neither is a pandas Timestamp, which subclasses it.
                PandasDateColumnDescriptor(),
                [_DATE, pd.Timestamp(_TIMESTAMP)],
                pytest.raises(
                    ValueError,
                    match=(
                        r"Column must contain only datetime\.date values; got"
                        r" pandas\._libs\.tslibs\.timestamps\.Timestamp instead"
                    ),
                ),
            ),
            (
                PandasDateColumnDescriptor(),
                [_DATE, "2020-01-01"],
                pytest.raises(
                    ValueError,
                    match=(
                        r"Column must contain only datetime\.date values; got str"
                        r" instead"
                    ),
                ),
            ),
        ],
    )
    def test_validate_column_date(
        self,
        descriptor: PandasColumnDescriptor,
        values: List[Any],
        expectation: ContextManager[None],
    ):
        """Date columns accept dates, and not the datetimes that subclass them."""
        with expectation:
            descriptor.validate_column(_one_column_frame(values, object), "A")

    @pytest.mark.parametrize(
        "descriptor, column, expectation",
        [
            (
                PandasTimestampColumnDescriptor(),
                pd.to_datetime(pd.Series([_TIMESTAMP])),
                does_not_raise(),
            ),
            (
                PandasTimestampColumnDescriptor(),
                pd.to_datetime(pd.Series([_TIMESTAMP, None])),
                pytest.raises(ValueError, match="Column contains null values"),
            ),
            (
                PandasTimestampColumnDescriptor(allow_null=True),
                pd.to_datetime(pd.Series([_TIMESTAMP, None])),
                does_not_raise(),
            ),
            (
                PandasTimestampColumnDescriptor(),
                pd.to_datetime(pd.Series([_TIMESTAMP])).dt.tz_localize("UTC"),
                pytest.raises(
                    ValueError,
                    match=(
                        r"Column must be timezone-naive; got dtype"
                        r" datetime64\[ns, UTC\] instead\. Convert it with"
                        r" \.dt\.tz_convert\('UTC'\)\.dt\.tz_localize\(None\)"
                    ),
                ),
            ),
            (
                PandasTimestampColumnDescriptor(),
                pd.Series([_TIMESTAMP], dtype=object),
                pytest.raises(
                    ValueError, match=r"Column must have dtype .*; got object instead"
                ),
            ),
            (
                PandasTimestampColumnDescriptor(),
                pd.Series([_DATE], dtype=object),
                pytest.raises(
                    ValueError, match=r"Column must have dtype .*; got object instead"
                ),
            ),
        ],
    )
    def test_validate_column_timestamp(
        self,
        descriptor: PandasColumnDescriptor,
        column: pd.Series,
        expectation: ContextManager[None],
    ):
        """Timestamp columns must be timezone-naive datetime64 columns."""
        with expectation:
            descriptor.validate_column(pd.DataFrame({"A": column}), "A")

    def test_validate_column_timestamp_units(self):
        """Every datetime64 unit the running pandas can hold is accepted."""
        descriptor = PandasTimestampColumnDescriptor()
        reachable = []
        for unit in PandasTimestampColumnDescriptor.UNITS:
            dtype = np.dtype(f"datetime64[{unit}]")
            column = pd.Series(np.array(["2020-01-01T12:30"], dtype=dtype))
            if column.dtype != dtype:
                # pandas 1 holds every datetime column in nanoseconds.
                assert column.dtype == np.dtype("datetime64[ns]")
                continue
            reachable.append(unit)
            descriptor.validate_column(pd.DataFrame({"A": column}), "A")
        assert "ns" in reachable

    def test_extension_constructor_collapses_nan_to_null(self):
        """The nullable float constructors read a NaN as a null.

        This is pandas' behavior rather than the domain's, and it is why the
        NaN-and-null cases here build a
        :class:`pandas.arrays.FloatingArray` from its values and its mask.
        """
        assert list(pd.Series([1.0, np.nan], dtype="Float64").isna()) == [False, True]
        assert list(pd.Series([1.0, np.nan]).astype("Float64").isna()) == [False, True]

    @pytest.mark.parametrize(
        "descriptor, values, dtype",
        [
            (PandasIntegerColumnDescriptor(), [], "int64"),
            (PandasIntegerColumnDescriptor(allow_null=True), [], "Int64"),
            (PandasFloatColumnDescriptor(), [], "float64"),
            (PandasFloatColumnDescriptor(allow_null=True), [], "Float64"),
            (PandasStringColumnDescriptor(), [], object),
            (PandasDateColumnDescriptor(), [], object),
            (PandasTimestampColumnDescriptor(), [], "datetime64[ns]"),
        ],
    )
    def test_validate_column_empty(
        self, descriptor: PandasColumnDescriptor, values: List[Any], dtype: Any
    ):
        """An empty column of an accepted dtype is in the domain."""
        descriptor.validate_column(_one_column_frame(values, dtype), "A")

    def test_validate_column_empty_wrong_dtype(self):
        """An empty column is still held to the dtype rules."""
        with pytest.raises(ValueError, match="got float64 instead"):
            PandasIntegerColumnDescriptor().validate_column(
                _one_column_frame([], "float64"), "A"
            )

    def test_validate_column_missing(self):
        """Validating a column that is not in the DataFrame raises an error."""
        with pytest.raises(ValueError, match="'B' is not in the DataFrame"):
            PandasIntegerColumnDescriptor().validate_column(
                _one_column_frame([1], "int64"), "B"
            )

    def test_validate_column_duplicated(self):
        """Validating a duplicated column name raises an error."""
        df = pd.DataFrame([[1, 2]], columns=["A", "A"])
        with pytest.raises(ValueError, match="'A' is duplicated in the DataFrame"):
            PandasIntegerColumnDescriptor().validate_column(df, "A")

    def test_validate_column_ignores_index(self):
        """A column is validated the same whatever the DataFrame is indexed by."""
        df = _one_column_frame(["a", "b"], object)
        df.index = pd.Index(["x", "y"])
        PandasStringColumnDescriptor().validate_column(df, "A")

    @pytest.mark.parametrize(
        "descriptor, value, expected",
        [
            # Integers, including the numpy scalars indexing a column yields.
            (PandasIntegerColumnDescriptor(), 1, True),
            (PandasIntegerColumnDescriptor(), np.int64(1), True),
            (PandasIntegerColumnDescriptor(), np.int32(1), True),
            (PandasIntegerColumnDescriptor(), 2**63, False),
            (PandasIntegerColumnDescriptor(size=32), 2**31, False),
            (PandasIntegerColumnDescriptor(size=32), 2**31 - 1, True),
            (PandasIntegerColumnDescriptor(), True, False),
            (PandasIntegerColumnDescriptor(), np.bool_(True), False),
            (PandasIntegerColumnDescriptor(), 1.0, False),
            (PandasIntegerColumnDescriptor(), None, False),
            (PandasIntegerColumnDescriptor(allow_null=True), None, True),
            (PandasIntegerColumnDescriptor(allow_null=True), pd.NA, True),
            # Floats.
            (PandasFloatColumnDescriptor(), 1.5, True),
            (PandasFloatColumnDescriptor(), np.float64(1.5), True),
            (PandasFloatColumnDescriptor(size=32), 1e40, False),
            (PandasFloatColumnDescriptor(), float("nan"), False),
            (PandasFloatColumnDescriptor(allow_nan=True), float("nan"), True),
            (  # A NaN is a NaN, not a null, even for valid_py_value.
                PandasFloatColumnDescriptor(allow_null=True),
                float("nan"),
                False,
            ),
            (PandasFloatColumnDescriptor(), float("inf"), False),
            (PandasFloatColumnDescriptor(allow_inf=True), float("-inf"), True),
            (PandasFloatColumnDescriptor(allow_null=True), None, True),
            (PandasFloatColumnDescriptor(allow_null=True), pd.NA, True),
            (PandasFloatColumnDescriptor(), None, False),
            # Strings.
            (PandasStringColumnDescriptor(), "a", True),
            (PandasStringColumnDescriptor(), np.str_("a"), True),
            (PandasStringColumnDescriptor(), 1, False),
            (PandasStringColumnDescriptor(), None, False),
            (PandasStringColumnDescriptor(allow_null=True), None, True),
            (PandasStringColumnDescriptor(allow_null=True), float("nan"), True),
            # A NaN is a null here whichever type spells it: pandas.Series.isna
            # makes no distinction, and a numpy float32 is not a Python float
            # where a numpy float64 is.
            (PandasStringColumnDescriptor(allow_null=True), np.float32("nan"), True),
            (PandasStringColumnDescriptor(allow_null=True), np.float64("nan"), True),
            (PandasStringColumnDescriptor(), np.float32("nan"), False),
            # ... but in a float column a NaN is a value, whichever type spells
            # it, so allow_null does not admit one and allow_nan does.
            (PandasFloatColumnDescriptor(allow_null=True), np.float32("nan"), False),
            (PandasFloatColumnDescriptor(allow_nan=True), np.float32("nan"), True),
            # Dates, which datetimes are not.
            (PandasDateColumnDescriptor(), _DATE, True),
            (PandasDateColumnDescriptor(), _TIMESTAMP, False),
            (PandasDateColumnDescriptor(), pd.Timestamp(_TIMESTAMP), False),
            (PandasDateColumnDescriptor(), None, False),
            (PandasDateColumnDescriptor(allow_null=True), None, True),
            # Timestamps, which dates are not.
            (PandasTimestampColumnDescriptor(), _TIMESTAMP, True),
            (PandasTimestampColumnDescriptor(), pd.Timestamp(_TIMESTAMP), True),
            (PandasTimestampColumnDescriptor(), _DATE, False),
            (
                PandasTimestampColumnDescriptor(),
                _TIMESTAMP.replace(tzinfo=datetime.timezone.utc),
                False,
            ),
            (PandasTimestampColumnDescriptor(), None, False),
            (PandasTimestampColumnDescriptor(allow_null=True), None, True),
            (PandasTimestampColumnDescriptor(allow_null=True), pd.NaT, True),
            # A NaT is a datetime.datetime by subclassing, so it takes an
            # explicit answer rather than falling into the timestamp branch,
            # which would have accepted it however allow_null was set.
            (PandasTimestampColumnDescriptor(), pd.NaT, False),
            (PandasTimestampColumnDescriptor(), np.datetime64("NaT"), False),
            (
                PandasTimestampColumnDescriptor(allow_null=True),
                np.datetime64("NaT"),
                True,
            ),
            # A described column is datetime64[ns], so a value it cannot hold
            # is not a valid one -- unlike in Spark, whose TimestampType covers
            # years 1 to 9999.
            (PandasTimestampColumnDescriptor(), pd.Timestamp.min, True),
            (PandasTimestampColumnDescriptor(), pd.Timestamp.max, True),
            (PandasTimestampColumnDescriptor(), datetime.datetime(9999, 12, 31), False),
            (PandasTimestampColumnDescriptor(), datetime.datetime(1, 1, 1), False),
            (
                PandasTimestampColumnDescriptor(allow_null=True),
                datetime.datetime(9999, 12, 31),
                False,
            ),
        ],
    )
    def test_valid_py_value(
        self, descriptor: PandasColumnDescriptor, value: Any, expected: bool
    ):
        """Tests that valid_py_value works correctly."""
        assert descriptor.valid_py_value(value) == expected

    @pytest.mark.parametrize(
        "value",
        [float("nan"), np.float32("nan"), np.float64("nan"), np.datetime64("NaT")],
        ids=["float-nan", "float32-nan", "float64-nan", "datetime64-nat"],
    )
    def test_a_column_and_its_values_agree_about_nulls(self, value: Any):
        """A column and its own values are called null by the same rule.

        Column validation asks :meth:`pandas.Series.isna`, which makes no
        distinction between these spellings of a missing value; per-value
        validation asks ``_is_null``, which used to recognise only some of
        them. An object column holding a ``numpy.float32`` NaN therefore
        validated while the very same value, handed to a map function's output
        row, did not.
        """
        frame = _one_column_frame(["a", value], object)
        nullable = PandasStringColumnDescriptor(allow_null=True)
        nullable.validate_column(frame, "A")
        assert nullable.valid_py_value(value)

        not_nullable = PandasStringColumnDescriptor()
        with pytest.raises(ValueError, match="Column contains null values"):
            not_nullable.validate_column(frame, "A")
        assert not not_nullable.valid_py_value(value)

    def test_out_of_range_timestamp_is_not_a_valid_value(self):
        """A value a datetime64[ns] column cannot hold is out of the domain.

        Without this it passed validation and then failed, as a raw pandas
        ``OutOfBoundsDatetime``, inside whatever went on to build the column.
        """
        descriptor = PandasTimestampColumnDescriptor()
        too_late = datetime.datetime(9999, 12, 31)
        assert not descriptor.valid_py_value(too_late)
        with pytest.raises(pd.errors.OutOfBoundsDatetime):
            pd.Series([too_late], dtype=descriptor.pandas_dtype)

    @pytest.mark.parametrize(
        "descriptor, other_descriptor, expected",
        [
            (base, other, base == other)
            for base, other in combinations_with_replacement(
                [
                    PandasIntegerColumnDescriptor(),
                    PandasFloatColumnDescriptor(),
                    PandasStringColumnDescriptor(),
                    PandasDateColumnDescriptor(),
                    PandasTimestampColumnDescriptor(),
                ],
                2,
            )
        ]
        + [
            (
                PandasIntegerColumnDescriptor(size=32),
                PandasIntegerColumnDescriptor(size=32, allow_null=True),
                False,
            ),
            (
                PandasIntegerColumnDescriptor(size=32),
                PandasIntegerColumnDescriptor(size=64),
                False,
            ),
            (
                PandasFloatColumnDescriptor(allow_nan=True),
                PandasFloatColumnDescriptor(allow_inf=True),
                False,
            ),
            (
                PandasIntegerColumnDescriptor(),
                SparkIntegerColumnDescriptor(),
                False,
            ),
        ],
    )
    def test_eq(
        self,
        descriptor: PandasColumnDescriptor,
        other_descriptor: PandasColumnDescriptor,
        expected: bool,
    ):
        """Tests that __eq__ works correctly."""
        assert (descriptor == other_descriptor) == expected

    @pytest.mark.parametrize(
        "descriptor_type, args",
        [
            (PandasIntegerColumnDescriptor, {"size": 16}),
            (PandasFloatColumnDescriptor, {"size": 16}),
        ],
    )
    def test_invalid_size(self, descriptor_type: Type, args: Dict[str, Any]):
        """Only 32- and 64-bit columns can be described."""
        with pytest.raises(ValueError, match="size must be 32 or 64, not 16"):
            descriptor_type(**args)

    @pytest.mark.parametrize(
        "descriptor_type",
        [
            PandasIntegerColumnDescriptor,
            PandasFloatColumnDescriptor,
            PandasStringColumnDescriptor,
            PandasDateColumnDescriptor,
            PandasTimestampColumnDescriptor,
        ],
    )
    def test_invalid_allow_null(self, descriptor_type: Type):
        """allow_null must be a bool."""
        with pytest.raises(TypeCheckError):
            descriptor_type(allow_null="yes")

    @pytest.mark.parametrize(
        "descriptor, expected",
        [
            (
                PandasIntegerColumnDescriptor(),
                "PandasIntegerColumnDescriptor allow_null=False size=64",
            ),
            (
                PandasFloatColumnDescriptor(allow_nan=True, size=32),
                (
                    "PandasFloatColumnDescriptor allow_nan=True allow_inf=False"
                    " allow_null=False size=32"
                ),
            ),
            (
                PandasStringColumnDescriptor(allow_null=True),
                "PandasStringColumnDescriptor allow_null=True",
            ),
        ],
    )
    def test_format(self, descriptor: PandasColumnDescriptor, expected: str):
        """Descriptors render their flags inline, and not their dtypes."""
        assert descriptor.format() == expected


class TestPandasTableDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return PandasTableDomain

    @pytest.fixture
    def domain(self) -> PandasTableDomain:
        """Get a base PandasTableDomain."""
        return PandasTableDomain(
            schema={
                "A": PandasStringColumnDescriptor(),
                "B": PandasStringColumnDescriptor(),
                "C": PandasFloatColumnDescriptor(),
            }
        )

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {"schema": invalid_schema},
                pytest.raises(TypeCheckError, match='"schema"'),
                None,
            )
            for invalid_schema in [StringType, ListDomain(NumpyIntegerDomain())]
        ]
        + [
            (
                {"schema": {"A": PandasStringColumnDescriptor(), "B": DictDomain({})}},
                pytest.raises(TypeCheckError, match="'B'"),
                None,
            ),
            (
                {"schema": {"A": "B"}},
                pytest.raises(TypeCheckError, match="'A'"),
                None,
            ),
            (  # A Spark descriptor is not a pandas one.
                {"schema": {"A": SparkStringColumnDescriptor()}},
                pytest.raises(TypeCheckError, match="'A'"),
                None,
            ),
        ]
        + [
            ({"schema": valid_schema}, does_not_raise(), None)
            for valid_schema in [
                {
                    "A": PandasStringColumnDescriptor(),
                    "B": PandasStringColumnDescriptor(),
                },
                {},
                {"A": PandasStringColumnDescriptor()},
            ]
        ],
    )
    def test_construct_component(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The domain is constructed correctly and raises exceptions when initialized with
        invalid inputs.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        super().test_construct_component(
            domain_type, domain_args, expectation, exception_properties
        )

    @pytest.mark.parametrize(
        "other_domain, expected",
        [
            (  # matching
                PandasTableDomain(
                    {
                        "A": PandasStringColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                    }
                ),
                True,
            ),
            (  # shuffled
                PandasTableDomain(
                    {
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                        "A": PandasStringColumnDescriptor(),
                    }
                ),
                False,
            ),
            (  # Mismatching Types
                PandasTableDomain(
                    {
                        "A": PandasStringColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(size=32),
                    }
                ),
                False,
            ),
            (  # Extra attribute
                PandasTableDomain(
                    {
                        "A": PandasStringColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                        "D": PandasFloatColumnDescriptor(),
                    }
                ),
                False,
            ),
            (  # Missing attribute
                PandasTableDomain(
                    {
                        "A": PandasStringColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                    }
                ),
                False,
            ),
            (  # A different domain type with the same columns
                PandasTableDomain(
                    {
                        "A": PandasStringColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                    }
                ).project(["A", "B", "C"]),
                True,
            ),
        ],
    )
    def test_eq(self, domain: Domain, other_domain: Domain, expected: bool):
        """__eq__ works correctly.

        Args:
            domain: The domain to test.
            other_domain: The domain to compare to.
            expected: The expected result of the comparison.
        """
        super().test_eq(domain, other_domain, expected)

    @pytest.mark.parametrize(
        "domain_args, key, mutator",
        [
            (
                {
                    "schema": {
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                    }
                },
                "schema",
                mutator,
            )
            for mutator in [
                lambda x: x.update({"A": PandasFloatColumnDescriptor()}),
                lambda x: x.pop("A"),
                lambda x: x.clear(),
            ]
        ],
    )
    def test_mutable_inputs(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        key: str,
        mutator: Callable[[Any], Any],
    ):
        """The mutable inputs to the domain are copied.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        super().test_mutable_inputs(domain_type, domain_args, key, mutator)

    @pytest.mark.parametrize(
        "domain, expected_properties",
        [
            (
                PandasTableDomain(
                    schema={
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                    }
                ),
                {
                    "schema": {
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                    },
                    "carrier_type": pd.DataFrame,
                    "pandas_dtypes": {
                        "A": np.dtype("int64"),
                        "B": np.dtype(object),
                        "C": np.dtype("float64"),
                    },
                },
            ),
            (
                PandasTableDomain(
                    schema={
                        "A": PandasIntegerColumnDescriptor(allow_null=True),
                        "B": PandasStringColumnDescriptor(allow_null=True),
                        "C": PandasFloatColumnDescriptor(
                            allow_inf=True, allow_nan=True, allow_null=True
                        ),
                    }
                ),
                {
                    "schema": {
                        "A": PandasIntegerColumnDescriptor(allow_null=True),
                        "B": PandasStringColumnDescriptor(allow_null=True),
                        "C": PandasFloatColumnDescriptor(
                            allow_inf=True, allow_nan=True, allow_null=True
                        ),
                    },
                    "carrier_type": pd.DataFrame,
                    "pandas_dtypes": {
                        "A": pd.Int64Dtype(),
                        "B": np.dtype(object),
                        "C": pd.Float64Dtype(),
                    },
                },
            ),
        ],
    )
    def test_properties(self, domain: Domain, expected_properties: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            domain: The constructed domain to be tested.
            expected_properties: A dictionary containing all the property:value pairs
                domain is expected to have.
        """
        super().test_properties(domain, expected_properties)

    @pytest.mark.parametrize(
        "domain",
        [
            PandasTableDomain(
                schema={
                    "A": PandasIntegerColumnDescriptor(),
                    "B": PandasStringColumnDescriptor(),
                    "C": PandasFloatColumnDescriptor(),
                }
            )
        ],
    )
    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        super().test_property_immutability(domain)

    @pytest.mark.parametrize(
        "candidate, expectation, exception_properties",
        [
            (  # Not a DataFrame at all
                {"A": ["a"], "B": ["b"], "C": [1.0]},
                pytest.raises(
                    OutOfDomainError,
                    match=r"Value must be pandas\.core\.frame\.DataFrame, instead"
                    r" it is dict\.",
                ),
                None,
            ),
            (  # int64 instead of float64
                pd.DataFrame(
                    [["A", "B", 10], ["V", "E", 12], ["A", "V", 13]],
                    columns=["A", "B", "C"],
                ),
                pytest.raises(
                    OutOfDomainError,
                    match="Found invalid value in column 'C': Column must have dtype"
                    " Float64 or float64; got int64 instead",
                ),
                {
                    "domain": PandasTableDomain(
                        schema={
                            "A": PandasStringColumnDescriptor(),
                            "B": PandasStringColumnDescriptor(),
                            "C": PandasFloatColumnDescriptor(),
                        }
                    ),
                    "value": pd.DataFrame(
                        [["A", "B", 10], ["V", "E", 12], ["A", "V", 13]],
                        columns=["A", "B", "C"],
                    ),
                },
            ),
            (  # Missing Columns
                pd.DataFrame([["A", "B"], ["V", "E"], ["A", "V"]], columns=["A", "B"]),
                pytest.raises(
                    OutOfDomainError,
                    match="Columns are not as expected. DataFrame and Domain "
                    "must contain the same columns in the same order.\n"
                    r"DataFrame columns: \['A', 'B'\]"
                    "\n"
                    r"Domain columns: \['A', 'B', 'C'\]",
                ),
                None,
            ),
            (  # Duplicated columns
                pd.DataFrame(
                    [["A", "B", 1.1]],
                    columns=["A", "A", "C"],
                ),
                pytest.raises(
                    OutOfDomainError,
                    match=r"Some columns are duplicated, \['A'\]",
                ),
                None,
            ),
            (  # Reordered columns
                pd.DataFrame(
                    [["A", 1.1, "B"]],
                    columns=["A", "C", "B"],
                ),
                pytest.raises(
                    OutOfDomainError,
                    match=r"Columns are not as expected\.",
                ),
                None,
            ),
            (  # A null in a column that does not allow one
                pd.DataFrame(
                    [["A", "B", 1.1], ["V", "E", 1.2], ["A", "V", None]],
                    columns=["A", "B", "C"],
                ),
                pytest.raises(
                    OutOfDomainError,
                    match="Found invalid value in column 'C': Column contains NaN "
                    "values",
                ),
                None,
            ),
            (
                pd.DataFrame(
                    [["A", "B", 1.1], ["V", "E", 1.2], ["A", "V", 1.3]],
                    columns=["A", "B", "C"],
                ),
                does_not_raise(),
                None,
            ),
            (  # An empty frame with the right columns and dtypes
                pd.DataFrame(
                    {
                        "A": pd.Series([], dtype=object),
                        "B": pd.Series([], dtype=object),
                        "C": pd.Series([], dtype="float64"),
                    }
                ),
                does_not_raise(),
                None,
            ),
        ],
    )
    def test_validate(
        self,
        domain: Domain,
        candidate: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Validate works correctly.

        Args:
            domain: The domain to test.
            candidate: The value to validate using domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            domain.validate(candidate)
        if exception_properties is None or len(exception_properties) == 0:
            return
        # Help out mypy
        assert isinstance(exception, pytest.ExceptionInfo)
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception.value, prop), f"Expected prop was missing: {prop}"
            actual_value = getattr(exception.value, prop)
            if isinstance(actual_value, pd.DataFrame):
                pd.testing.assert_frame_equal(actual_value, expected_value)
                continue
            assert actual_value == expected_value, (
                f"Expected {prop} to be {expected_value}, got {actual_value}"
            )

    def test_validate_empty_domain(self):
        """A domain with no columns holds the DataFrames with no columns."""
        domain = PandasTableDomain({})
        domain.validate(pd.DataFrame())
        # An empty frame still has to have the domain's columns -- none.
        with pytest.raises(OutOfDomainError, match="Columns are not as expected"):
            domain.validate(pd.DataFrame({"A": pd.Series([], dtype="int64")}))

    def test_validate_all_types(self):
        """A frame with a column of every described type is in its domain."""
        domain = PandasTableDomain(
            {
                "int": PandasIntegerColumnDescriptor(),
                "nullable_int": PandasIntegerColumnDescriptor(allow_null=True),
                "float": PandasFloatColumnDescriptor(allow_nan=True, allow_inf=True),
                "nullable_float": PandasFloatColumnDescriptor(allow_null=True),
                "string": PandasStringColumnDescriptor(allow_null=True),
                "date": PandasDateColumnDescriptor(allow_null=True),
                "timestamp": PandasTimestampColumnDescriptor(allow_null=True),
            }
        )
        df = pd.DataFrame(
            {
                "int": pd.Series([1, 2], dtype="int64"),
                "nullable_int": pd.Series([1, None], dtype="Int64"),
                "float": pd.Series([np.nan, np.inf], dtype="float64"),
                "nullable_float": pd.Series([1.0, None], dtype="Float64"),
                "string": pd.Series(["a", None], dtype=object),
                "date": pd.Series([_DATE, None], dtype=object),
                "timestamp": pd.to_datetime(pd.Series([_TIMESTAMP, None])),
            }
        )
        domain.validate(df)
        assert df in domain

    def test_getitem(self, domain: PandasTableDomain):
        """__getitem__ returns the descriptor for a column."""
        assert domain["A"] == PandasStringColumnDescriptor()
        assert domain["C"] == PandasFloatColumnDescriptor()
        with pytest.raises(KeyError):
            domain["D"]

    @pytest.mark.parametrize(
        "cols, expected",
        [
            (
                ["A", "C"],
                PandasTableDomain(
                    {
                        "A": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                    }
                ),
            ),
            (  # The schema's ordering wins over the input's.
                ["C", "A"],
                PandasTableDomain(
                    {
                        "A": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                    }
                ),
            ),
            ([], PandasTableDomain({})),
            (
                ["A", "B", "C"],
                PandasTableDomain(
                    {
                        "A": PandasStringColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                        "C": PandasFloatColumnDescriptor(),
                    }
                ),
            ),
        ],
    )
    def test_project(
        self, domain: PandasTableDomain, cols: List[str], expected: PandasTableDomain
    ):
        """Project keeps the named columns, in the schema's order."""
        assert domain.project(cols) == expected

    def test_project_invalid(self, domain: PandasTableDomain):
        """Projecting to a column that is not in the schema raises an error."""
        with pytest.raises(ValueError, match="do not exist in this schema"):
            domain.project(["A", "D"])

    def test_format(self):
        """A table domain renders its columns as labeled siblings."""
        domain = PandasTableDomain(
            {
                "x": PandasIntegerColumnDescriptor(),
                "longer_key": PandasStringColumnDescriptor(),
            }
        )
        assert domain.format() == textwrap.dedent(
            """\
            PandasTableDomain
            * x:          PandasIntegerColumnDescriptor allow_null=False size=64
            * longer_key: PandasStringColumnDescriptor allow_null=False"""
        )
        assert PandasTableDomain({}).format() == "PandasTableDomain"

    def test_repr(self):
        """A table domain reprs as its constructor call."""
        domain = PandasTableDomain({"x": PandasIntegerColumnDescriptor()})
        assert repr(domain) == (
            "PandasTableDomain(schema={'x': "
            "PandasIntegerColumnDescriptor(allow_null=False, size=64)})"
        )


# Keyed by case name: the Spark type, the values to put in it, the pandas
# descriptor for the same values, the dtype toPandas() is expected to emit, and
# whether the emitted column has to be converted to satisfy the descriptor.
_ROUND_TRIP_CASES: Dict[str, Any] = {
    "long without nulls": (
        LongType(),
        [1, 2],
        PandasIntegerColumnDescriptor(),
        "int64",
        False,
    ),
    "int without nulls": (
        IntegerType(),
        [1, 2],
        PandasIntegerColumnDescriptor(size=32),
        "int32",
        False,
    ),
    "long with nulls": (
        LongType(),
        [1, None],
        PandasIntegerColumnDescriptor(allow_null=True),
        "float64",
        True,
    ),
    "int with nulls": (
        IntegerType(),
        [1, None],
        PandasIntegerColumnDescriptor(allow_null=True, size=32),
        "float64",
        True,
    ),
    "double without nulls": (
        DoubleType(),
        [1.0, 2.0],
        PandasFloatColumnDescriptor(),
        "float64",
        False,
    ),
    "float without nulls": (
        FloatType(),
        [1.0, 2.0],
        PandasFloatColumnDescriptor(size=32),
        "float32",
        False,
    ),
    "double with nulls": (
        DoubleType(),
        [1.0, None],
        PandasFloatColumnDescriptor(allow_null=True),
        "float64",
        True,
    ),
    "string without nulls": (
        StringType(),
        ["a", "b"],
        PandasStringColumnDescriptor(),
        "object",
        False,
    ),
    "string with nulls": (
        StringType(),
        ["a", None],
        PandasStringColumnDescriptor(allow_null=True),
        "object",
        False,
    ),
    "date without nulls": (
        DateType(),
        [_DATE, datetime.date(2021, 2, 3)],
        PandasDateColumnDescriptor(),
        "object",
        False,
    ),
    "date with nulls": (
        DateType(),
        [_DATE, None],
        PandasDateColumnDescriptor(allow_null=True),
        "object",
        False,
    ),
    "timestamp without nulls": (
        TimestampType(),
        [_TIMESTAMP, datetime.datetime(2021, 2, 3, 4, 5)],
        PandasTimestampColumnDescriptor(),
        "datetime64[ns]",
        False,
    ),
    "timestamp with nulls": (
        TimestampType(),
        [_TIMESTAMP, None],
        PandasTimestampColumnDescriptor(allow_null=True),
        "datetime64[ns]",
        False,
    ),
}


class TestSparkRoundTrip:
    """Tests pinning the dtype contract to what Spark's toPandas() emits.

    Two of Spark's conversions do not land on a dtype that satisfies the
    descriptor for the same values, and the domain deliberately does not widen
    to accept them:

    * A nullable ``LongType``/``IntegerType`` column comes back as ``float64``,
      with its nulls as NaNs -- Spark's integers do not survive the trip at all,
      and validating such a column as an integer one would mean accepting a
      float column whose values may have been silently rounded.
    * A nullable ``DoubleType``/``FloatType`` column comes back as a numpy float
      column, where a null is indistinguishable from a NaN. Accepting that as a
      nullable float column would erase the very distinction the descriptor
      draws.

    In both cases the documented recipe -- ``astype`` to the descriptor's
    canonical dtype -- restores it, and these tests assert both halves: that the
    raw conversion is rejected, and that the recipe is accepted.
    """

    @pytest.mark.parametrize(
        "spark_type, values, descriptor, expected_dtype, needs_conversion",
        _ROUND_TRIP_CASES.values(),
        ids=_ROUND_TRIP_CASES.keys(),
    )
    def test_to_pandas_round_trip(
        self,
        spark: SparkSession,
        spark_type: DataType,
        values: List[Any],
        descriptor: PandasColumnDescriptor,
        expected_dtype: str,
        needs_conversion: bool,
    ):
        """toPandas() of a described Spark column lands in the pandas domain."""
        sdf = spark.createDataFrame(
            [(value,) for value in values],
            schema=StructType([StructField("A", spark_type, True)]),
        )
        descriptor.to_spark_descriptor().validate_column(sdf, "A")

        pdf: pd.DataFrame = sdf.toPandas()
        assert str(pdf["A"].dtype) == expected_dtype

        domain = PandasTableDomain({"A": descriptor})
        if needs_conversion:
            with pytest.raises(OutOfDomainError):
                domain.validate(pdf)
            pdf = pdf.astype({"A": descriptor.pandas_dtype})
            assert pdf["A"].dtype == descriptor.pandas_dtype
        domain.validate(pdf)

    def test_null_double_round_trip_loses_the_null(self, spark: SparkSession):
        """A Spark NULL in a double column comes back as a NaN.

        The conversion recipe reads it back as a null, which is right for a
        column Spark held no NaNs in -- and is exactly why a numpy float column
        is never read as holding nulls.
        """
        sdf = spark.createDataFrame(
            [(1.0,), (None,), (float("nan"),)],
            schema=StructType([StructField("A", DoubleType(), True)]),
        )
        pdf: pd.DataFrame = sdf.toPandas()
        column = pdf["A"]
        # Spark's NULL and its NaN are both NaN here: the distinction is gone
        # before any pandas domain sees the column.
        assert str(column.dtype) == "float64"
        assert list(column.isna()) == [False, True, True]

        # Read as NaNs, the column is in a NaN-allowing domain and no other.
        nan_domain = PandasTableDomain(
            {"A": PandasFloatColumnDescriptor(allow_nan=True)}
        )
        nan_domain.validate(pd.DataFrame({"A": column}))
        with pytest.raises(OutOfDomainError, match="Column contains NaN values"):
            PandasTableDomain({"A": PandasFloatColumnDescriptor()}).validate(
                pd.DataFrame({"A": column})
            )

        # The recipe reads both values as nulls, not as NaNs.
        converted = pd.DataFrame({"A": column.astype("Float64")})
        PandasTableDomain({"A": PandasFloatColumnDescriptor(allow_null=True)}).validate(
            converted
        )
        assert list(converted["A"].isna()) == [False, True, True]

    def test_round_trip_all_columns(self, spark: SparkSession):
        """A whole frame round-trips into the domain built from its descriptors."""
        schema: PandasTableColumnsDescriptor = {
            "string": PandasStringColumnDescriptor(allow_null=True),
            "date": PandasDateColumnDescriptor(allow_null=True),
            "timestamp": PandasTimestampColumnDescriptor(allow_null=True),
            "long": PandasIntegerColumnDescriptor(),
            "double": PandasFloatColumnDescriptor(),
        }
        spark_schema = StructType(
            [
                StructField(name, desc.to_spark_descriptor().data_type, desc.allow_null)
                for name, desc in schema.items()
            ]
        )
        sdf = spark.createDataFrame(
            [("a", _DATE, _TIMESTAMP, 1, 1.5), (None, None, None, 2, 2.5)],
            schema=spark_schema,
        )
        pdf: pd.DataFrame = sdf.toPandas()
        PandasTableDomain(schema).validate(pdf)

    def test_empty_frame_round_trip(self, spark: SparkSession):
        """An empty Spark DataFrame converts to an empty frame in the domain."""
        schema: PandasTableColumnsDescriptor = {
            "long": PandasIntegerColumnDescriptor(),
            "double": PandasFloatColumnDescriptor(),
            "string": PandasStringColumnDescriptor(),
            "date": PandasDateColumnDescriptor(),
            "timestamp": PandasTimestampColumnDescriptor(),
        }
        spark_schema = StructType(
            [
                StructField(name, desc.to_spark_descriptor().data_type, False)
                for name, desc in schema.items()
            ]
        )
        pdf: pd.DataFrame = spark.createDataFrame([], schema=spark_schema).toPandas()
        assert len(pdf) == 0
        PandasTableDomain(schema).validate(pdf)


@pytest.mark.slow
def test_validate_is_vectorized():
    """Validating a million rows takes a moment, not a walk over the rows.

    The bound is generous next to the ~0.2s this takes when everything is
    vectorized, so that it does not turn into a benchmark of the machine
    running it; a per-row implementation of the same checks takes minutes.
    """
    rows = 1_000_000
    df = pd.DataFrame(
        {
            "int": pd.Series(np.arange(rows), dtype="int64"),
            "nullable_int": pd.array(np.arange(rows), dtype="Int64"),
            "float": pd.Series(np.arange(rows), dtype="float64"),
            "nullable_float": pd.array(np.arange(rows), dtype="Float64"),
            "string": pd.Series(["abcdef"] * rows, dtype=object),
            "date": pd.Series([_DATE] * rows, dtype=object),
            "timestamp": pd.to_datetime(pd.Series([_TIMESTAMP] * rows)),
        }
    )
    domain = PandasTableDomain(
        {
            "int": PandasIntegerColumnDescriptor(),
            "nullable_int": PandasIntegerColumnDescriptor(allow_null=True),
            "float": PandasFloatColumnDescriptor(),
            "nullable_float": PandasFloatColumnDescriptor(allow_null=True),
            "string": PandasStringColumnDescriptor(),
            "date": PandasDateColumnDescriptor(),
            "timestamp": PandasTimestampColumnDescriptor(),
        }
    )
    start = time.monotonic()
    domain.validate(df)
    elapsed = time.monotonic() - start
    assert elapsed < 2, f"Validating {rows} rows took {elapsed:.2f}s"


def test_get_fullname_of_descriptors():
    """The descriptors are where the error messages say they are."""
    assert (
        get_fullname(PandasIntegerColumnDescriptor())
        == "tmlt.core.domains.pandas_domains.PandasIntegerColumnDescriptor"
    )
