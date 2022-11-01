"""Unit tests for :mod:`~tmlt.core.domains.pandas_domains`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from typing import Any, Optional
from unittest.case import TestCase

import numpy as np
import pandas as pd
from parameterized import parameterized

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class TestPandasSeriesDomain(TestCase):
    """Tests for :class:`~tmlt.core.domains.pandas_domains.PandasSeriesDomain`."""

    def setUp(self):
        """Setup."""
        self.int32_series_domain = PandasSeriesDomain(
            element_domain=NumpyIntegerDomain(size=32)
        )
        self.int64_series_domain = PandasSeriesDomain(
            element_domain=NumpyIntegerDomain(size=64)
        )
        self.float32_inf_series_domain = PandasSeriesDomain(
            element_domain=NumpyFloatDomain(size=32, allow_inf=True)
        )

    @parameterized.expand(
        [
            (
                PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
                False,
                True,
                False,
            ),
            (
                PandasSeriesDomain(element_domain=NumpyIntegerDomain(size=32)),
                True,
                False,
                False,
            ),
            (
                PandasSeriesDomain(element_domain=NumpyFloatDomain()),
                False,
                False,
                False,
            ),
            (
                PandasSeriesDomain(element_domain=NumpyFloatDomain(size=32)),
                False,
                False,
                False,
            ),
            (
                PandasSeriesDomain(
                    element_domain=NumpyFloatDomain(size=32, allow_inf=True)
                ),
                False,
                False,
                True,
            ),
            (NumpyFloatDomain(size=32), False, False, False),
        ]
    )
    def test_eq(
        self,
        domain: Domain,
        int32_series: bool,
        int64_series: bool,
        float32_series: bool,
    ):
        """Tests that __eq__ works correctly."""
        self.assertEqual(domain == self.int32_series_domain, int32_series)
        self.assertEqual(domain == self.int64_series_domain, int64_series)
        self.assertEqual(domain == self.float32_inf_series_domain, float32_series)

    @parameterized.expand(
        [
            (pd.Series([1, 2], dtype=np.dtype("int32")), True, False, False),
            (pd.Series([1, 2], dtype=np.dtype("int64")), False, True, False),
            (pd.Series([1.0, 2.0], dtype=np.dtype("float32")), False, False, True),
            (
                pd.Series([1.0, float("inf")], dtype=np.dtype("float32")),
                False,
                False,
                True,
            ),
            (
                pd.Series([1.0, float("nan")], dtype=np.dtype("float32")),
                False,
                False,
                False,
            ),
            ("Not a member", False, False, False),
        ]
    )
    def test_validate(
        self,
        candidate: Any,
        in_int32_domain: bool,
        in_int64_domain: bool,
        in_float32_domain: bool,
    ):
        """Tests that validate works correctly."""
        if not isinstance(candidate, pd.Series):
            self.assertRaisesRegex(
                OutOfDomainError,
                f"Value must be {pd.Series}, instead it is {candidate.__class__}",
            )
            return

        exception = "Found invalid value in Series: *"
        if in_int32_domain:
            self.assertEqual(self.int32_series_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.int32_series_domain.validate(candidate)

        if in_int64_domain:
            self.assertEqual(self.int64_series_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.int64_series_domain.validate(candidate)

        if in_float32_domain:
            self.assertEqual(self.float32_inf_series_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.float32_inf_series_domain.validate(candidate)


class TestPandasDataFrameDomain(TestCase):
    """Tests for class PandasDataFrameDomain.

    Tests :class:`~tmlt.core.domains.pandas_domains.PandasDataFrameDomain`."""

    def setUp(self):
        """Create a test PandasDataFrameDomain"""
        self.pdfd = PandasDataFrameDomain(
            {
                "A": PandasSeriesDomain(NumpyIntegerDomain()),
                "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                "C": PandasSeriesDomain(NumpyStringDomain()),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        schema = {"A": PandasSeriesDomain(NumpyIntegerDomain())}
        domain = PandasDataFrameDomain(schema=schema)
        schema["A"] = NumpyFloatDomain()
        self.assertDictEqual(
            domain.schema, {"A": PandasSeriesDomain(NumpyIntegerDomain())}
        )

    @parameterized.expand(get_all_props(PandasDataFrameDomain))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.pdfd, prop_name)

    def test_bad_init(self):
        """Test that PandasDataFrameDomain raises error when create with wrong type."""
        with self.assertRaises(TypeError):
            PandasDataFrameDomain(schema="incorrect type")

    @parameterized.expand(
        [
            (pd.DataFrame({"A": [1, 2], "B": [1.0, 2.0], "C": ["1", "2"]}), None),
            # wrong column type
            (
                pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": ["1", "2"]}),
                "Found invalid value in column 'B': Found invalid value in Series: "
                f"Value must be {np.float64}, instead it is {np.int64}.",
            ),
            # missing column
            (
                pd.DataFrame({"A": [1, 2], "B": [1.0, 2.0]}),
                "Columns are not as expected. DataFrame and Domain must contain the "
                r"same columns in the same order.\nDataFrame columns: \['A', 'B'\]\n"
                r"Domain columns: \['A', 'B', 'C'\]",
            ),
            # columns out of order
            (
                pd.DataFrame({"A": [1, 2], "C": ["1", "2"], "B": [1.0, 2.0]}),
                "Columns are not as expected. DataFrame and Domain must contain the "
                "same columns in the same order.\nDataFrame columns: "
                r"\['A', 'C', 'B'\]\n"
                r"Domain columns: \['A', 'B', 'C'\]",
            ),
            # wrong type
            (
                pd.Series([1, 2, 3]),
                f"Value must be {pd.DataFrame}, instead it is {pd.Series}.",
            ),
            # duplicated columns
            (
                pd.DataFrame(
                    [["A", "A", "B", 1.1], ["V", "V", "E", 1.2], ["A", "A", "V", 1.3]],
                    columns=["A", "A", "B", "C"],
                ),
                r"Some columns are duplicated, \['A'\]",
            ),
        ]
    )
    def test_validate(self, value: Any, exception: Optional[str]):
        """Test that validate works as expected."""
        if exception is not None:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.pdfd.validate(value)
        else:
            self.assertEqual(self.pdfd.validate(value), exception)

    @parameterized.expand(
        [
            (
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain()),
                        "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                        "C": PandasSeriesDomain(NumpyStringDomain()),
                    }
                ),
                True,
            ),
            # wrong column type
            (
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                        "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                        "C": PandasSeriesDomain(NumpyStringDomain()),
                    }
                ),
                False,
            ),
            # columns out of order
            (
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain()),
                        "C": PandasSeriesDomain(NumpyStringDomain()),
                        "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                    }
                ),
                False,
            ),
        ]
    )
    def test_eq(self, other: Any, is_match: bool):
        """Test that to_spark_domain works as expected."""
        self.assertEqual(self.pdfd == other, is_match)

    def test_repr(self):
        """Tests that __repr__ works correctly"""
        domain = PandasDataFrameDomain({"A": PandasSeriesDomain(NumpyIntegerDomain())})

        expected = (
            "PandasDataFrameDomain(schema={'A':"
            " PandasSeriesDomain(element_domain=NumpyIntegerDomain(size=64))})"
        )
        self.assertEqual(repr(domain), expected)
