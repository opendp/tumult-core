"""Unit tests for :mod:`~tmlt.core.domains.numpy`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from typing import Any, Optional
from unittest.case import TestCase

import numpy as np
from parameterized import parameterized

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)


class TestNumpyFloatDomain(TestCase):
    """Tests for :class:`~tmlt.core.domains.numpy.NumpyFloatDomain`."""

    def setUp(self):
        """Setup."""
        self.float_domain = NumpyFloatDomain()
        self.float32_domain = NumpyFloatDomain(size=32)
        self.nullable_float_domain = NumpyFloatDomain(allow_nan=True)
        self.float_inf_domain = NumpyFloatDomain(allow_inf=True)
        self.nullable_float_inf_domain = NumpyFloatDomain(
            allow_nan=True, allow_inf=True
        )

    @parameterized.expand([(True, None), ("False", True), (True, 0.1)])
    def test_invalid_inputs(self, allow_inf: bool, allow_nan: bool):
        """Test FloatDomain with invalid inputs."""
        with self.assertRaises(TypeError):
            NumpyFloatDomain(allow_inf, allow_nan)

    @parameterized.expand(
        [
            (NumpyFloatDomain(), True, False, False, False, False),
            (NumpyFloatDomain(size=32), False, True, False, False, False),
            (NumpyFloatDomain(allow_nan=True), False, False, True, False, False),
            (NumpyFloatDomain(allow_inf=True), False, False, False, True, False),
            (
                NumpyFloatDomain(allow_nan=True, allow_inf=True),
                False,
                False,
                False,
                False,
                True,
            ),
        ]
    )
    def test_eq(
        self,
        domain: NumpyFloatDomain,
        float_only: bool,
        float32_only: bool,
        nullable_float: bool,
        float_with_inf: bool,
        nullable_float_inf: bool,
    ):
        """Tests that __eq__  works correctly."""
        self.assertEqual(self.float_domain == domain, float_only)
        self.assertEqual(self.float32_domain == domain, float32_only)
        self.assertEqual(self.nullable_float_domain == domain, nullable_float)
        self.assertEqual(self.float_inf_domain == domain, float_with_inf)
        self.assertEqual(self.nullable_float_inf_domain == domain, nullable_float_inf)

    @parameterized.expand(
        [
            (
                np.float64(1.0),
                True,
                False,
                True,
                True,
                True,
                f"Value must be {np.float32}, instead it is {np.float64}.",
            ),
            (
                np.float32(1.0),
                False,
                True,
                False,
                False,
                False,
                f"Value must be {np.float64}, instead it is {np.float32}.",
            ),
            (
                np.float64(float("inf")),
                False,
                False,
                False,
                True,
                True,
                "Value is infinite.",
            ),
            (
                np.float64(-float("inf")),
                False,
                False,
                False,
                True,
                True,
                "Value is infinite.",
            ),
            (
                np.float64(float("nan")),
                False,
                False,
                True,
                False,
                True,
                "Value is NaN.",
            ),
            (
                1,
                False,
                False,
                False,
                False,
                False,
                f"Value must be {np.float64}, instead it is {int}.",
            ),
        ]
    )
    def test_validate(
        self,
        candidate: Any,
        float_only: bool,
        float32_only: bool,
        float_or_nan: bool,
        float_or_inf: bool,
        float_or_nan_or_inf: bool,
        exception: str,
    ):
        """Tests that validate works correctly."""
        if float_only:
            self.assertEqual(self.float_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.float_domain.validate(candidate)

        if float_or_nan:
            self.assertEqual(self.nullable_float_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.nullable_float_domain.validate(candidate)

        if float_or_inf:
            self.assertEqual(self.float_inf_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.float_inf_domain.validate(candidate)

        if float_or_nan_or_inf:
            self.assertEqual(self.nullable_float_inf_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.nullable_float_inf_domain.validate(candidate)

        if float32_only:
            self.assertEqual(self.float32_domain.validate(candidate), None)
        else:
            if isinstance(candidate, int):
                exception = f"Value must be {np.float32}, instead it is {int}."
            else:
                exception = f"Value must be {np.float32}, instead it is {np.float64}."

            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.float32_domain.validate(candidate)


class TestNumpyIntegerDomain(TestCase):
    """Tests for :class:`~tmlt.core.domains.numpy.NumpyIntegerDomain`."""

    def setUp(self):
        """Setup."""
        self.int64_domain = NumpyIntegerDomain()
        self.int32_domain = NumpyIntegerDomain(size=32)

    @parameterized.expand(
        [
            (NumpyFloatDomain(), False, False),
            (NumpyIntegerDomain(), True, False),
            (NumpyIntegerDomain(size=32), False, True),
        ]
    )
    def test_eq(self, domain: Domain, int64_domain: bool, int32_domain: bool):
        """Tests that __eq__  works correctly."""
        self.assertEqual(self.int64_domain == domain, int64_domain)
        self.assertEqual(self.int32_domain == domain, int32_domain)

    @parameterized.expand(
        [
            (np.int64(1), True, False),
            (np.float64(1.1), False, False),
            (np.int32(3), False, True),
        ]
    )
    def test_validate(self, candidate: Any, is_int64: bool, is_int32: bool):
        """Tests that validate works correctly."""
        if is_int64:
            self.assertEqual(self.int64_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(
                OutOfDomainError,
                f"Value must be {np.int64}, instead it is {candidate.__class__}",
            ):
                self.int64_domain.validate(candidate)

        if is_int32:
            self.assertEqual(self.int32_domain.validate(candidate), None)
        else:
            with self.assertRaisesRegex(
                OutOfDomainError,
                f"Value must be {np.int32}, instead it is {candidate.__class__}",
            ):
                self.int32_domain.validate(candidate)


class TestNumpyStringDomain(TestCase):
    """Tests for :class:`~tmlt.core.domains.numpy.NumpyStringDomain`."""

    @parameterized.expand(
        [
            (None, NumpyStringDomain(allow_null=True), None),
            (None, NumpyStringDomain(allow_null=False), "Value is null."),
            ("ABC", NumpyStringDomain(allow_null=False), None),
            ("ABC", NumpyStringDomain(allow_null=True), None),
        ]
    )
    def test_validate(self, candidate: Any, domain: Domain, exception: Optional[str]):
        """Tests that validate works correctly."""
        if exception is not None:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                domain.validate(candidate)
        else:
            self.assertEqual(domain.validate(candidate), exception)

    @parameterized.expand(
        [
            (
                NumpyStringDomain(allow_null=True),
                NumpyStringDomain(allow_null=True),
                True,
            ),
            (
                NumpyStringDomain(allow_null=True),
                NumpyStringDomain(allow_null=False),
                False,
            ),
            (
                NumpyStringDomain(allow_null=False),
                NumpyStringDomain(allow_null=False),
                True,
            ),
        ]
    )
    def test_eq(self, domain1: Domain, domain2: Domain, expected: bool):
        """Tests that __eq__ works correctly."""
        self.assertEqual(domain1 == domain2, expected)
