"""Unit tests for :mod:`~tmlt.core.domains.collections`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from typing import Any, Dict, Optional
from unittest.case import TestCase
from unittest.mock import Mock, create_autospec

import numpy as np
from parameterized import parameterized
from pyspark.sql.types import StringType

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class TestListDomain(TestCase):
    """Tests for :class:`~tmlt.core.domains.collections.ListDomain`."""

    def setUp(self):
        """Setup."""
        self.list_domain = ListDomain(NumpyIntegerDomain())

    @parameterized.expand(
        [
            (NumpyFloatDomain(), False),
            (ListDomain(NumpyFloatDomain()), False),
            (ListDomain(NumpyIntegerDomain()), True),
        ]
    )
    def test_eq(self, domain: Domain, equal_domain: bool):
        """Tests that __eq__  works correctly."""
        self.assertEqual(self.list_domain == domain, equal_domain)

    @parameterized.expand([(None,), (StringType,)])
    def test_invalid_inputs(self, element_domain: Domain):
        """Test ListDomain with invalid input."""
        with self.assertRaises(TypeError):
            ListDomain(element_domain)

    @parameterized.expand(
        [
            ([np.int64(1)], None),
            ("Not a list", f"Value must be {list}, instead it is {str}."),
            (
                ["invalid"],
                f"Found invalid value in list: Value must be {np.int64}, "
                f"instead it is {str}.",
            ),
        ]
    )
    def test_validate(self, candidate: Any, exception: Optional[str]):
        """Tests that validate works correctly."""
        if exception is not None:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.list_domain.validate(candidate)
        else:
            self.assertEqual(self.list_domain.validate(candidate), exception)


class TestDictDomain(TestCase):
    """Tests for :class:`~tmlt.core.domains.collections.DictDomain`."""

    def setUp(self):
        """Setup."""
        self.domain_a = create_autospec(spec=Domain, instance=True)
        self.domain_b = create_autospec(spec=Domain, instance=True)
        self.dict_domain = DictDomain({"A": self.domain_a, "B": self.domain_b})

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        domain_map = {"A": NumpyIntegerDomain()}
        domain = DictDomain(key_to_domain=domain_map)
        domain_map["A"] = NumpyFloatDomain()
        self.assertDictEqual(domain.key_to_domain, {"A": NumpyIntegerDomain()})

    @parameterized.expand(get_all_props(DictDomain))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.dict_domain, prop_name)

    @parameterized.expand(
        [
            ({"A": Mock(), "B": Mock()}, True, True),
            ({"A": Mock(), "B": Mock()}, True, False),
            ({"A": Mock(), "B": Mock()}, False, True),
            ({"A": Mock(), "B": Mock()}, False, False),
            ({"C": Mock(), "B": Mock()}, True, True),
            ({"A": Mock(), "B": Mock(), "C": Mock()}, True, True),
            ({"A": Mock()}, True, True),
        ]
    )
    def test_validate(self, candidate: Dict[str, Any], in_A: bool, in_B: bool):
        """Tests that validate works correctly."""
        self.domain_a.validate = (
            Mock(side_effect=OutOfDomainError("Test"))
            if not in_A
            else Mock(return_value=None)
        )
        self.domain_b.validate = (
            Mock(side_effect=OutOfDomainError("Test"))
            if not in_B
            else Mock(return_value=None)
        )

        if (in_A and in_B) and set(candidate) == {"A", "B"}:
            self.dict_domain.validate(candidate)
        else:
            issue_object = "'B'" if in_A else "'A'"
            exception = f"Found invalid value at {issue_object}: Test"
            if set(candidate) != {"A", "B"}:
                exception = (
                    "Keys are not as expected, value must match domain.\n"
                    fr"Value keys: \[{str(sorted(set(candidate)))[1:-1]}\]"
                    "\n"
                    r"Domain keys: \['A', 'B'\]"
                )
            with self.assertRaisesRegex(OutOfDomainError, exception):
                self.dict_domain.validate(candidate)

        if set(candidate) == {"A", "B"}:
            self.domain_a.validate.assert_called_once_with(candidate["A"])
            if in_A:
                self.domain_b.validate.assert_called_once_with(candidate["B"])

    @parameterized.expand(
        [
            (DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()}), True),
            (DictDomain({"B": NumpyFloatDomain(), "A": NumpyIntegerDomain()}), True),
            (
                DictDomain(
                    {"A": NumpyIntegerDomain(), "B": NumpyFloatDomain(allow_nan=True)}
                ),
                False,
            ),
            (DictDomain({"A": NumpyIntegerDomain(), "C": NumpyFloatDomain()}), False),
            (
                DictDomain(
                    {
                        "A": NumpyIntegerDomain(),
                        "B": NumpyFloatDomain(),
                        "C": NumpyFloatDomain(),
                    }
                ),
                False,
            ),
            (DictDomain({"A": NumpyIntegerDomain()}), False),
            (NumpyIntegerDomain(), False),
        ]
    )
    def test_eq(self, candidate: Domain, expected: bool):
        """Tests that __eq__ works correctly."""
        domain = DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()})
        self.assertEqual(domain == candidate, expected)

    def test_repr(self):
        """Tests that __repr__ works correctly"""
        domain = DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()})

        expected = (
            "DictDomain(key_to_domain={'A': NumpyIntegerDomain(size=64), "
            "'B': NumpyFloatDomain(allow_nan=False, allow_inf=False, size=64)})"
        )
        self.assertEqual(repr(domain), expected)
