"""Tests for :mod:`~tmlt.core.util.type_utils`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Sequence, Type
from unittest import TestCase

from parameterized import parameterized

from tmlt.core.utils.type_utils import get_element_type


class TestGetElementType(TestCase):
    """Tests from :meth:`~tmlt.core.util.type_utils.get_element_type`."""

    @parameterized.expand(
        [
            ([0, 1, 2], True, int),
            ([0, 1, 2], False, int),
            ([0.0, 1.0, 2.0], True, float),
            ([0.0, 1.0, 2.0], False, float),
            ([0, None, 2], True, int),
            ([None, 1, 2], True, int),
            ([None, None, 2], True, int),
            ([None, None, None], True, type(None)),
        ]
    )
    def test_get_element_type(
        self, l: Sequence[Any], allow_none: bool, expected_type: Type
    ):
        """get_element_type works as expected."""
        self.assertEqual(get_element_type(l, allow_none=allow_none), expected_type)

    @parameterized.expand(
        [
            ([0, 1.0, 2], True, "list contains elements of multiple types"),
            ([None, 1.0, 2], True, "list contains elements of multiple types"),
            ([None, 1.0, 2], False, "None is not allowed"),
            ([0, None, 2], False, "None is not allowed"),
            ([None], False, "None is not allowed"),
            ([], False, "cannot determine element type of empty list"),
            ([], True, "cannot determine element type of empty list"),
        ]
    )
    def test_get_element_type_error(
        self, l: Sequence[Any], allow_none: bool, expected_error: str
    ):
        """get_element_type works as expected."""
        with self.assertRaisesRegex(ValueError, expected_error):
            get_element_type(l, allow_none=allow_none)
