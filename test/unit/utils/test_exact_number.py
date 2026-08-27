"""Unit tests for :mod:`tmlt.core.utils.exact_number`."""

import itertools
from fractions import Fraction
from typing import ClassVar
from unittest import TestCase

import sympy as sp
from parameterized import parameterized

from tmlt.core.exceptions import UnsupportedSympyExprError
from tmlt.core.utils.exact_number import (
    _MEMO_SIZE,
    ExactNumber,
    ExactNumberInput,
    _memoized_to_sympy,
)

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026


class TestExactNumber(TestCase):
    """TestCase for ExactNumber."""

    @parameterized.expand(
        [(3, 3), (-3, 3), (0, 0), (-sp.oo, sp.oo), (sp.oo, sp.oo), ("-31", 31)]
    )
    def test_abs(self, value: ExactNumberInput, expected: ExactNumberInput):
        """__abs__ returns the expected value."""
        self.assertEqual(abs(ExactNumber(value)), ExactNumber(expected))

    @parameterized.expand(
        [
            (3, 1, 3),
            (7, 2, Fraction(7, 2)),
            (2, 7, Fraction(2, 7)),
            (0, 3, 0),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, 0),
            (-Fraction(3, 5), -Fraction(5, 10), "6/5"),
        ]
    )
    def test_division(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__truediv__ and __rtruediv__ return the expected values."""
        self.assertEqual(ExactNumber(value1) / value2, expected)
        self.assertEqual(value1 / ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3, 1, 4),
            (7, 2, 9),
            (2, 7, 9),
            (0, 3, 3),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, -sp.oo),
            (-Fraction(3, 5), -Fraction(5, 10), "-11/10"),
        ]
    )
    def test_addition(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__add__ and __radd__ return the expected values."""
        self.assertEqual(ExactNumber(value1) + value2, expected)
        self.assertEqual(value1 + ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3, 1, 3),
            (7, 2, 14),
            (2, 7, 14),
            (0, 3, 0),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, -sp.oo),
            (-Fraction(3, 5), -Fraction(5, 10), "3/10"),
        ]
    )
    def test_multiplication(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__mul__ and __rmul__ return the expected values."""
        self.assertEqual(ExactNumber(value1) * value2, expected)
        self.assertEqual(value1 * ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3, 1, 2),
            (7, 2, 5),
            (2, 7, -5),
            (0, 3, -3),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, sp.oo),
            (-Fraction(3, 5), -Fraction(5, 10), "-1/10"),
        ]
    )
    def test_subtraction(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__sub__ and __rsub__ return the expected values."""
        self.assertEqual(ExactNumber(value1) - value2, expected)
        self.assertEqual(value1 - ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3, 1, 3),
            (7, 2, 49),
            (2, 7, 128),
            (0, 3, 0),
            (sp.oo, 100, sp.oo),
            (100, -sp.oo, 0),
            (sp.Rational(3, 5), Fraction(1, 2), "sqrt(3/5)"),
        ]
    )
    def test_exponentiation(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: ExactNumberInput,
    ):
        """__pow__ and __rpow__ return the expected values."""
        self.assertEqual(ExactNumber(value1) ** value2, expected)
        self.assertEqual(value1 ** ExactNumber(value2), expected)

    @parameterized.expand(
        [
            (3,),
            (-3,),
            (0,),
            (-sp.oo,),
            (sp.oo,),
            ("-31",),
            ("1234/433",),
            (sp.Rational(10000, 3),),
            (Fraction(123456789, 987654321),),
        ]
    )
    def test_to_float(self, value: ExactNumberInput):
        """to_float returns the expected value."""
        exact_value = ExactNumber(value)
        ceil_value = exact_value.to_float(round_up=True)
        floor_value = exact_value.to_float(round_up=False)
        self.assertTrue(
            float(exact_value.expr) + 0.000001
            >= ceil_value
            >= float(exact_value.expr)
            >= floor_value
            >= float(exact_value.expr) - 0.000001
        )

    @parameterized.expand(
        [
            (3,),
            (-3,),
            (0,),
            (-float("inf"),),
            (float("inf"),),
            (-31,),
            (1234 / 433,),
            (10000 / 3,),
            (123456789 / 987654321,),
        ]
    )
    def test_from_float(self, value: float):
        """from_float returns the expected value."""
        ceil_value = ExactNumber.from_float(value, round_up=True)
        floor_value = ExactNumber.from_float(value, round_up=False)
        self.assertTrue(
            value + 0.000001
            >= float(ceil_value.expr)
            >= value
            >= float(floor_value.expr)
            >= value - 0.000001
        )

    @parameterized.expand(
        itertools.combinations([3, 7, 2, 0, sp.oo, 100, -Fraction(3, 5)], 2)
    )
    def test_comparison(self, value1: ExactNumberInput, value2: ExactNumberInput):
        """from_float returns the expected value."""
        comparisons = [
            lambda a, b: a == b,
            lambda a, b: a < b,
            lambda a, b: a > b,
            lambda a, b: a <= b,
            lambda a, b: a >= b,
        ]
        for compare in comparisons:
            expected = bool(compare(ExactNumber(value1).expr, ExactNumber(value2)))
            self.assertEqual(compare(ExactNumber(value1), value2), expected)
            self.assertEqual(compare(value1, ExactNumber(value2)), expected)


class TestToSympyMemo(TestCase):
    """The memo in front of the conversion changes nothing but its cost."""

    #: One value of every input type, and the ExactNumber each must give.
    CASES: ClassVar = [
        (5, "5"),
        (-1000000000000000, "-1000000000000000"),
        (0, "0"),
        (1, "1"),
        (True, "1"),
        (False, "0"),
        (Fraction(1, 10), "1/10"),
        ("0.5", "1/2"),
        ("2 + 7**2", "51"),
        ("2 * pi**2", "2*pi**2"),
        ("sqrt(5/3)", "sqrt(15)/3"),
        (float("inf"), "oo"),
        (-float("inf"), "-oo"),
        (sp.Integer(7), "7"),
        (sp.Rational(3, 4), "3/4"),
        (sp.oo, "oo"),
    ]

    @parameterized.expand(CASES)
    def test_conversion_is_unchanged(self, value: ExactNumberInput, expected: str):
        """Every input type converts to what it did before, twice running."""
        self.assertEqual(ExactNumber(value), ExactNumber(expected))
        self.assertEqual(ExactNumber(value), ExactNumber(expected))

    def test_a_float_does_not_share_an_integer_s_entry(self):
        """A memo keyed by value alone would answer 1.0 with 1.

        ``hash(1) == hash(1.0) == hash(True)`` and the three compare equal, so
        only memoizing the *exact* types int and str keeps them apart. A float
        that is not an infinity is an error, and has to stay one after an equal
        integer has been converted.
        """
        self.assertEqual(ExactNumber(1), ExactNumber("1"))
        with self.assertRaisesRegex(ValueError, r"Expected \+/-float\('inf'\)"):
            ExactNumber(1.0)
        self.assertEqual(ExactNumber(0), ExactNumber("0"))
        with self.assertRaisesRegex(ValueError, r"Expected \+/-float\('inf'\)"):
            ExactNumber(0.0)

    def test_an_invalid_string_raises_every_time(self):
        """A conversion that fails is not remembered as a success."""
        for _ in range(2):
            with self.assertRaises(UnsupportedSympyExprError):
                ExactNumber("x + 1")

    def test_the_memo_is_bounded(self):
        """Converting many distinct values does not accumulate all of them."""
        _memoized_to_sympy.cache_clear()
        for value in range(_MEMO_SIZE + 100):
            ExactNumber(value)
        self.assertLessEqual(_memoized_to_sympy.cache_info().currsize, _MEMO_SIZE)
