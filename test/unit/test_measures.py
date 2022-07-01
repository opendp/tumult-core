"""Unit tests for :mod:`tmlt.core.measures`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
import itertools
from fractions import Fraction
from typing import Any, Tuple
from unittest.case import TestCase

import sympy as sp
from parameterized import parameterized

from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class TestPureDP(TestCase):
    """TestCase for PureDP."""

    def setUp(self):
        """Setup."""
        self.pureDP = PureDP()

    @parameterized.expand(
        [
            (0,),
            (10,),
            (float("inf"),),
            ("3",),
            ("32",),
            (0,),
            (sp.Integer(1),),
            (sp.Rational("42.17"),),
            (sp.oo,),
        ]
    )
    def test_valid(self, value: ExactNumberInput):
        """Tests for valid values of epsilon."""
        self.pureDP.validate(value)

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(1), True),
            (sp.Rational("42.17"), sp.Rational("42.17"), True),
            (sp.Integer(0), sp.oo, True),
            (sp.oo, sp.oo, True),
            (sp.Integer(1), sp.Integer(0), False),
            (sp.Integer(1), sp.Rational("0.5"), False),
            (sp.oo, sp.Integer(1000), False),
        ]
    )
    def test_compare(
        self, value1: ExactNumberInput, value2: ExactNumberInput, expected: bool
    ):
        """Tests that compare returns the expected result."""
        self.assertEqual(self.pureDP.compare(value1, value2), expected)

    @parameterized.expand([(2.0,), (sp.Float(2),), ("wat",), ({},)])
    def test_invalid(self, val: Any):
        """Only valid ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            self.pureDP.validate(val)

    def test_negative(self):
        """Negative epsilon should be forbidden."""
        with self.assertRaises(ValueError):
            self.pureDP.validate(sp.Integer(-1))


class TestApproxDP(TestCase):
    """TestCase for ApproxDP."""

    def setUp(self):
        """Setup."""
        self.approxDP = ApproxDP()

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(0)),
            (0, "0"),
            (sp.Integer(0), "0.5"),
            ("0", 1),
            (Fraction(1, 2), Fraction(1, 2)),
            (sp.Rational("1.7"), sp.Integer(0)),
            (sp.Rational("1.7"), sp.Integer(1)),
            ("42.17", sp.Integer(0)),
            (sp.oo, sp.Integer(0)),
            (sp.oo, sp.Integer(1)),
        ]
    )
    def test_valid(self, epsilon: ExactNumberInput, delta: ExactNumberInput):
        """Tests for valid values of epsilon and delta."""
        self.approxDP.validate((epsilon, delta))

    @parameterized.expand(
        [
            (
                (epsilon1, delta1),
                (epsilon2, delta2),
                (
                    (ExactNumber(epsilon1) == sp.oo or ExactNumber(delta1) == 1)
                    and (ExactNumber(epsilon2) == sp.oo or ExactNumber(delta2) == 1)
                )
                or (
                    ExactNumber(epsilon1) <= ExactNumber(epsilon2)
                    and ExactNumber(delta1) <= ExactNumber(delta2)
                ),
            )
            for epsilon1, epsilon2 in itertools.combinations(
                ["0", 1, sp.Rational("1.3"), sp.oo], 2
            )
            for delta1, delta2 in itertools.combinations([0, "0.5", sp.Integer(1)], 2)
        ]
    )
    def test_compare(
        self,
        value1: Tuple[ExactNumberInput, ExactNumberInput],
        value2: Tuple[ExactNumberInput, ExactNumberInput],
        expected: bool,
    ):
        """Tests that compare returns the expected result."""
        self.assertEqual(self.approxDP.compare(value1, value2), expected)

    @parameterized.expand(
        [
            ((sp.Integer(1), sp.Integer(1), sp.Integer(1)),),
            ((sp.Integer(1),),),
            ([sp.Integer(1), sp.Float(1e-5)],),  # list, not a pair
        ]
    )
    def test_not_pair(self, val: Any):
        """Input should be a pair."""
        with self.assertRaises(ValueError):
            self.approxDP.validate(val)

    @parameterized.expand(
        [
            (sp.Integer(1), 1e-10),
            (sp.Integer(1), "wat"),
            (17.42, sp.Integer(0)),
            (17, sp.Float((0.1))),
            ({"w", "u", "t"}, sp.Integer(0)),
        ]
    )
    def test_invalid(self, epsilon: Any, delta: Any):
        """Only valid ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            self.approxDP.validate((epsilon, delta))

    def test_negative_epsilon(self):
        """Negative epsilon should be forbidden."""
        with self.assertRaises(ValueError):
            self.approxDP.validate((sp.Integer(-1), sp.Float(1e-10)))

    def test_negative_delta(self):
        """Negative delta should be forbidden."""
        with self.assertRaises(ValueError):
            self.approxDP.validate((sp.Integer(1), sp.Float(-1e-10)))

    @parameterized.expand([(sp.Integer(1), sp.Rational("1.1")), (sp.Integer(1), sp.oo)])
    def test_delta_too_large(self, epsilon: ExactNumberInput, delta: ExactNumberInput):
        """Delta should be smaller than 1."""
        with self.assertRaises(ValueError):
            self.approxDP.validate((epsilon, delta))


class TestRhoZCDP(TestCase):
    """Test cases for RhoZCDP."""

    def setUp(self):
        """Setup."""
        self.rhoZCDP = RhoZCDP()

    @parameterized.expand(
        [
            (0,),
            (10,),
            (float("inf"),),
            ("3",),
            ("32",),
            (sp.Integer(0),),
            (sp.Integer(1),),
            (sp.Rational("42.17"),),
            (sp.oo,),
        ]
    )
    def test_valid(self, expr: ExactNumberInput):
        """Tests for valid values of rho."""
        self.rhoZCDP.validate(expr)

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(1), True),
            (sp.Rational("42.17"), sp.Rational("42.17"), True),
            (sp.Integer(0), sp.oo, True),
            (sp.oo, sp.oo, True),
            (sp.Integer(1), sp.Integer(0), False),
            (sp.Integer(1), sp.Rational("0.5"), False),
            (sp.oo, sp.Integer(1000), False),
        ]
    )
    def test_compare(
        self, value1: ExactNumberInput, value2: ExactNumberInput, expected: bool
    ):
        """Tests that compare returns the expected result."""
        self.assertEqual(self.rhoZCDP.compare(value1, value2), expected)

    @parameterized.expand([(2.0,), ({"w", "a", "t"},), (sp.Float(17.42),)])
    def test_invalid(self, val: Any):
        """Only valid ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            self.rhoZCDP.validate(val)

    def test_negative(self):
        """Negative rho should be forbidden."""
        with self.assertRaises(ValueError):
            self.rhoZCDP.validate(sp.Integer(-1))
