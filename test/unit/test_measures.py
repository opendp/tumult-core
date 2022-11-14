"""Unit tests for :mod:`tmlt.core.measures`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
import itertools
from typing import Any, Tuple
from unittest.case import TestCase

import sympy as sp
from parameterized import parameterized

from tmlt.core.measures import (
    ApproxDP,
    ApproxDPBudget,
    PureDP,
    PureDPBudget,
    RhoZCDP,
    RhoZCDPBudget,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput

VALID_PRIMARY_BUDGET_INPUTS = [
    0,
    10,
    float("inf"),
    "3",
    "32",
    sp.Integer(0),
    sp.Integer(1),
    sp.Rational("42.17"),
    sp.oo,
]
INVALID_PRIMARY_BUDGET_INPUTS = [-1, 2.0, sp.Float(2), "wat", {}]
VALID_DELTA_INPUTS = [0, sp.Integer(1), "1", sp.Rational("0.5")]
INVALID_DELTA_INPUTS = [-1, 0.5, 2, sp.Float(1), "wat", {}]
VALID_APPROX_DP_INPUTS = list(
    itertools.product(VALID_PRIMARY_BUDGET_INPUTS, VALID_DELTA_INPUTS)
)
INVALID_APPROX_DP_INPUTS = (
    list(itertools.product(VALID_PRIMARY_BUDGET_INPUTS, INVALID_DELTA_INPUTS))
    + list(itertools.product(INVALID_PRIMARY_BUDGET_INPUTS, VALID_DELTA_INPUTS))
    + VALID_PRIMARY_BUDGET_INPUTS
    + [(1, 1, 1), [1, 1]]
)

to_singletons = lambda x: map(lambda y: (y,), x)


class TestPureDP(TestCase):
    """TestCase for PureDP."""

    def setUp(self):
        """Setup."""
        self.pureDP = PureDP()

    @parameterized.expand(to_singletons(VALID_PRIMARY_BUDGET_INPUTS))
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

    @parameterized.expand(to_singletons(INVALID_PRIMARY_BUDGET_INPUTS))
    def test_invalid(self, val: Any):
        """Only valid ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            self.pureDP.validate(val)


class TestApproxDP(TestCase):
    """TestCase for ApproxDP."""

    def setUp(self):
        """Setup."""
        self.approxDP = ApproxDP()

    @parameterized.expand(to_singletons(VALID_APPROX_DP_INPUTS))
    def test_valid(self, value: Any):
        """Tests for valid values of epsilon and delta."""
        self.approxDP.validate(value)

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

    @parameterized.expand(to_singletons(INVALID_APPROX_DP_INPUTS))
    def test_invalid(self, value: Any):
        """Only valid budgets should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            self.approxDP.validate(value)


class TestRhoZCDP(TestCase):
    """Test cases for RhoZCDP."""

    def setUp(self):
        """Setup."""
        self.rhoZCDP = RhoZCDP()

    @parameterized.expand(to_singletons(VALID_PRIMARY_BUDGET_INPUTS))
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

    @parameterized.expand(to_singletons(INVALID_PRIMARY_BUDGET_INPUTS))
    def test_invalid(self, val: Any):
        """Only valid budgets should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            self.rhoZCDP.validate(val)


class TestPureDPBudget(TestCase):
    """Test cases for PureDPBudget."""

    @parameterized.expand(to_singletons(VALID_PRIMARY_BUDGET_INPUTS))
    def test_init_valid(self, value):  # pylint: disable=no-self-use
        """Tests that the budget can be created with a valid budget input."""
        PureDPBudget(value)

    @parameterized.expand(to_singletons(INVALID_PRIMARY_BUDGET_INPUTS))
    def test_init_invalid(self, value):
        """Throw an error when created with an invalid budget."""
        with self.assertRaises((TypeError, ValueError)):
            PureDPBudget(value)

    @parameterized.expand(
        [
            (1, True),
            (sp.Rational("76.4"), True),
            (sp.Integer(5), True),
            (sp.oo, False),
            (float("inf"), False),
        ]
    )
    def test_is_finite(self, value, expected: bool):
        """Test that is_finite returns the expected value."""
        self.assertEqual(PureDPBudget(value).is_finite(), expected)

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(1), False),
            (sp.Rational("42.17"), sp.Rational("42.17"), True),
            (sp.Integer(0), sp.oo, False),
            (sp.oo, sp.oo, True),
            (sp.Integer(1), sp.Integer(0), True),
            (sp.Integer(1), sp.Rational("0.5"), True),
            (sp.oo, sp.Integer(1000), True),
        ]
    )
    def test_can_spend_budget(self, value1, value2, expected: bool):
        """Test that can_spend_budget returns the expected value."""
        self.assertEqual(PureDPBudget(value1).can_spend_budget(value2), expected)

    @parameterized.expand(to_singletons(INVALID_PRIMARY_BUDGET_INPUTS))
    def test_can_spend_budget_invalid(self, value):
        """Test that can_spend_budget raises an error when passed an invalid value."""
        with self.assertRaises((TypeError, ValueError)):
            PureDPBudget(1).can_spend_budget(value)

    @parameterized.expand(
        [
            (5, "3", 2),
            (sp.Integer(5), sp.Rational("1.5"), sp.Rational("3.5")),
            (sp.oo, 10, sp.oo),
            (sp.oo, sp.oo, sp.oo),
            ("4", sp.Rational("0"), 4),
            ("4", sp.Integer(4), 0),
        ]
    )
    def test_subtract(self, value1, value2, expected):
        """Test that subtract returns the expected value."""
        self.assertEqual(PureDPBudget(value1).subtract(value2), PureDPBudget(expected))

    @parameterized.expand(
        list(to_singletons(INVALID_PRIMARY_BUDGET_INPUTS)) + [(sp.Rational("1.1"),)]
    )
    def test_subtract_invalid(self, value):
        """Test that subtract raises an error when passed an invalid value."""
        with self.assertRaises((TypeError, ValueError)):
            PureDPBudget(1).subtract(value)


class TestApproxDPBudget(TestCase):
    """Test cases for ApproxDPBudget."""

    # @parameterized.expand(to_singletons(VALID_APPROX_DP_INPUTS))
    # def test_init_valid(self, value):
    #     """Tests that the budget can be created with a valid budget input."""
    #     ApproxDPBudget(value)

    @parameterized.expand(to_singletons(INVALID_APPROX_DP_INPUTS))
    def test_init_invalid(self, value):
        """Throw an error when created with an invalid budget."""
        with self.assertRaises((TypeError, ValueError)):
            ApproxDPBudget(value)

    @parameterized.expand(
        [
            ((1, 0), True),
            ((sp.Rational("76.4"), sp.Rational("0.5")), True),
            ((sp.Integer(5), sp.Rational("0.3")), True),
            ((sp.oo, 0), False),
            ((float("inf"), sp.Rational("0.5")), False),
            ((1, 1), False),
            ((float("inf"), 1), False),
        ]
    )
    def test_is_finite(self, value, expected: bool):
        """Test that is_finite returns the expected value."""
        self.assertEqual(ApproxDPBudget(value).is_finite(), expected)

    @parameterized.expand(
        [
            ((5, sp.Rational("0.1")), ("6", sp.Rational("0.2")), False),
            (("7", sp.Rational("0.1")), (sp.Integer(6), sp.Rational("0.2")), False),
            (
                (sp.Rational("4.5"), sp.Rational("0.2")),
                (sp.Integer(6), sp.Rational("0.1")),
                False,
            ),
            (("6", sp.Rational("0.1")), (sp.Integer(6), sp.Rational("0.1")), True),
            ((7, sp.Rational("0.1")), (sp.Integer(6), sp.Rational("0.1")), True),
            ((6, sp.Rational("0.2")), (sp.Integer(6), sp.Rational("0.1")), True),
            (
                (sp.Rational("8.0"), sp.Rational("0.3")),
                (sp.Integer(6), sp.Rational("0.1")),
                True,
            ),
            ((6, 1), (6, 1), True),
            ((sp.oo, sp.Rational("0.1")), (sp.oo, sp.Rational("0.1")), True),
            ((sp.oo, 1), (sp.oo, 1), True),
        ]
    )
    def test_can_spend_budget(self, value1, value2, expected: bool):
        """Test that can_spend_budget returns the expected value."""
        self.assertEqual(ApproxDPBudget(value1).can_spend_budget(value2), expected)

    @parameterized.expand(to_singletons(INVALID_APPROX_DP_INPUTS))
    def test_can_spend_budget_invalid(self, value):
        """Test that can_spend_budget raises an error when passed an invalid value."""
        with self.assertRaises((TypeError, ValueError)):
            ApproxDPBudget((1, sp.Rational("0.1"))).can_spend_budget(value)

    @parameterized.expand(
        [
            (
                (sp.Rational("5.5"), sp.Rational("0.5")),
                (sp.Integer(2), sp.Rational("0.3")),
                (sp.Rational("3.5"), sp.Rational("0.2")),
            ),
            (
                (sp.Rational("5.5"), sp.Rational("0.5")),
                (0, 0),
                (sp.Rational("5.5"), sp.Rational("0.5")),
            ),
            (("1", sp.Rational("0.1")), ("1", sp.Rational("0.1")), (0, 0)),
            (
                # when subtracting from an infinite budget,
                # the initial budget is returned
                (sp.Rational("1.5"), "1"),
                ("3", sp.Rational("0.9")),
                (sp.Rational("1.5"), "1"),
            ),
            (
                # when subtracting from an infinite budget,
                # the initial budget is returned
                (float("inf"), 0),
                ("1", sp.Rational("0.9")),
                (float("inf"), 0),
            ),
        ]
    )
    def test_subtract(self, value1, value2, expected):
        """Test that subtract returns the expected value."""
        self.assertEqual(
            ApproxDPBudget(value1).subtract(value2), ApproxDPBudget(expected)
        )

    @parameterized.expand(
        list(to_singletons(INVALID_APPROX_DP_INPUTS))
        + [((sp.Rational("1.1"), 0),), (("1", sp.Rational("0.2")),)]
    )
    def test_subtract_invalid(self, value):
        """Test that subtract raises an error when passed an invalid value."""
        with self.assertRaises((TypeError, ValueError)):
            ApproxDPBudget((1, sp.Rational("0.1"))).subtract(value)


class TestRhoZCDPBudget(TestCase):
    """Test cases for RhoZCDPBudget."""

    @parameterized.expand(to_singletons(VALID_PRIMARY_BUDGET_INPUTS))
    def test_init_valid(self, value):  # pylint: disable=no-self-use
        """Tests that the budget can be created with a valid budget input."""
        RhoZCDPBudget(value)

    @parameterized.expand(to_singletons(INVALID_PRIMARY_BUDGET_INPUTS))
    def test_init_invalid(self, value):
        """Throw an error when created with an invalid budget."""
        with self.assertRaises((TypeError, ValueError)):
            RhoZCDPBudget(value)

    @parameterized.expand(
        [
            (1, True),
            (sp.Rational("76.4"), True),
            (sp.Integer(5), True),
            (sp.oo, False),
            (float("inf"), False),
        ]
    )
    def test_is_finite(self, value, expected: bool):
        """Test that is_finite returns the expected value."""
        self.assertEqual(RhoZCDPBudget(value).is_finite(), expected)

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(1), False),
            (sp.Rational("42.17"), sp.Rational("42.17"), True),
            (sp.Integer(0), sp.oo, False),
            (sp.oo, sp.oo, True),
            (sp.Integer(1), sp.Integer(0), True),
            (sp.Integer(1), sp.Rational("0.5"), True),
            (sp.oo, sp.Integer(1000), True),
        ]
    )
    def test_can_spend_budget(self, value1, value2, expected: bool):
        """Test that can_spend_budget returns the expected value."""
        self.assertEqual(RhoZCDPBudget(value1).can_spend_budget(value2), expected)

    @parameterized.expand(to_singletons(INVALID_PRIMARY_BUDGET_INPUTS))
    def test_can_spend_budget_invalid(self, value):
        """Test that can_spend_budget raises an error when passed an invalid value."""
        with self.assertRaises((TypeError, ValueError)):
            RhoZCDPBudget(1).can_spend_budget(value)

    @parameterized.expand(
        [
            (5, "3", 2),
            (sp.Integer(5), sp.Rational("1.5"), sp.Rational("3.5")),
            (sp.oo, 10, sp.oo),
            (sp.oo, sp.oo, sp.oo),
            ("4", sp.Rational("0"), 4),
            ("4", sp.Integer(4), 0),
        ]
    )
    def test_subtract(self, value1, value2, expected):
        """Test that subtract returns the expected value."""
        self.assertEqual(
            RhoZCDPBudget(value1).subtract(value2), RhoZCDPBudget(expected)
        )

    @parameterized.expand(
        list(to_singletons(INVALID_PRIMARY_BUDGET_INPUTS)) + [(sp.Rational("1.1"),)]
    )
    def test_subtract_invalid(self, value):
        """Test that subtract raises an error when passed an invalid value."""
        with self.assertRaises((TypeError, ValueError)):
            RhoZCDPBudget(1).subtract(value)
