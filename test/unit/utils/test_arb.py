"""Test for :mod:`tmlt.core.utils.arb`"""


# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import math
from unittest import TestCase

from parameterized import parameterized
from scipy.special import erf  # pylint: disable=no-name-in-module

from tmlt.core.utils.arb import (
    Arb,
    arb_abs,
    arb_add,
    arb_div,
    arb_erfinv,
    arb_log,
    arb_max,
    arb_min,
    arb_mul,
    arb_sgn,
    arb_sub,
)


class TestArb(TestCase):
    """Tests for :class:`tmlt.core.utils.arb.Arb`."""

    @parameterized.expand(
        [
            (-42 * 10 ** 9,),
            (-42 * 10 ** 7,),
            (-42,),
            (0,),
            (42,),
            (42 * 10 ** 7,),
            (42 * 10 ** 9,),
            (42 * 10 ** 11,),
        ]
    )
    def test_from_int(self, val: int):
        """Tests `Arb.from_int`."""
        x = Arb.from_int(val)
        man, exp = x.man_exp()
        self.assertEqual(man * 2 ** exp, val)

    @parameterized.expand([(0.0,), (1.1,), (-1.1,), (9.9 * 0.123,), (99.12,)])
    def test_from_float(self, val: float):
        """Tests `Arb.from_float`."""
        x = Arb.from_float(val)
        man, exp = x.man_exp()
        self.assertEqual(man * 2 ** exp, val)

    def test_from_float_inf(self):
        """Tests `Arb.from_float` works with +/- inf."""
        posinf = Arb.from_float(float("inf"))
        neginf = Arb.from_float(float("-inf"))

        self.assertFalse(posinf.is_finite())
        self.assertFalse(neginf.is_finite())
        self.assertEqual(float(posinf), float("inf"))
        self.assertEqual(float(neginf), float("-inf"))

    @parameterized.expand([(2, 30), (4, 300), (5 * 10 ** 2, 7 ** 8)])
    def test_from_man_exp(self, man: int, exp: int):
        """Tests `Arb.from_man_exp`."""
        x = Arb.from_man_exp(man, exp)
        m, e = x.man_exp()
        self.assertEqual(m * 2 ** e, man * 2 ** exp)

    @parameterized.expand([(10, 1), (10000, 5), (10, 1), (10, 1)])
    def test_from_midpoint_radius(self, mid: int, rad: int):
        """Tests `Arb.from_midpoint_radius`."""
        mid_arb = Arb.from_int(mid)
        rad_arb = Arb.from_int(rad)
        x = Arb.from_midpoint_radius(mid_arb, rad_arb)
        assert x.midpoint() == mid_arb
        actual_radius = float(x.radius())
        self.assertAlmostEqual(actual_radius, rad)

    @parameterized.expand(
        [
            (Arb.from_int(10), True),
            (Arb.from_float(0.01), True),
            (Arb.from_float(-float("inf")), True),
            (Arb.from_midpoint_radius(1, 0), True),
            (Arb.from_midpoint_radius(1, 1), False),
        ]
    )
    def test_is_exact(self, arb: Arb, exact: bool):
        """Tests `Arb.is_exact`."""
        self.assertEqual(arb.is_exact(), exact)

    def test_is_finite(self):
        """Tests `Arb.is_finite`."""
        self.assertFalse(Arb.from_float(-float("inf")).is_finite())
        self.assertFalse(Arb.from_float(float("inf")).is_finite())
        self.assertTrue(Arb.from_int(10).is_finite())

    def test_is_nan(self):
        """Tests `Arb.is_nan`."""
        self.assertTrue(Arb.from_float(float("nan")).is_nan())
        self.assertFalse(Arb.from_float(0.0).is_nan())

    def test_lower(self):
        """Tests `Arb.lower`."""
        arb = Arb.from_midpoint_radius(1, 0.5)
        self.assertAlmostEqual(float(arb.lower(prec=100)), 0.5)

    def test_upper(self):
        """Tests `Arb.upper`."""
        arb = Arb.from_midpoint_radius(1, 0.5)
        self.assertAlmostEqual(float(arb.upper(prec=100)), 1.5)

    @parameterized.expand(
        [
            (
                Arb.from_midpoint_radius(mid=9, rad=1),
                Arb.from_midpoint_radius(mid=10, rad=2),
                True,
            ),
            (
                Arb.from_midpoint_radius(mid=10, rad=2),
                Arb.from_midpoint_radius(mid=9, rad=1),
                False,
            ),
            (Arb.from_int(10), Arb.from_midpoint_radius(mid=9, rad=1), True),
            (Arb.from_float(10.1), Arb.from_midpoint_radius(mid=9, rad=1), False),
        ]
    )
    def test_contains(self, x: Arb, y: Arb, expected: bool):
        """`y.__contains__(x)` returns True iff every number in `x` is also in `y`."""
        self.assertEqual(x in y, expected)


class TestArbFunctions(TestCase):
    """Tests for arithmetic functions in :mod:`tmlt.core.utils.arb`.

    NOTE: Correctness of an arb function `F` is specified as follows:

        If `f` is the corresponding real-valued arithmetic function, `F` is correct
        only if, for any Arb X and any real number x in the interval X,
        `f(x)` is in `F(X)`.

    These tests assume Arb.__contains__ is correct.
    """

    def test_arb_sub(self):
        """`arb_sub` works as expected."""
        arb1 = Arb.from_midpoint_radius(2, 0.5)
        arb2 = Arb.from_midpoint_radius(1, 1)
        actual = arb_sub(arb1, arb2, prec=100)
        # Smallest value in diff => 1.5 - 2 = -0.5
        # Largest value in diff => 2.5 - 0 = 2.5
        true_interval = Arb.from_midpoint_radius(1, 1.5)  # [-0.5, 2.5]
        self.assertTrue(true_interval in actual)

    def test_arb_add(self):
        """`arb_add` works as expected."""
        arb1 = Arb.from_midpoint_radius(2, 1)
        arb2 = Arb.from_midpoint_radius(1, 1)
        actual = arb_add(arb1, arb2, prec=100)
        true_interval = Arb.from_midpoint_radius(3, 2)  # [1, 5]
        self.assertTrue(true_interval in actual)

    def test_arb_mul(self):
        """`arb_mul` works as expected."""
        arb1 = Arb.from_midpoint_radius(2, 1)
        arb2 = Arb.from_midpoint_radius(1, 1)
        actual = arb_mul(arb1, arb2, prec=100)
        true_interval = Arb.from_midpoint_radius(3, 3)  # [0, 6]
        self.assertTrue(true_interval in actual)

    def test_arb_div(self):
        """`arb_div` works as expected."""
        arb1 = Arb.from_midpoint_radius(4, 1)
        arb2 = Arb.from_midpoint_radius(2, 1)
        actual = arb_div(arb1, arb2, prec=100)
        true_interval = Arb.from_midpoint_radius(4, 1)  # [3, 5]
        self.assertTrue(true_interval in actual)

    def test_arb_log(self):
        """`arb_log` works as expected."""
        midpoint = (1 + math.exp(10)) / 2
        arb = Arb.from_midpoint_radius(midpoint, midpoint - 1)  # [1, exp(10)]
        actual = arb_log(arb, prec=100)
        true_interval = Arb.from_midpoint_radius(5, 5)  # [0,10]
        self.assertTrue(true_interval in actual)

    def test_arb_max(self):
        """`arb_max` works as expected."""
        arb1 = Arb.from_midpoint_radius(1.5, 0.5)  # [1, 2]
        arb2 = Arb.from_midpoint_radius(1, 2)  # [-1, 3]
        actual = arb_max(arb1, arb2, prec=100)
        true_interval = Arb.from_midpoint_radius(2, 1)  # [1, 3]
        self.assertTrue(true_interval in actual)

    def test_arb_min(self):
        """`arb_min` works as expected."""
        arb1 = Arb.from_midpoint_radius(1.5, 0.5)  # [1, 2]
        arb2 = Arb.from_midpoint_radius(1, 2)  # [-1, 3]
        actual = arb_min(arb1, arb2, prec=100)
        true_interval = Arb.from_midpoint_radius(0.5, 1.5)  # [-1, 2]
        self.assertTrue(true_interval in actual)

    def test_arb_abs(self):
        """`arb_abs` works as expected."""
        arb = Arb.from_midpoint_radius(1, 2)  # [-1,3]
        actual = arb_abs(arb)
        true_interval = Arb.from_midpoint_radius(1.5, 1.5)
        self.assertTrue(true_interval in actual)

    def test_arb_sgn(self):
        """`arb_sgn` works as expected."""
        arb1 = Arb.from_midpoint_radius(1, 0.5)  # [0.5,1.5]
        arb2 = Arb.from_midpoint_radius(-1, 0.5)  # [-1.5,-0.5]
        arb3 = Arb.from_midpoint_radius(1, 2)  # [-1,3]
        self.assertAlmostEqual(arb_sgn(arb1).to_float(), 1)
        self.assertAlmostEqual(arb_sgn(arb2).to_float(), -1)
        # arb3 contains both positive and negative numbers
        # So, arb_sgn returns [0, 1]
        self.assertAlmostEqual(float(arb_sgn(arb3).midpoint()), 0)
        self.assertAlmostEqual(float(arb_sgn(arb3).radius()), 1)

    def test_erfinv(self):
        """`erfinv` works as expected."""
        midpoint = (erf(1 / 8) + erf(1 / 16)) / 2
        radius = midpoint - erf(1 / 16)
        arb = Arb.from_midpoint_radius(midpoint, radius)
        actual = arb_erfinv(arb, prec=100)
        true_interval = Arb.from_midpoint_radius(3 / 32, 1 / 32)  # [1/16, 1/8]
        self.assertTrue(true_interval in actual)
