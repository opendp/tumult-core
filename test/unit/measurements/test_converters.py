"""Unit tests for :mod:`~tmlt.core.measurements.converters`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from typing import Tuple
from unittest.case import TestCase
from unittest.mock import call

import sympy as sp
from parameterized import parameterized

from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.converters import (
    PureDPToApproxDP,
    PureDPToRhoZCDP,
    RhoZCDPToApproxDP,
)
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import create_mock_measurement


class TestPureDPToZCDP(TestCase):
    """Tests for :class:`~tmlt.core.measurements.converters.PureDPToRhoZCDP`."""

    @parameterized.expand(
        [
            (create_mock_measurement(output_measure=PureDP(), is_interactive=True),),
            (create_mock_measurement(output_measure=ApproxDP(), is_interactive=False),),
            (create_mock_measurement(output_measure=RhoZCDP(), is_interactive=False),),
        ]
    )
    def test_invalid_inner_measurement(self, inner_measure: Measurement):
        """Test that PureDPToRhoZCDP class raises an error for invalid input."""
        with self.assertRaises(ValueError):
            PureDPToRhoZCDP(inner_measure)

    @parameterized.expand(
        [
            (1, "0.5", True, ExactNumber(1)),
            (2, 2, True, ExactNumber(2)),
            (3, "4.5", False, ExactNumber(1)),
        ]
    )
    def test_privacy_function_and_relation(
        self,
        d_in: ExactNumberInput,
        d_out: ExactNumberInput,
        inner_privacy_function_implemented: bool,
        inner_privacy_function_return_value: ExactNumber,
    ):
        """Test that the privacy function and relation are computed correctly."""
        d_in = ExactNumber(d_in)
        d_out = ExactNumber(d_out)
        inner_measurement = create_mock_measurement(
            privacy_function_implemented=inner_privacy_function_implemented,
            privacy_function_return_value=inner_privacy_function_return_value,
        )
        inner_measurement.privacy_relation = lambda d_in, d_out: d_in <= d_out
        measurement = PureDPToRhoZCDP(inner_measurement)
        if not inner_privacy_function_implemented:
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                measurement.privacy_function(d_in)
        else:
            self.assertEqual(measurement.privacy_function(d_in), d_out)
        self.assertTrue(measurement.privacy_relation(d_in, d_out))
        self.assertFalse(
            measurement.privacy_relation(d_in, d_out - sp.Rational("0.00001"))
        )

    def test_call(self):
        """Test that __call__() calls the inner measurement."""
        inner_measurement = create_mock_measurement(return_value=6)
        result = PureDPToRhoZCDP(inner_measurement)(5)
        self.assertEqual(inner_measurement.mock_calls, [call(5)])
        self.assertEqual(result, 6)


class TestPureDPToApproxDP(TestCase):
    """Tests for :class:`PureDPToApproxDP`."""

    @parameterized.expand(
        [
            (create_mock_measurement(output_measure=PureDP(), is_interactive=True),),
            (create_mock_measurement(output_measure=ApproxDP(), is_interactive=False),),
            (create_mock_measurement(output_measure=RhoZCDP(), is_interactive=False),),
        ]
    )
    def test_invalid_inner_measurement(self, inner_measure: Measurement):
        """Test that :class:`PureDPToApproxDP` raises an error for invalid input."""
        with self.assertRaises(ValueError):
            PureDPToApproxDP(inner_measure)

    @parameterized.expand(
        [
            (1, (1, 0), True, ExactNumber(1)),
            (2, (2, 0), True, ExactNumber(2)),
            (1, (1, 0), False, ExactNumber(1)),
        ]
    )
    def test_privacy_function_and_relation(
        self,
        d_in: ExactNumberInput,
        d_out: Tuple[ExactNumberInput, ExactNumberInput],
        inner_privacy_function_implemented: bool,
        inner_privacy_function_return_value: ExactNumber,
    ):
        """Test that the privacy function and relation are computed correctly."""
        inner_measurement = create_mock_measurement(
            privacy_function_implemented=inner_privacy_function_implemented,
            privacy_function_return_value=inner_privacy_function_return_value,
        )
        inner_measurement.privacy_relation = lambda d_in, d_out: bool(d_in <= d_out)
        measurement = PureDPToApproxDP(inner_measurement)
        if not inner_privacy_function_implemented:
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                measurement.privacy_function(d_in)
        else:
            self.assertEqual(measurement.privacy_function(d_in), d_out)
        self.assertTrue(measurement.privacy_relation(d_in, d_out))
        self.assertFalse(
            measurement.privacy_relation(
                d_in, (d_out[0] - sp.Rational("0.00001"), d_out[1])
            )
        )

    def test_call(self):
        """Test that __call__() calls the inner measurement."""
        inner_measurement = create_mock_measurement(return_value=6)
        result = PureDPToRhoZCDP(inner_measurement)(5)
        self.assertEqual(inner_measurement.mock_calls, [call(5)])
        self.assertEqual(result, 6)


class TestRhoZCDPToApproxDP(TestCase):
    """Tests for :class:`RhoZCDPToApproxDP`."""

    @parameterized.expand(
        [
            (create_mock_measurement(output_measure=RhoZCDP(), is_interactive=True),),
            (create_mock_measurement(output_measure=PureDP(), is_interactive=False),),
            (create_mock_measurement(output_measure=ApproxDP(), is_interactive=False),),
        ]
    )
    def test_invalid_inner_measurement(self, inner_measure: Measurement):
        """Test that :class:`RhoZCDPToApproxDP` raises an error for invalid input."""
        with self.assertRaises(ValueError):
            RhoZCDPToApproxDP(inner_measure)

    @parameterized.expand(
        [
            (1, (3, sp.exp(-1))),
            (1, (5, sp.exp(-4))),
            (1, (21, sp.exp(-100))),
            ("0.5", ("4.5", sp.exp(-8))),
            (sp.oo, (sp.oo, 0)),
            # delta = 1 is equivalent to rho = infinity
            (2, (1, 1)),
            (sp.oo, (0, 1)),
        ]
    )
    def test_privacy_relation_valid(
        self, d_in: ExactNumberInput, d_out: Tuple[ExactNumberInput, ExactNumberInput]
    ):
        """Test that the privacy relation is computed correctly."""
        inner_measurement = create_mock_measurement(output_measure=RhoZCDP())
        inner_measurement.privacy_relation = lambda d_in, d_out: ExactNumber(
            d_in
        ) <= ExactNumber(d_out)
        measurement = RhoZCDPToApproxDP(inner_measurement)
        self.assertTrue(measurement.privacy_relation(d_in, d_out))

    @parameterized.expand(
        [
            (1, ("2.99", sp.exp(-1))),
            (1, ("4.99", sp.exp(-4))),
            (1, (20, sp.exp(-100))),
            ("0.5", ("2.499999", sp.exp(-8))),
            (sp.oo, ("4217", "0")),
        ]
    )
    def test_privacy_relation_tight(
        self, d_in: ExactNumberInput, d_out: Tuple[ExactNumberInput, ExactNumberInput]
    ):
        """Test that the privacy relation fails when the conversion shouldn't work."""
        inner_measurement = create_mock_measurement(output_measure=RhoZCDP())
        inner_measurement.privacy_relation = lambda d_in, d_out: ExactNumber(
            d_in
        ) <= ExactNumber(d_out)
        measurement = RhoZCDPToApproxDP(inner_measurement)
        self.assertFalse(measurement.privacy_relation(d_in, d_out))

    def test_call(self):
        """Test that __call__() calls the inner measurement."""
        inner_measurement = create_mock_measurement(return_value=6)
        result = PureDPToRhoZCDP(inner_measurement)(5)
        self.assertEqual(inner_measurement.mock_calls, [call(5)])
        self.assertEqual(result, 6)
