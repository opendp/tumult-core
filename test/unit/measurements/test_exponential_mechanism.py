"""Unit tests for :mod:`~tmlt.core.measurements.exponential_mechanism`."""

# <placeholder: boilerplate>

# pylint: disable=no-self-use

from unittest.case import TestCase

import sympy as sp
from parameterized import parameterized

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain
from tmlt.core.measurements.exponential_mechanism import (
    ExponentialMechanism,
    PermuteAndFlip,
)
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference, DictMetric
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class TestExponentialMechanism(TestCase):
    """Tests for class ExponentialMechanism.

    Tests :class:`~tmlt.core.measurements.exponential_mechanism.ExponentialMechanism`.
    """

    @parameterized.expand(get_all_props(ExponentialMechanism))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        candidates = ["A", "B", "C", "D"]
        measurement = ExponentialMechanism(
            output_measure=PureDP(), candidates=candidates, epsilon=10000000
        )
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """Test that properties have the expected values."""
        candidates = ["A", "B", "C", "D"]
        measurement = ExponentialMechanism(
            output_measure=PureDP(), candidates=candidates, epsilon=10000000
        )

        expected_input_domain = DictDomain(
            {key: NumpyFloatDomain() for key in candidates}
        )
        expected_input_metric = DictMetric(
            {key: AbsoluteDifference() for key in candidates}
        )

        self.assertEqual(measurement.input_domain, expected_input_domain)
        self.assertEqual(measurement.input_metric, expected_input_metric)
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.epsilon, 10000000)

    def test_correctness(self):
        """Tests that the mechanism is correct when epsilon is infinity."""
        candidates = [("A", 1), "B", 3, "D"]
        quality_scores = {("A", 1): 10.0, "B": 3.5, 3: -5.4, "D": 0.6}
        measurement = ExponentialMechanism(
            output_measure=PureDP(), candidates=candidates, epsilon=sp.oo
        )
        private_max = measurement(quality_scores)
        true_max = max(quality_scores, key=lambda r: quality_scores[r])

        self.assertEqual(
            measurement.privacy_function({("A", 1): 1, "B": 1, 3: 1, "D": 1}), sp.oo
        )
        self.assertTrue(private_max == true_max)

    def test_zero_epsilon(self):
        """Works with zero epsilon."""
        candidates = [1, 2, 3, 4]
        quality_scores = {1: 10.0, 2: 3.5, 3: -5.4, 4: 0.6}
        measurement = ExponentialMechanism(
            output_measure=PureDP(), candidates=candidates, epsilon=0
        )
        self.assertEqual(measurement.privacy_function({1: 1, 2: 1, 3: 1, 4: 1}), 0)
        self.assertTrue(measurement(quality_scores) in candidates)

    def test_privacy_function_and_relation(self):
        """Test that the privacy function and relation are computed correctly."""
        candidates = ["A", "B", "C", "D"]
        measurement = ExponentialMechanism(
            output_measure=PureDP(), candidates=candidates, epsilon=2
        )

        d_in = {"A": 1, "B": 1, "C": 1, "D": 1}
        self.assertEqual(measurement.privacy_function(d_in), 2)
        self.assertTrue(measurement.privacy_relation(d_in, 2))
        self.assertFalse(measurement.privacy_relation(d_in, sp.Rational("1.99")))

        d_in = {"A": 1, "B": 3, "C": 1, "D": 1}
        self.assertEqual(measurement.privacy_function(d_in), 6)
        self.assertTrue(measurement.privacy_relation(d_in, 6))
        self.assertFalse(measurement.privacy_relation(d_in, 2))

        measurement = ExponentialMechanism(
            output_measure=RhoZCDP(), candidates=candidates, epsilon=1
        )
        d_in = {"A": 1, "B": 1, "C": 1, "D": 1}
        self.assertEqual(measurement.privacy_function(d_in), "1/8")
        self.assertTrue(measurement.privacy_relation(d_in, "1/8"))
        self.assertFalse(measurement.privacy_relation(d_in, sp.Rational("0.1249")))


class TestPermuteAndFlip(TestCase):
    """Tests for class PermuteAndFlip.

    Tests :class:`~tmlt.core.measurements.exponential_mechanism.PermuteAndFlip`.
    """

    @parameterized.expand(get_all_props(PermuteAndFlip))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        candidates = ["A", "B", "C", "D"]
        measurement = PermuteAndFlip(candidates=candidates, epsilon=10000000)
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """Test that properties have the expected values."""
        candidates = ["A", "B", "C", "D"]
        measurement = PermuteAndFlip(candidates=candidates, epsilon=10000000)

        expected_input_domain = DictDomain(
            {key: NumpyFloatDomain() for key in candidates}
        )
        expected_input_metric = DictMetric(
            {key: AbsoluteDifference() for key in candidates}
        )

        self.assertEqual(measurement.input_domain, expected_input_domain)
        self.assertEqual(measurement.input_metric, expected_input_metric)
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.epsilon, 10000000)

    def test_correctness(self):
        """Tests that the mechanism is correct when epsilon is infinity."""
        candidates = [("A", 1), "B", "C", "D"]
        quality_scores = {("A", 1): 10.0, "B": 3.5, "C": -5.4, "D": 0.6}
        measurement = PermuteAndFlip(candidates=candidates, epsilon=sp.oo)
        private_max = measurement(quality_scores)
        true_max = max(quality_scores, key=lambda r: quality_scores[r])

        self.assertEqual(
            measurement.privacy_function({("A", 1): 1, "B": 1, "C": 1, "D": 1}), sp.oo
        )
        self.assertTrue(private_max == true_max)

    def test_zero_epsilon(self):
        """Works with zero epsilon."""
        candidates = [1, 2, 3, 4]
        quality_scores = {1: 10.0, 2: 3.5, 3: -5.4, 4: 0.6}
        measurement = PermuteAndFlip(candidates=candidates, epsilon=0)
        self.assertEqual(measurement.privacy_function({1: 1, 2: 1, 3: 1, 4: 1}), 0)
        self.assertTrue(measurement(quality_scores) in candidates)

    def test_privacy_relation(self):
        """Test that the privacy relation is computed correctly."""
        candidates = ["A", "B", "C", "D"]
        measurement = PermuteAndFlip(candidates=candidates, epsilon=2)

        d_in = {"A": 1, "B": 1, "C": 1, "D": 1}
        self.assertEqual(measurement.privacy_function(d_in), 2)
        self.assertTrue(measurement.privacy_relation(d_in, 2))
        self.assertFalse(measurement.privacy_relation(d_in, sp.Rational("1.99")))

        d_in = {"A": 1, "B": 3, "C": 1, "D": 1}
        self.assertEqual(measurement.privacy_function(d_in), 6)
        self.assertTrue(measurement.privacy_relation(d_in, 6))
        self.assertFalse(measurement.privacy_relation(d_in, 2))
