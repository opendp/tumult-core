"""Unit tests for :mod:`~tmlt.core.measurements.postprocess`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use

from unittest.mock import MagicMock, call

from parameterized import parameterized

from tmlt.core.domains.base import Domain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.measurements.interactive_measurements import Queryable
from tmlt.core.measurements.postprocess import NonInteractivePostProcess, PostProcess
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference, Metric
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    create_mock_measurement,
    get_all_props,
)


class TestPostProcess(TestComponent):
    """Tests for :class:`~tmlt.core.measurements.postprocess.PostProcess`."""

    @parameterized.expand([(NumpyFloatDomain(), AbsoluteDifference(), PureDP())])
    def test_properties(
        self, input_domain: Domain, input_metric: Metric, output_measure: Measure
    ):
        """PostProcess's properties have the expected values."""
        measurement = create_mock_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=False,
        )
        postprocessed_measurement = PostProcess(measurement=measurement, f=lambda x: x)
        self.assertEqual(postprocessed_measurement.input_domain, input_domain)
        self.assertEqual(postprocessed_measurement.input_metric, input_metric)
        self.assertEqual(postprocessed_measurement.output_measure, output_measure)
        self.assertEqual(postprocessed_measurement.is_interactive, False)

    @parameterized.expand(get_all_props(PostProcess))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = PostProcess(measurement=create_mock_measurement(), f=lambda x: x)
        assert_property_immutability(measurement, prop_name)

    def test_correctness(self):
        """PostProcess gives the correct result."""
        measurement = create_mock_measurement(
            return_value=5,
            privacy_function_implemented=True,
            privacy_function_return_value=2,
            privacy_relation_return_value=True,
        )
        f = MagicMock(return_value=10)
        postprocessed_measurement = PostProcess(measurement=measurement, f=f)

        result = postprocessed_measurement(3)
        # TODO: I couldn't get the below code to work. I think it is related
        #  to self being passed to the call. https://bugs.python.org/issue27715
        # measurement.assert_called_with(3)
        self.assertEqual(measurement.mock_calls, [call(3)])  # using this instead.
        f.assert_called_with(5)
        self.assertEqual(result, 10)
        self.assertEqual(measurement.privacy_function(1), 2)
        self.assertTrue(measurement.privacy_relation(1, 2))

    @parameterized.expand(
        [
            (
                privacy_function_implemented,
                privacy_function_return_value,
                privacy_relation_return_value,
            )
            for privacy_function_implemented in [True, False]
            for privacy_function_return_value, privacy_relation_return_value in [
                (1, True),
                (2, False),
            ]
        ]
    )
    def test_privacy_function_and_relation(
        self,
        privacy_function_implemented: bool,
        privacy_function_return_value: ExactNumberInput,
        privacy_relation_return_value: bool,
    ):
        """Tests that the privacy function and relation work correctly."""
        measurement = create_mock_measurement(
            return_value=5,
            privacy_function_implemented=privacy_function_implemented,
            privacy_function_return_value=privacy_function_return_value,
            privacy_relation_return_value=privacy_relation_return_value,
        )
        f = MagicMock(return_value=10)
        postprocessed_measurement = PostProcess(measurement=measurement, f=f)
        if not privacy_function_implemented:
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                postprocessed_measurement.privacy_function(1)
        else:
            self.assertEqual(
                postprocessed_measurement.privacy_function(1),
                privacy_function_return_value,
            )
        self.assertEqual(
            postprocessed_measurement.privacy_relation(1, 1),
            privacy_relation_return_value,
        )

    def test_postprocess_raises_error_on_interactive_measurements(self):
        """PostProcess raises error if measurement is interactive."""
        measurement = create_mock_measurement(return_value=5, is_interactive=True)
        f = MagicMock(return_value=10)
        with self.assertRaisesRegex(
            ValueError,
            "PostProcess can only be used with a non-interactive measurement",
        ):
            PostProcess(measurement=measurement, f=f)


class TestNonInteractivePostProcess(TestComponent):
    """Tests for class NonInteractivePostProcess.

    Tests :class:`~tmlt.core.measurements.postprocess.NonInteractivePostProcess`.
    """

    @parameterized.expand(
        [
            (NumpyFloatDomain(), AbsoluteDifference(), PureDP()),
            (NumpyIntegerDomain(), AbsoluteDifference(), RhoZCDP()),
        ]
    )
    def test_properties(
        self, input_domain: Domain, input_metric: Metric, output_measure: Measure
    ):
        """NonInteractivePostProcess's properties have the expected values."""
        measurement = create_mock_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=True,
        )
        postprocessed_measurement = NonInteractivePostProcess(
            measurement=measurement, f=lambda x: x
        )
        self.assertEqual(postprocessed_measurement.input_domain, input_domain)
        self.assertEqual(postprocessed_measurement.input_metric, input_metric)
        self.assertEqual(postprocessed_measurement.output_measure, output_measure)
        self.assertEqual(postprocessed_measurement.is_interactive, False)

    @parameterized.expand(get_all_props(PostProcess))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = NonInteractivePostProcess(
            measurement=create_mock_measurement(is_interactive=True), f=lambda x: x
        )
        assert_property_immutability(measurement, prop_name)

    def test_correctness(self):
        """NonInteractivePostProcess gives the correct result."""
        mock_queryable = MagicMock(spec=Queryable, return_value=10)
        measurement = create_mock_measurement(
            is_interactive=True,
            return_value=mock_queryable,
            privacy_function_implemented=True,
            privacy_function_return_value=2,
            privacy_relation_return_value=True,
        )

        def f(queryable: Queryable) -> int:
            return queryable(5)

        postprocessed_measurement = NonInteractivePostProcess(
            measurement=measurement, f=f
        )

        result = postprocessed_measurement(3)
        self.assertEqual(measurement.mock_calls, [call(3), call()(5)])
        self.assertEqual(mock_queryable.mock_calls, [call(5)])
        self.assertEqual(result, 10)
        self.assertEqual(measurement.privacy_function(1), 2)
        self.assertTrue(measurement.privacy_relation(1, 2))

    @parameterized.expand(
        [
            (
                privacy_function_implemented,
                privacy_function_return_value,
                privacy_relation_return_value,
            )
            for privacy_function_implemented in [True, False]
            for privacy_function_return_value, privacy_relation_return_value in [
                (1, True),
                (2, False),
            ]
        ]
    )
    def test_privacy_function_and_relation(
        self,
        privacy_function_implemented: bool,
        privacy_function_return_value: ExactNumberInput,
        privacy_relation_return_value: bool,
    ):
        """Tests that the privacy function and relation work correctly."""
        measurement = create_mock_measurement(
            return_value=5,
            privacy_function_implemented=privacy_function_implemented,
            privacy_function_return_value=privacy_function_return_value,
            privacy_relation_return_value=privacy_relation_return_value,
            is_interactive=True,
        )
        f = MagicMock(return_value=10)
        postprocessed_measurement = NonInteractivePostProcess(
            measurement=measurement, f=f
        )
        if not privacy_function_implemented:
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                postprocessed_measurement.privacy_function(1)
        else:
            self.assertEqual(
                postprocessed_measurement.privacy_function(1),
                privacy_function_return_value,
            )
        self.assertEqual(
            postprocessed_measurement.privacy_relation(1, 1),
            privacy_relation_return_value,
        )
