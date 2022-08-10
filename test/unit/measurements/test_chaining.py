"""Unit tests for :mod:`~tmlt.core.measurements.chaining`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use
import itertools
from unittest.mock import MagicMock, call

import sympy as sp
from parameterized import parameterized

from tmlt.core.domains.base import Domain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.measurements.chaining import ChainTM
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference, Metric, SymmetricDifference
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    create_mock_measurement,
    create_mock_transformation,
    get_all_props,
)


class TestChainTM(TestComponent):
    """Tests for :class:`~tmlt.core.measurements.chaining.ChainTM`."""

    @parameterized.expand(get_all_props(ChainTM))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = create_mock_transformation(
            output_domain=NumpyIntegerDomain(), output_metric=AbsoluteDifference()
        )
        measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(), input_metric=AbsoluteDifference()
        )
        chained_measurement = ChainTM(
            transformation=transformation,
            measurement=measurement,
            hint=lambda _, __: sp.Integer(1),
        )
        assert_property_immutability(chained_measurement, prop_name)

    @parameterized.expand(
        [
            (
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                NumpyFloatDomain(),
                SymmetricDifference(),
                PureDP(),
                True,
            ),
            (
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                NumpyFloatDomain(),
                SymmetricDifference(),
                RhoZCDP(),
                False,
            ),
        ]
    )
    def test_properties(
        self,
        input_domain: Domain,
        input_metric: Metric,
        mid_domain: Domain,
        mid_metric: Metric,
        output_measure: Measure,
        is_interactive: bool,
    ):
        """ChainTM's properties have the expected values."""
        transformation = create_mock_transformation(
            input_domain=input_domain,
            output_domain=mid_domain,
            input_metric=input_metric,
            output_metric=mid_metric,
        )

        measurement = create_mock_measurement(
            input_domain=mid_domain,
            input_metric=mid_metric,
            output_measure=output_measure,
            is_interactive=is_interactive,
        )
        chained_measurement = ChainTM(
            transformation=transformation,
            measurement=measurement,
            hint=lambda _, __: sp.Integer(1),
        )
        self.assertEqual(chained_measurement.input_domain, input_domain)
        self.assertEqual(chained_measurement.input_metric, input_metric)
        self.assertEqual(chained_measurement.output_measure, output_measure)
        self.assertEqual(chained_measurement.is_interactive, is_interactive)

    def test_chained_measurement(self):
        """Tests that chained measurement is correct."""
        transformation = create_mock_transformation(return_value=4)
        measurement = create_mock_measurement(return_value=5)
        chained_measurement = ChainTM(
            transformation=transformation,
            measurement=measurement,
            hint=lambda d_in, _: 2 * d_in,
        )
        expected = 5
        actual = chained_measurement(2)
        self.assertEqual(actual, expected)
        self.assertIn(call(4), measurement.mock_calls)

    @parameterized.expand(
        [
            (*params1, *params2, use_hint)
            for params1, params2 in itertools.combinations(
                [
                    (True, sp.Integer(1), True),
                    (True, sp.Integer(2), False),
                    (False, sp.Integer(1), True),
                    (False, sp.Integer(2), False),
                ],
                2,
            )
            for use_hint in [True, False]
        ]
    )
    def test_privacy_function_and_relation(
        self,
        stability_function_implemented: bool,
        stability_function_return_value: sp.Expr,
        stability_relation_return_value: bool,
        privacy_function_implemented: bool,
        privacy_function_return_value: sp.Expr,
        privacy_relation_return_value: bool,
        use_hint: bool,
    ):
        """Tests that the privacy function and relation work correctly."""
        mock_transformation = create_mock_transformation(
            stability_function_implemented=stability_function_implemented,
            stability_function_return_value=stability_function_return_value,
            stability_relation_return_value=stability_relation_return_value,
        )
        mock_measurement = create_mock_measurement(
            privacy_function_implemented=privacy_function_implemented,
            privacy_function_return_value=privacy_function_return_value,
            privacy_relation_return_value=privacy_relation_return_value,
        )
        mock_hint = MagicMock(return_value=(sp.Integer(1), sp.Integer(1)))
        measurement = ChainTM(
            transformation=mock_transformation,
            measurement=mock_measurement,
            hint=mock_hint if use_hint else None,
        )
        if not (privacy_function_implemented and stability_function_implemented):
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                measurement.privacy_function(sp.Integer(1))
        else:
            self.assertEqual(
                measurement.privacy_function(sp.Integer(1)),
                privacy_function_return_value,
            )
        if not stability_function_implemented and not use_hint:
            with self.assertRaisesRegex(
                ValueError,
                "A hint is needed to check this privacy relation, because the "
                "stability_relation of self.transformation raised a "
                "NotImplementedError: TEST",
            ):
                measurement.privacy_relation(sp.Integer(1), sp.Integer(1))
        else:
            self.assertEqual(
                measurement.privacy_relation(sp.Integer(1), sp.Integer(1)),
                mock_transformation.stability_relation(sp.Integer(1), sp.Integer(1))
                and mock_measurement.privacy_relation(sp.Integer(1), sp.Integer(1)),
            )
            if mock_hint.called:
                mock_hint.assert_called_with(sp.Integer(1), sp.Integer(1))

    def test_incompatible_domains_fails(self):
        """Tests that chaining fails with incompatible domains."""
        transformation = create_mock_transformation(output_domain=NumpyFloatDomain())
        measurement = create_mock_measurement(input_domain=NumpyIntegerDomain())
        with self.assertRaisesRegex(
            ValueError,
            "Can not chain transformation and measurement: Mismatching domains.",
        ):
            ChainTM(
                transformation=transformation,
                measurement=measurement,
                hint=lambda _, __: None,
            )
