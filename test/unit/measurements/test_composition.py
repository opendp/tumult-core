"""Unit tests for :mod:`~tmlt.core.measurements.composition`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use
import itertools
from typing import Tuple
from unittest.mock import MagicMock, create_autospec

import sympy as sp
from parameterized import parameterized

from tmlt.core.domains.base import Domain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.composition import Composition
from tmlt.core.measurements.noise_mechanisms import (
    AddGeometricNoise as AddGeometricNoiseToNumber,
)
from tmlt.core.measurements.noise_mechanisms import (
    AddLaplaceNoise as AddLaplaceNoiseToNumber,
)
from tmlt.core.measures import ApproxDP, Measure, PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference, HammingDistance, SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    create_mock_measurement,
    get_all_props,
)


class TestComposition(TestComponent):
    """Tests for :class:`~tmlt.core.measurements.composition.Composition`."""

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        measurements = [AddGeometricNoiseToNumber(alpha=0)]
        measurement = Composition(measurements=measurements)
        measurements.append(measurements[0])
        self.assertEqual(len(measurement.measurements), 1)

    @parameterized.expand(get_all_props(Composition))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = Composition(
            [
                AddGeometricNoiseToNumber(alpha=0),
                AddLaplaceNoiseToNumber(scale=0, input_domain=NumpyIntegerDomain()),
                AddGeometricNoiseToNumber(alpha=0),
            ]
        )
        assert_property_immutability(measurement, prop_name)

    @parameterized.expand(
        [
            (NumpyFloatDomain(), PureDP()),
            (NumpyIntegerDomain(), PureDP()),
            (NumpyFloatDomain(), RhoZCDP()),
            (NumpyFloatDomain(), ApproxDP()),
        ]
    )
    def test_properties(self, input_domain: Domain, output_measure: Measure):
        """Composition's properties have the expected values."""
        input_metric = AbsoluteDifference()

        measurement1 = create_mock_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=False,
        )

        measurement2 = create_mock_measurement(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=False,
        )

        measurement = Composition([measurement1, measurement2])
        self.assertEqual(measurement.input_domain, input_domain)
        self.assertEqual(measurement.input_metric, input_metric)
        self.assertEqual(measurement.output_measure, output_measure)
        self.assertEqual(measurement.is_interactive, False)

    def test_empty_measurement(self):
        """Tests that empty measurement raises error."""
        with self.assertRaisesRegex(ValueError, "No measurements!"):
            Composition([])

    def test_domains(self):
        """Tests that composition fails with mismatching domains."""
        with self.assertRaisesRegex(
            ValueError, "Can not compose measurements: mismatching input domains"
        ):
            Composition(
                [
                    AddLaplaceNoiseToNumber(
                        scale=10, input_domain=NumpyIntegerDomain()
                    ),
                    AddLaplaceNoiseToNumber(scale=1, input_domain=NumpyFloatDomain()),
                ]
            )

    def test_metric_compatibility(self):
        """Tests that composition fails with mismatching metrics."""
        with self.assertRaisesRegex(
            ValueError, "Can not compose measurements: mismatching input metrics"
        ):
            Composition(
                [
                    create_mock_measurement(input_metric=SymmetricDifference()),
                    create_mock_measurement(input_metric=HammingDistance()),
                ]
            )

    @parameterized.expand(
        [
            (d_in, *params1, *params2, use_hint)
            for d_in in [1, 2]
            for params1, params2 in itertools.combinations(
                [
                    (True, d_in * 1, True),
                    (True, d_in * 2, False),
                    (False, d_in * 1, True),
                    (False, d_in * 2, False),
                ],
                2,
            )
            for use_hint in [True, False]
        ]
    )
    def test_privacy_function_and_relation_pure_dp(
        self,
        d_in: ExactNumberInput,
        privacy_function_implemented1: bool,
        privacy_function_return_value1: ExactNumberInput,
        privacy_relation_return_value1: bool,
        privacy_function_implemented2: bool,
        privacy_function_return_value2: ExactNumberInput,
        privacy_relation_return_value2: bool,
        use_hint: bool,
    ):
        """Tests that the privacy function and relation for pure dp."""
        privacy_function_return_value1 = ExactNumber(privacy_function_return_value1)
        privacy_function_return_value2 = ExactNumber(privacy_function_return_value2)
        mock_measurement1 = create_mock_measurement(
            privacy_function_implemented=privacy_function_implemented1,
            privacy_function_return_value=privacy_function_return_value1,
            privacy_relation_return_value=privacy_relation_return_value1,
            output_measure=PureDP(),
        )
        mock_measurement2 = create_mock_measurement(
            privacy_function_implemented=privacy_function_implemented2,
            privacy_function_return_value=privacy_function_return_value2,
            privacy_relation_return_value=privacy_relation_return_value2,
            output_measure=PureDP(),
        )
        mock_hint = MagicMock(return_value=(d_in * 1, d_in * 1))

        measurement = Composition(
            [mock_measurement1, mock_measurement2], hint=mock_hint if use_hint else None
        )
        if not (privacy_function_implemented1 and privacy_function_implemented2):
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                measurement.privacy_function(d_in)
        else:
            self.assertEqual(
                measurement.privacy_function(d_in),
                privacy_function_return_value1 + privacy_function_return_value2,
            )
        if (
            not (privacy_function_implemented1 and privacy_function_implemented2)
            and not use_hint
        ):
            self.assertRaisesRegex(
                ValueError,
                "A hint is needed to check this privacy relation, because the "
                "privacy_relation from one of self.measurements raised a "
                "NotImplementedError: TEST",
            )
        else:
            self.assertEqual(
                measurement.privacy_relation(d_in, d_in * 2),
                mock_measurement1.privacy_relation(d_in, d_in * 1)
                and mock_measurement2.privacy_relation(d_in, d_in * 1),
            )
            if mock_hint.called:
                mock_hint.assert_called_with(d_in, d_in * 2)
            self.assertFalse(
                measurement.privacy_relation(d_in, d_in * sp.Rational("1.99"))
            )

    @parameterized.expand(
        [
            (d_in, *params1, *params2, use_hint)
            for d_in in [1, 2]
            for params1, params2 in itertools.combinations(
                [
                    (True, d_in ** 2 * 1, True),
                    (True, d_in ** 2 * 2, False),
                    (False, d_in ** 2 * 1, True),
                    (False, d_in ** 2 * 2, False),
                ],
                2,
            )
            for use_hint in [True, False]
        ]
    )
    def test_privacy_function_and_relation_rho_zcdp(
        self,
        d_in: ExactNumberInput,
        privacy_function_implemented1: bool,
        privacy_function_return_value1: ExactNumberInput,
        privacy_relation_return_value1: bool,
        privacy_function_implemented2: bool,
        privacy_function_return_value2: ExactNumberInput,
        privacy_relation_return_value2: bool,
        use_hint: bool,
    ):
        """Tests that the privacy function and relation for pure dp."""
        privacy_function_return_value1 = ExactNumber(privacy_function_return_value1)
        privacy_function_return_value2 = ExactNumber(privacy_function_return_value2)
        d_in = ExactNumber(d_in)
        mock_measurement1 = create_mock_measurement(
            privacy_function_implemented=privacy_function_implemented1,
            privacy_function_return_value=privacy_function_return_value1,
            privacy_relation_return_value=privacy_relation_return_value1,
            output_measure=RhoZCDP(),
        )
        mock_measurement2 = create_mock_measurement(
            privacy_function_implemented=privacy_function_implemented2,
            privacy_function_return_value=privacy_function_return_value2,
            privacy_relation_return_value=privacy_relation_return_value2,
            output_measure=RhoZCDP(),
        )
        mock_hint = MagicMock(return_value=(d_in ** 2 * 1, d_in ** 2 * 1))

        measurement = Composition(
            [mock_measurement1, mock_measurement2], hint=mock_hint if use_hint else None
        )
        if not (privacy_function_implemented1 and privacy_function_implemented2):
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                measurement.privacy_function(d_in)
        else:
            self.assertEqual(
                measurement.privacy_function(d_in),
                privacy_function_return_value1 + privacy_function_return_value2,
            )
        if (
            not (privacy_function_implemented1 and privacy_function_implemented2)
            and not use_hint
        ):
            self.assertRaisesRegex(
                ValueError,
                "A hint is needed to check this privacy relation, because the "
                "privacy_relation from one of self.measurements raised a "
                "NotImplementedError: TEST",
            )
        else:
            self.assertEqual(
                measurement.privacy_relation(d_in, d_in ** 2 * 2),
                mock_measurement1.privacy_relation(d_in, d_in ** 2 * 1)
                and mock_measurement2.privacy_relation(d_in, d_in ** 2 * 1),
            )
            if mock_hint.called:
                mock_hint.assert_called_with(d_in, d_in ** 2 * 2)
            self.assertFalse(
                measurement.privacy_relation(d_in, d_in ** 2 * sp.Rational("1.99"))
            )

    @parameterized.expand(
        [
            (d_in, *params1, *params2, use_hint)
            for d_in in [1, 2]
            for params1, params2 in itertools.combinations(
                [
                    (True, (d_in * 1, sp.Rational("0.2")), True),
                    (True, (d_in * 2, sp.Rational("0.2")), False),
                    (True, (d_in * 1, sp.Rational("0.3")), False),
                    (False, (d_in * 1, sp.Rational("0.2")), True),
                    (False, (d_in * 2, sp.Rational("0.2")), False),
                    (False, (d_in * 1, sp.Rational("0.3")), False),
                ],
                2,
            )
            for use_hint in [True, False]
        ]
    )
    def test_privacy_function_and_relation_approx_dp(
        self,
        d_in: ExactNumberInput,
        privacy_function_implemented1: bool,
        privacy_function_return_value1: Tuple[ExactNumberInput, ExactNumberInput],
        privacy_relation_return_value1: bool,
        privacy_function_implemented2: bool,
        privacy_function_return_value2: Tuple[ExactNumberInput, ExactNumberInput],
        privacy_relation_return_value2: bool,
        use_hint: bool,
    ):
        """Tests that the privacy function and relation for pure dp."""
        privacy_function_return_value1 = (
            ExactNumber(privacy_function_return_value1[0]),
            ExactNumber(privacy_function_return_value1[1]),
        )
        privacy_function_return_value2 = (
            ExactNumber(privacy_function_return_value2[0]),
            ExactNumber(privacy_function_return_value2[1]),
        )
        mock_measurement1 = create_mock_measurement(
            privacy_function_implemented=privacy_function_implemented1,
            privacy_function_return_value=privacy_function_return_value1,
            privacy_relation_return_value=privacy_relation_return_value1,
            output_measure=ApproxDP(),
        )
        mock_measurement2 = create_mock_measurement(
            privacy_function_implemented=privacy_function_implemented2,
            privacy_function_return_value=privacy_function_return_value2,
            privacy_relation_return_value=privacy_relation_return_value2,
            output_measure=ApproxDP(),
        )
        mock_hint = MagicMock(
            return_value=(
                (d_in * 1, sp.Rational("0.2")),
                (d_in * 1, sp.Rational("0.2")),
            )
        )

        measurement = Composition(
            [mock_measurement1, mock_measurement2], hint=mock_hint if use_hint else None
        )
        if not (privacy_function_implemented1 and privacy_function_implemented2):
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                measurement.privacy_function(d_in)
        else:
            self.assertEqual(
                measurement.privacy_function(d_in),
                (
                    privacy_function_return_value1[0]
                    + privacy_function_return_value2[0],
                    privacy_function_return_value1[1]
                    + privacy_function_return_value2[1],
                ),
            )
        if (
            not (privacy_function_implemented1 and privacy_function_implemented2)
            and not use_hint
        ):
            self.assertRaisesRegex(
                ValueError,
                "A hint is needed to check this privacy relation, because the "
                "privacy_relation from one of self.measurements raised a "
                "NotImplementedError: TEST",
            )
        else:
            self.assertEqual(
                measurement.privacy_relation(d_in, (d_in * 2, sp.Rational("0.4"))),
                mock_measurement1.privacy_relation(d_in, (d_in * 1, sp.Rational("0.2")))
                and mock_measurement2.privacy_relation(
                    d_in, (d_in * 1, sp.Rational("0.2"))
                ),
            )
            if mock_hint.called:
                mock_hint.assert_called_with(d_in, (d_in * 2, sp.Rational("0.4")))
            self.assertFalse(
                measurement.privacy_relation(
                    d_in, (d_in * sp.Rational("1.99"), sp.Rational("0.4"))
                )
            )
            self.assertFalse(
                measurement.privacy_relation(d_in, (d_in * 2, sp.Rational("0.399")))
            )

    def test_composed_measurement(self):
        """Tests composition works correctly."""
        measurement = Composition(
            [
                AddGeometricNoiseToNumber(alpha=0),
                AddLaplaceNoiseToNumber(scale=0, input_domain=NumpyIntegerDomain()),
                AddGeometricNoiseToNumber(alpha=0),
            ]
        )
        actual_answer = measurement(2)
        self.assertEqual(actual_answer, [2, 2.0, 2])

    @parameterized.expand(
        [
            # mismatching output measure
            (
                create_mock_measurement(),
                create_mock_measurement(output_measure=RhoZCDP()),
                "mismatching output measures",
            ),
            # interactive PurePD
            (
                create_mock_measurement(output_measure=PureDP()),
                create_mock_measurement(output_measure=PureDP(), is_interactive=True),
                "Cannot compose interactive measurements.",
            ),
            # interactive ApproxDP
            (
                create_mock_measurement(output_measure=ApproxDP()),
                create_mock_measurement(output_measure=ApproxDP(), is_interactive=True),
                "Cannot compose interactive measurements.",
            ),
            # interactive RhoZCDP
            (
                create_mock_measurement(output_measure=RhoZCDP()),
                create_mock_measurement(output_measure=RhoZCDP(), is_interactive=True),
                "Cannot compose interactive measurements.",
            ),
            # unsupported output measure
            (
                create_mock_measurement(
                    output_measure=create_autospec(spec=Measure, instance=True)
                ),
                create_mock_measurement(
                    output_measure=create_autospec(spec=Measure, instance=True)
                ),
                "Unsupported output measure",
            ),
        ]
    )
    def test_validation(
        self, measurement1: Measurement, measurement2: Measurement, msg: str
    ):
        """Test that exceptions are correctly raised."""
        with self.assertRaisesRegex(ValueError, msg):
            Composition([measurement1, measurement2])
