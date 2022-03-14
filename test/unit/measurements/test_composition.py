"""Unit tests for :mod:`~tmlt.core.measurements.composition`."""

# <placeholder: boilerplate>

# pylint: disable=no-self-use
import itertools
import re
import unittest
from typing import Optional, Tuple, Type, Union
from unittest.mock import MagicMock, call, create_autospec

import numpy as np
import sympy as sp
from parameterized import parameterized, parameterized_class

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.measurements.base import Measurement, Queryable
from tmlt.core.measurements.composition import (
    Composition,
    MeasurementQuery,
    NestedQuery,
    ParallelComposition,
    ParallelCompositionQueryable,
    SequentialComposition,
    SequentialCompositionQueryable,
    create_adaptive_composition,
    unpack_parallel_composition_queryable,
)
from tmlt.core.measurements.noise_mechanisms import (
    AddGeometricNoise as AddGeometricNoiseToNumber,
)
from tmlt.core.measurements.noise_mechanisms import (
    AddLaplaceNoise as AddLaplaceNoiseToNumber,
)
from tmlt.core.measures import ApproxDP, Measure, PureDP, RhoZCDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    HammingDistance,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    create_mock_measurement,
    get_all_props,
)


class TestComposition(TestComponent):
    """Tests for :class:`~tmlt.core.measurements.composition.Composition`."""

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


@parameterized_class([{"output_measure": PureDP()}, {"output_measure": RhoZCDP()}])
class TestSequentialComposition(unittest.TestCase):
    """Tests for class SequentialComposition.

    Tests :class:`~tmlt.core.measurements.composition.SequentialComposition`.
    """

    output_measure: Union[PureDP, RhoZCDP]
    """The output measure to use during the tests."""

    def setUp(self):
        """Set up class."""
        self._data = np.int64(10)
        self._measurement = SequentialComposition(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            d_in=1,
            privacy_budget=6,
        )
        self._noninteractive_measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            is_interactive=False,
            return_value=123,
            privacy_relation_return_value=True,
        )

        self._inner_queryable = create_autospec(spec=Queryable, instance=True)
        self._inner_queryable.return_value = 456
        self._interactive_measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            is_interactive=True,
            return_value=self._inner_queryable,
            privacy_relation_return_value=True,
        )

    @parameterized.expand(get_all_props(SequentialComposition))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self._measurement, prop_name)

    @parameterized.expand(get_all_props(SequentialCompositionQueryable))
    def test_queryable_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self._measurement(self._data), prop_name)

    def test_properties(self):
        """SequentialComposition's properties have the expected values."""
        self.assertEqual(self._measurement.input_domain, NumpyIntegerDomain())
        self.assertEqual(self._measurement.input_metric, AbsoluteDifference())
        self.assertEqual(self._measurement.output_measure, self.output_measure)
        self.assertEqual(self._measurement.is_interactive, True)
        self.assertEqual(self._measurement.d_in, 1)
        self.assertEqual(self._measurement.privacy_budget, 6)

    def test_queryable_properties(self):
        """SequentialCompositionQueryable's properties have the expected values."""
        queryable = self._measurement(self._data)
        self.assertEqual(queryable.input_domain, NumpyIntegerDomain())
        self.assertEqual(queryable.input_metric, AbsoluteDifference())
        self.assertEqual(queryable.output_measure, self.output_measure)
        self.assertEqual(queryable.d_in, 1)
        self.assertEqual(queryable.privacy_budget, 6)
        self.assertEqual(queryable.remaining_budget, 6)

    def test_multiple_noninteractive_measurements(self):
        """Test repeated noninteractive measurements until budget runs out."""
        queryable = self._measurement(self._data)
        query = NestedQuery(
            index=0,
            get_property=False,
            inner_query=MeasurementQuery(
                measurement=self._noninteractive_measurement, d_out=2
            ),
        )
        for i in range(3):
            return_value = queryable(query)
            self.assertEqual(return_value, 123)
            self.assertEqual(queryable.remaining_budget, 6 - 2 * (i + 1))
        self.assertEqual(
            self._noninteractive_measurement.mock_calls,
            [call.privacy_function(1), call.privacy_relation(1, 2), call(self._data)]
            * 3,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot answer measurement without exceeding maximum privacy loss: "
            "it needs 2, but the remaining budget is 0",
        ):
            queryable(query)

    def test_multiple_interactive_measurements(self):
        """Test repeated interactive measurements until budget runs out."""
        queryable = self._measurement(self._data)
        query = NestedQuery(
            index=0,
            get_property=False,
            inner_query=MeasurementQuery(
                measurement=self._interactive_measurement, d_out=2
            ),
        )
        for i in range(3):
            return_value = queryable(query)
            self.assertEqual(return_value, i + 1)

        self.assertEqual(
            self._interactive_measurement.mock_calls,
            [call.privacy_function(1), call.privacy_relation(1, 2), call(self._data)]
            * 3,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot answer measurement without exceeding maximum privacy loss: "
            "it needs 2, but the remaining budget is 0",
        ):
            queryable(query)

    @parameterized.expand(
        [(2, 3, True), (1, 4, True), (2, 2, False), (3, 3, False), (4, 6, False)]
    )
    def test_privacy_relation_and_function(
        self,
        d_in: ExactNumberInput,
        d_out: ExactNumberInput,
        expected_privacy_relation: bool,
    ):
        """Tests that privacy function and relation work correctly."""
        d_in = ExactNumber(d_in)
        d_out = ExactNumber(d_out)
        measurement = SequentialComposition(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            d_in=2,
            privacy_budget=3,
        )
        if d_in > 2:
            with self.assertRaisesRegex(ValueError, f"d_in must be <= 2, not {d_in}"):
                measurement.privacy_function(d_in)
            with self.assertRaisesRegex(ValueError, f"d_in must be <= 2, not {d_in}"):
                measurement.privacy_relation(d_in, d_out)
        else:
            self.assertEqual(measurement.privacy_function(d_in), 3)
            self.assertEqual(
                measurement.privacy_relation(d_in, d_out), expected_privacy_relation
            )

    @parameterized.expand(
        [
            (privacy_function_implemented, d_out)
            for privacy_function_implemented in [True, False]
            for d_out in [1, None]
        ]
    )
    def test_d_out_needed_unless_privacy_function_implemented(
        self, privacy_function_implemented: bool, d_out: ExactNumberInput
    ):
        """Only measurements with privacy functions don't need to pass d_out."""
        queryable = SequentialComposition(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            d_in=2,
            privacy_budget=3,
        )(np.int64(1))
        query = NestedQuery(
            index=0,
            get_property=False,
            inner_query=MeasurementQuery(
                create_mock_measurement(
                    privacy_function_implemented=privacy_function_implemented,
                    output_measure=self.output_measure,
                ),
                d_out=d_out,
            ),
        )
        if not privacy_function_implemented and d_out is None:
            with self.assertRaisesRegex(
                ValueError,
                "A d_out is required for the MeasurementQuery because the "
                "provided measurement's privacy_function raised "
                "NotImplementedError: TEST",
            ):
                queryable(query)
        else:
            self.assertEqual(queryable(query), np.int64(0))

    def test_infinite_budget(self):
        """Infinite budget allows spending finite or infinite privacy budget."""
        queryable = SequentialComposition(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            d_in=1,
            privacy_budget=sp.oo,
        )(self._data)
        measurement_queries = [
            MeasurementQuery(measurement=self._interactive_measurement, d_out=2),
            MeasurementQuery(measurement=self._noninteractive_measurement, d_out=2),
            MeasurementQuery(measurement=self._interactive_measurement, d_out=sp.oo),
            MeasurementQuery(measurement=self._noninteractive_measurement, d_out=sp.oo),
            MeasurementQuery(measurement=self._interactive_measurement, d_out=sp.oo),
        ]
        for measurement_query in measurement_queries:
            query = NestedQuery(
                index=0, get_property=False, inner_query=measurement_query
            )
            queryable(query)
            self.assertFalse(queryable.remaining_budget.is_finite)

    def test_activating_queryable_disables_existing_queryable(self):
        """Activating another queryable should deactivate existing queryable."""
        queryable = self._measurement(self._data)
        # create first inner queryable
        self.assertEqual(
            queryable(
                NestedQuery(
                    index=0,
                    get_property=False,
                    inner_query=MeasurementQuery(
                        measurement=self._interactive_measurement, d_out=2
                    ),
                )
            ),
            1,
        )
        # create second inner queryable
        self.assertEqual(
            queryable(
                NestedQuery(
                    index=0,
                    get_property=False,
                    inner_query=MeasurementQuery(
                        measurement=self._interactive_measurement, d_out=2
                    ),
                )
            ),
            2,
        )
        # activate first inner queryable
        self.assertEqual(
            queryable(NestedQuery(index=1, get_property=False, inner_query=111)), 456
        )
        self.assertEqual(self._inner_queryable.mock_calls, [call(111)])
        # deactivate it
        self.assertEqual(
            queryable(NestedQuery(index=2, get_property=False, inner_query=111)), 456
        )
        with self.assertRaisesRegex(
            RuntimeError, re.escape("The specified queryable (1) is no longer active")
        ):
            queryable(NestedQuery(index=1, get_property=False, inner_query=111))

    def test_answering_noninteractive_measurement_disables_queryable(self):
        """Answering a measurement should deactivate existing queryable."""
        queryable = self._measurement(self._data)
        # create inner queryable
        self.assertEqual(
            queryable(
                NestedQuery(
                    index=0,
                    get_property=False,
                    inner_query=MeasurementQuery(
                        measurement=self._interactive_measurement, d_out=2
                    ),
                )
            ),
            1,
        )
        # activate inner queryable
        self.assertEqual(
            queryable(NestedQuery(index=1, get_property=False, inner_query=111)), 456
        )
        self.assertEqual(self._inner_queryable.mock_calls, [call(111)])
        # deactivate it
        self.assertEqual(
            queryable(
                NestedQuery(
                    index=0,
                    get_property=False,
                    inner_query=MeasurementQuery(
                        measurement=self._noninteractive_measurement, d_out=2
                    ),
                )
            ),
            123,
        )
        with self.assertRaisesRegex(
            RuntimeError, re.escape("The specified queryable (1) is no longer active")
        ):
            queryable(NestedQuery(index=1, get_property=False, inner_query=111))

    def test_accessing_inner_property(self):
        """Test accessing a property from an inner queryable."""
        queryable = self._measurement(self._data)
        self._inner_queryable.some_property = 17
        # create inner queryable
        self.assertEqual(
            queryable(
                NestedQuery(
                    index=0,
                    get_property=False,
                    inner_query=MeasurementQuery(
                        measurement=self._interactive_measurement, d_out=2
                    ),
                )
            ),
            1,
        )
        self.assertEqual(
            queryable(
                NestedQuery(index=1, get_property=True, inner_query="some_property")
            ),
            17,
        )

    def test_accessing_base_property(self):
        """Test accessing a property from the base queryable."""
        queryable = self._measurement(self._data)
        self.assertEqual(
            queryable(
                NestedQuery(index=0, get_property=True, inner_query="remaining_budget")
            ),
            6,
        )

    def test_nested_sequential_composition(self):
        """Test nested sequential composition."""
        queryable = self._measurement(self._data)
        # Create the nested sequential composition
        query = NestedQuery(
            index=0,
            get_property=False,
            inner_query=MeasurementQuery(measurement=self._measurement, d_out=6),
        )
        index = queryable(query)
        self.assertEqual(index, 1)
        # Send a measurement to the inner sequential composition queryable
        nested_query = NestedQuery(
            index=1,
            get_property=False,
            inner_query=NestedQuery(
                index=0,
                get_property=False,
                inner_query=MeasurementQuery(
                    measurement=self._noninteractive_measurement, d_out=2
                ),
            ),
        )
        return_value = queryable(nested_query)
        self.assertEqual(return_value, 123)
        self.assertEqual(queryable.remaining_budget, 0)
        # Check the remaining budget of the inner sequential composition queryable
        budget_query = NestedQuery(
            index=1, get_property=True, inner_query="remaining_budget"
        )
        remaining_budget = queryable(budget_query)
        self.assertEqual(remaining_budget, 4)

    @parameterized.expand([(True,), (False,)])
    def test_bad_measurements(self, is_interactive: bool):
        """Test that bad measurements are rejected.

        Args:
            is_interactive: Whether the measurement to test is interactive.
        """
        queryable = self._measurement(self._data)

        def try_measurement(message: str, measurement: Measurement):
            with self.assertRaisesRegex(ValueError, re.escape(message)):
                queryable(
                    NestedQuery(
                        index=0,
                        get_property=False,
                        inner_query=MeasurementQuery(measurement=measurement, d_out=2),
                    )
                )

        try_measurement(
            "Measurement does not satisfy provided d_out",
            create_mock_measurement(
                input_domain=NumpyIntegerDomain(),
                input_metric=AbsoluteDifference(),
                output_measure=self.output_measure,
                is_interactive=is_interactive,
                privacy_relation_return_value=False,
            ),
        )

        try_measurement(
            "Input domain of measurement does not match input domain of the Queryable",
            create_mock_measurement(
                input_domain=NumpyStringDomain(),
                input_metric=AbsoluteDifference(),
                output_measure=self.output_measure,
                is_interactive=is_interactive,
            ),
        )

        try_measurement(
            "Input metric of measurement does not match input metric of the Queryable",
            create_mock_measurement(
                input_domain=NumpyIntegerDomain(),
                input_metric=SymmetricDifference(),
                output_measure=self.output_measure,
                is_interactive=is_interactive,
            ),
        )
        try_measurement(
            "Output measure of measurement does not match output measure of the "
            "Queryable",
            create_mock_measurement(
                input_domain=NumpyIntegerDomain(),
                input_metric=AbsoluteDifference(),
                output_measure=PureDP()
                if self.output_measure == RhoZCDP()
                else RhoZCDP(),
                is_interactive=is_interactive,
            ),
        )

    @parameterized.expand([(-1,), (1,)])
    def test_bad_index(self, index: int):
        """Test that bad indexes raise an error."""
        queryable = self._measurement(self._data)
        with self.assertRaises(IndexError):
            queryable(NestedQuery(index=index, get_property=False, inner_query=123))


@parameterized_class(
    [
        {"output_measure": PureDP(), "input_metric_class": SumOf},
        {"output_measure": RhoZCDP(), "input_metric_class": RootSumOfSquared},
    ]
)
class TestParallelComposition(unittest.TestCase):
    """Tests for :class:`~tmlt.core.measurements.composition.ParallelComposition`."""

    output_measure: Union[PureDP, RhoZCDP]
    """The output measure to use during the tests."""

    input_metric_class: Union[Type[SumOf], Type[RootSumOfSquared]]
    """The class of the input metric to use during the test."""

    def setUp(self):
        """Set up class."""
        self._data = [np.int64(10)] * 3
        self._noninteractive_measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            is_interactive=False,
            return_value=123,
            privacy_relation_return_value=True,
        )

        self._inner_queryable = create_autospec(spec=Queryable, instance=True)
        self._inner_queryable.return_value = 456
        self._interactive_measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            is_interactive=True,
            return_value=self._inner_queryable,
            privacy_relation_return_value=True,
        )
        self._measurement = ParallelComposition(
            input_domain=ListDomain(NumpyIntegerDomain(), length=3),
            input_metric=self.input_metric_class(AbsoluteDifference()),
            output_measure=self.output_measure,
            measurements=[
                self._interactive_measurement,
                self._interactive_measurement,
                self._noninteractive_measurement,
            ],
        )

    @parameterized.expand(get_all_props(ParallelComposition))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self._measurement, prop_name)

    @parameterized.expand(get_all_props(ParallelCompositionQueryable))
    def test_queryable_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self._measurement(self._data), prop_name)

    def test_properties(self):
        """ParallelComposition's properties have the expected values."""
        self.assertEqual(
            self._measurement.input_domain, ListDomain(NumpyIntegerDomain(), length=3)
        )
        self.assertEqual(
            self._measurement.input_metric,
            self.input_metric_class(AbsoluteDifference()),
        )
        self.assertEqual(self._measurement.output_measure, self.output_measure)
        self.assertEqual(self._measurement.is_interactive, True)
        self.assertEqual(
            self._measurement.measurements,
            [
                self._interactive_measurement,
                self._interactive_measurement,
                self._noninteractive_measurement,
            ],
        )

    def test_queryable_properties(self):
        """ParallelCompositionQueryable's properties have the expected values."""
        queryable = self._measurement(self._data)
        self.assertEqual(
            queryable.measurements,
            [
                self._interactive_measurement,
                self._interactive_measurement,
                self._noninteractive_measurement,
            ],
        )

    @parameterized.expand(
        [
            (SumOf(AbsoluteDifference()), RhoZCDP()),
            (RootSumOfSquared(AbsoluteDifference()), PureDP()),
        ]
    )
    def test_invalid_metric_measure_combination(
        self,
        input_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
    ):
        """Invalid combinations of input metric and output measure are rejected."""
        message = (
            f"Input metric {input_metric.__class__} is incompatible with "
            f"output measure {output_measure.__class__}"
        )
        with self.assertRaisesRegex(ValueError, message):
            ParallelComposition(
                input_domain=ListDomain(NumpyIntegerDomain(), length=3),
                input_metric=input_metric,
                output_measure=output_measure,
                measurements=[
                    self._interactive_measurement,
                    self._interactive_measurement,
                    self._noninteractive_measurement,
                ],
            )

    def test_bad_measurement(self):
        """Incompatible measurements are rejected."""

        def try_measurement(message: str, measurement: Measurement):
            with self.assertRaisesRegex(ValueError, re.escape(message)):
                ParallelComposition(
                    input_domain=ListDomain(NumpyIntegerDomain(), length=1),
                    input_metric=self.input_metric_class(AbsoluteDifference()),
                    output_measure=self.output_measure,
                    measurements=[measurement],
                )

        try_measurement(
            "Input domain for each measurement must match "
            "element domain of the input domain for ParallelComposition",
            create_mock_measurement(
                input_domain=NumpyStringDomain(),
                input_metric=AbsoluteDifference(),
                output_measure=self.output_measure,
            ),
        )
        try_measurement(
            "Input metric for each supplied measurement must match "
            "inner metric of input metric for ParallelComposition",
            create_mock_measurement(
                input_domain=NumpyIntegerDomain(),
                input_metric=SymmetricDifference(),
                output_measure=self.output_measure,
            ),
        )
        try_measurement(
            "Output measure for each supplied measurement must match "
            "output measure for ParallelComposition",
            create_mock_measurement(
                input_domain=NumpyIntegerDomain(),
                input_metric=AbsoluteDifference(),
                output_measure=PureDP()
                if self.output_measure == RhoZCDP()
                else RhoZCDP(),
            ),
        )

    @parameterized.expand(
        [
            (
                None,
                "Input domain for ParallelComposition must specify number of elements",
            ),
            (
                2,
                "Length of input domain (2) does not match the number of measurements "
                "(1)",
            ),
        ]
    )
    def test_bad_input_domain_length(self, length: Optional[int], message: str):
        """Raises an error if input domain doesn't have the correct length."""
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            ParallelComposition(
                input_domain=ListDomain(NumpyIntegerDomain(), length=length),
                input_metric=self.input_metric_class(AbsoluteDifference()),
                output_measure=self.output_measure,
                measurements=[
                    create_mock_measurement(
                        input_domain=NumpyIntegerDomain(),
                        input_metric=AbsoluteDifference(),
                        output_measure=self.output_measure,
                    )
                ],
            )

    def test_activating_queryable_disables_existing_queryable(self):
        """Activating another queryable should deactivate existing queryable."""
        queryable = self._measurement(self._data)
        # activate first inner queryable
        self.assertEqual(
            queryable(NestedQuery(index=0, get_property=False, inner_query=111)), 456
        )
        self.assertEqual(self._inner_queryable.mock_calls, [call(111)])
        # deactivate it
        self.assertEqual(
            queryable(NestedQuery(index=1, get_property=False, inner_query=111)), 456
        )
        with self.assertRaisesRegex(
            RuntimeError, re.escape("The specified queryable (0) is no longer active")
        ):
            queryable(NestedQuery(index=0, get_property=False, inner_query=111))

    def test_accessing_noninteractive_measurement_doesnt_disable_queryable(self):
        """Retrieving a measurement's answer shouldn't deactivate existing queryable."""
        queryable = self._measurement(self._data)
        # activate queryable
        self.assertEqual(
            queryable(NestedQuery(index=0, get_property=False, inner_query=111)), 456
        )
        self.assertEqual(self._inner_queryable.mock_calls, [call(111)])
        # access measurement's answer
        self.assertEqual(
            queryable(
                NestedQuery(
                    index=2, get_property=False, inner_query="this should be ignored"
                )
            ),
            123,
        )
        # can still access queryable
        self.assertEqual(
            queryable(NestedQuery(index=0, get_property=False, inner_query=111)), 456
        )

    def test_accessing_inner_property(self):
        """Test accessing a property from an inner queryable."""
        queryable = self._measurement(self._data)
        self._inner_queryable.some_property = 17
        self.assertEqual(
            queryable(
                NestedQuery(index=0, get_property=True, inner_query="some_property")
            ),
            17,
        )

    @parameterized.expand([(-1,), (3,)])
    def test_bad_index(self, index: int):
        """Test that bad indexes raise an error."""
        queryable = self._measurement(self._data)
        with self.assertRaises(IndexError):
            queryable(NestedQuery(index=index, get_property=False, inner_query=123))


@parameterized_class([{"output_measure": PureDP()}, {"output_measure": RhoZCDP()}])
class TestCreateAdaptiveComposition(unittest.TestCase):
    """Tests for create_adaptive_composition fucntion.

    Tests :func:`~tmlt.core.measurements.composition.create_adaptive_composition`.
    """

    output_measure: Union[PureDP, RhoZCDP]
    """The output measure to use in the tests."""

    def setUp(self):
        """Set up class."""
        self._data = np.int64(10)
        self._measurement = create_adaptive_composition(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            d_in=1,
            privacy_budget=6,
        )
        self._noninteractive_measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            is_interactive=False,
            return_value=123,
            privacy_relation_return_value=True,
        )

        self._inner_queryable = create_autospec(spec=Queryable, instance=True)
        self._inner_queryable.return_value = 456
        self._inner_queryable.some_property = 17
        self._interactive_measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            is_interactive=True,
            return_value=self._inner_queryable,
            privacy_relation_return_value=True,
        )

    @parameterized.expand(get_all_props(SequentialCompositionQueryable))
    def test_queryable_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self._measurement(self._data), prop_name)

    def test_properties(self):
        """The created measurement's properties have the expected values."""
        self.assertEqual(self._measurement.input_domain, NumpyIntegerDomain())
        self.assertEqual(self._measurement.input_metric, AbsoluteDifference())
        self.assertEqual(self._measurement.output_measure, self.output_measure)
        self.assertEqual(self._measurement.is_interactive, True)

    def test_queryable_properties(self):
        """The created queryable's properties have the expected values."""
        queryable = self._measurement(self._data)
        self.assertEqual(queryable.input_domain, NumpyIntegerDomain())
        self.assertEqual(queryable.input_metric, AbsoluteDifference())
        self.assertEqual(queryable.output_measure, self.output_measure)
        self.assertEqual(queryable.d_in, 1)
        self.assertEqual(queryable.privacy_budget, 6)
        self.assertEqual(queryable.remaining_budget, 6)

    def test_noninteractive_measurements(self):
        """Noninteractive measurements are answered correctly."""
        queryable = self._measurement(self._data)
        query = MeasurementQuery(measurement=self._noninteractive_measurement, d_out=2)
        self.assertEqual(queryable(query), 123)
        self.assertEqual(queryable.remaining_budget, 4)

    def test_interactive_measurements(self):
        """Interactive measurements are answered correctly.

        Tests both normal queries and accessing properties on the returned queryable.
        """
        queryable = self._measurement(self._data)
        query = MeasurementQuery(measurement=self._interactive_measurement, d_out=2)
        self._inner_queryable.some_property = 17
        output_queryable = queryable(query)

        self.assertEqual(queryable.remaining_budget, 4)
        self.assertEqual(output_queryable.some_property, 17)
        self.assertEqual(output_queryable(42), 456)
        self.assertEqual(self._inner_queryable.mock_calls, [call(42)])

    def test_nested_composition(self):
        """Composition can be nested."""
        queryable = self._measurement(self._data)
        # Create the nested sequential composition
        query = MeasurementQuery(measurement=self._measurement, d_out=6)
        new_queryable = queryable(query)
        # Send a measurement to the inner sequential composition queryable
        new_query = MeasurementQuery(
            measurement=self._noninteractive_measurement, d_out=2
        )
        return_value = new_queryable(new_query)
        self.assertEqual(return_value, 123)
        self.assertEqual(queryable.remaining_budget, 0)
        # Check the remaining budget of the inner sequential composition queryable
        self.assertEqual(new_queryable.remaining_budget, 4)


@parameterized_class(
    [
        {"output_measure": PureDP(), "input_metric_class": SumOf},
        {"output_measure": RhoZCDP(), "input_metric_class": RootSumOfSquared},
    ]
)
class TestUnpackParallelCompositionQueryable(unittest.TestCase):
    """Tests for unpack_parallel_composition_queryable function.

    Tests
    :func:`~tmlt.core.measurements.composition.unpack_parallel_composition_queryable`.
    """

    output_measure: Union[PureDP, RhoZCDP]
    """The output measure to use in the tests."""
    input_metric_class: Union[Type[SumOf], Type[RootSumOfSquared]]
    """The class for the input metric to use in the tests."""

    def setUp(self):
        """Set up class."""
        self._data = [np.int64(10)] * 3
        self._noninteractive_measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            is_interactive=False,
            return_value=123,
            privacy_relation_return_value=True,
        )

        self._inner_queryable = create_autospec(spec=Queryable, instance=True)
        self._inner_queryable.return_value = 456
        self._inner_queryable.some_property = 17
        self._interactive_measurement = create_mock_measurement(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            is_interactive=True,
            return_value=self._inner_queryable,
            privacy_relation_return_value=True,
        )
        self._measurement = ParallelComposition(
            input_domain=ListDomain(NumpyIntegerDomain(), length=3),
            input_metric=self.input_metric_class(AbsoluteDifference()),
            output_measure=self.output_measure,
            measurements=[
                self._interactive_measurement,
                self._interactive_measurement,
                self._noninteractive_measurement,
            ],
        )

    def test_outputs(self):
        """Noninteractive measurements are answered correctly."""
        queryable = self._measurement(self._data)
        (
            queryable1,
            queryable2,
            noninteractive_answer,
        ) = unpack_parallel_composition_queryable(queryable)
        for output_queryable in [queryable1, queryable2]:
            self.assertEqual(output_queryable.some_property, 17)
            self.assertEqual(output_queryable(42), 456)
        self.assertEqual(noninteractive_answer, 123)
