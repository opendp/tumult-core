"""Unit tests for :mod:`~tmlt.core.measurements.interactive_measurements`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use

from typing import Any, List, Optional, Type, Union
from unittest import TestCase
from unittest.mock import ANY, MagicMock, Mock, patch

import numpy as np
from parameterized import parameterized, parameterized_class

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.interactive_measurements import (
    DecoratedQueryable,
    DecorateQueryable,
    GetAnswerQueryable,
    IndexQuery,
    MakeInteractive,
    MeasurementQuery,
    ParallelComposition,
    ParallelQueryable,
    PrivacyAccountant,
    PrivacyAccountantState,
    Queryable,
    RetirableQueryable,
    RetireQuery,
    SequentialComposition,
    SequentialQueryable,
    TransformationQuery,
    create_adaptive_composition,
)
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    DictMetric,
    HammingDistance,
    Metric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    create_mock_measurement,
    create_mock_queryable,
    create_mock_transformation,
    get_all_props,
)


@parameterized_class([{"output_measure": PureDP()}, {"output_measure": RhoZCDP()}])
class TestSequentialComposition(PySparkTest):
    """Tests for :class:`~.SequentialComposition`."""

    output_measure: Union[PureDP, RhoZCDP]
    """The output measure to use during the tests."""

    def setUp(self):
        """Set up class."""
        self.data = np.int64(10)
        self.measurement = SequentialComposition(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=self.output_measure,
            d_in=1,
            privacy_budget=6,
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        d_in = {"A": 2}
        measurement = SequentialComposition(
            input_domain=DictDomain({"A": NumpyIntegerDomain()}),
            input_metric=DictMetric({"A": AbsoluteDifference()}),
            output_measure=self.output_measure,
            d_in=d_in,
            privacy_budget=6,
        )
        d_in["A"] = 1
        d_in["B"] = 2
        self.assertDictEqual(measurement.d_in, {"A": 2})

    @parameterized.expand(get_all_props(SequentialComposition))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = SequentialComposition(
            input_domain=DictDomain({"A": NumpyIntegerDomain()}),
            input_metric=DictMetric({"A": AbsoluteDifference()}),
            output_measure=self.output_measure,
            d_in={"A": 1},
            privacy_budget=6,
        )
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """SequentialComposition's properties have the expected values."""
        self.assertEqual(self.measurement.input_domain, NumpyIntegerDomain())
        self.assertEqual(self.measurement.input_metric, AbsoluteDifference())
        self.assertEqual(self.measurement.output_measure, self.output_measure)
        self.assertEqual(self.measurement.is_interactive, True)
        self.assertEqual(self.measurement.d_in, 1)
        self.assertEqual(self.measurement.privacy_budget, 6)

    def test_privacy_function(self):
        """SequentialComposition's privacy function is correct."""
        self.assertEqual(self.measurement.privacy_function(1), 6)

    def test_privacy_function_invalid_d_in(self):
        """SequentialComposition's privacy function raises error if d_in is invalid."""
        with self.assertRaisesRegex(ValueError, "d_in must be <= 1, not 2"):
            self.measurement.privacy_function(2)

    def test_correctness(self):
        """SequentialComposition returns the expected Queryable object."""
        # pylint: disable=protected-access
        actual = self.measurement(self.data)
        self.assertIsInstance(actual, SequentialQueryable)
        self.assertEqual(actual._input_domain, self.measurement.input_domain)
        self.assertEqual(actual._input_metric, self.measurement.input_metric)
        self.assertEqual(actual._output_measure, self.measurement.output_measure)
        self.assertEqual(actual._remaining_budget, self.measurement.privacy_budget)
        self.assertEqual(actual._data, self.data)


class TestParallelComposition(PySparkTest):
    """Tests for :class:`~.ParallelComposition`."""

    def setUp(self):
        """Test setup."""
        self.input_metric = SumOf(AbsoluteDifference())
        self.input_domain = ListDomain(NumpyIntegerDomain(), length=2)
        self.output_measure = PureDP()
        self.composed_measurements = [
            create_mock_measurement(
                output_measure=self.output_measure,
                is_interactive=True,
                privacy_function_implemented=True,
                privacy_function_return_value=ExactNumber(6),
            )
            for _ in range(2)
        ]
        self.measurement = ParallelComposition(
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            output_measure=self.output_measure,
            measurements=self.composed_measurements,
        )
        self.data = [np.int64(10), np.int64(30)]

    @parameterized.expand(get_all_props(ParallelComposition))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.measurement, prop_name)

    @parameterized.expand(
        [
            (
                ListDomain(NumpyIntegerDomain(), length=1),
                SumOf(AbsoluteDifference()),
                PureDP(),
                [create_mock_measurement(is_interactive=False)],
                "All measurements must be interactive",
            ),
            (
                ListDomain(NumpyIntegerDomain(), length=2),
                SumOf(AbsoluteDifference()),
                PureDP(),
                [create_mock_measurement(is_interactive=True)],
                r"Length of input domain \(2\) does not match the number of"
                r" measurements \(1\)",
            ),
            (
                ListDomain(NumpyIntegerDomain(), length=1),
                SumOf(AbsoluteDifference()),
                RhoZCDP(),
                [create_mock_measurement(is_interactive=True)],
                f"Input metric {SumOf} is incompatible with output measure {RhoZCDP}",
            ),
            (
                ListDomain(NumpyIntegerDomain(), length=1),
                RootSumOfSquared(AbsoluteDifference()),
                RhoZCDP(),
                [
                    create_mock_measurement(
                        is_interactive=True,
                        output_measure=RhoZCDP(),
                        input_metric=SymmetricDifference(),
                    )
                ],
                "Input metric for each supplied measurement must match inner metric of"
                " input metric for ParallelComposition",
            ),
            (
                ListDomain(NumpyIntegerDomain(), length=1),
                RootSumOfSquared(AbsoluteDifference()),
                RhoZCDP(),
                [create_mock_measurement(is_interactive=True, output_measure=PureDP())],
                "Output measure for each supplied measurement must match output measure"
                " for ParallelComposition",
            ),
        ]
    )
    def test_invalid_constructor_arguments(
        self,
        input_domain: ListDomain,
        input_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
        measurements: List[Measurement],
        expected_error_message: str,
    ):
        """ParallelComposition constructor raises appropriate error on invalid args."""
        with self.assertRaisesRegex(ValueError, expected_error_message):
            ParallelComposition(
                input_domain=input_domain,
                input_metric=input_metric,
                output_measure=output_measure,
                measurements=measurements,
            )

    def test_properties(self):
        """SequentialComposition's properties have the expected values."""
        self.assertEqual(self.measurement.input_domain, self.input_domain)
        self.assertEqual(self.measurement.input_metric, self.input_metric)
        self.assertEqual(self.measurement.output_measure, self.output_measure)
        self.assertEqual(self.measurement.is_interactive, True)

    def test_privacy_function(self):
        """SequentialComposition's privacy function is correct."""
        self.assertEqual(self.measurement.privacy_function(1), ExactNumber(6))

    def test_correctness(self):
        """SequentialComposition returns the expected Queryable object."""
        # pylint: disable=protected-access
        actual = self.measurement(self.data)
        self.assertIsInstance(actual, ParallelQueryable)
        self.assertEqual(actual._next_index, 0)
        self.assertEqual(actual._data, self.data)
        self.assertEqual(actual._measurements, self.composed_measurements)


class TestMakeInteractive(PySparkTest):
    """Tests for :class:`~.MakeInteractive`."""

    def test_interactive_measurement_raises_error(self):
        """MakeInteractive can not be constructed with an interactive measurement."""
        with self.assertRaisesRegex(ValueError, "Measurement must be non-interactive"):
            MakeInteractive(measurement=create_mock_measurement(is_interactive=True))

    @parameterized.expand(get_all_props(MakeInteractive))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = MakeInteractive(measurement=create_mock_measurement())
        assert_property_immutability(measurement, prop_name)

    def test_privacy_function(self):
        """MakeInteractive has correct privacy_function."""
        non_interactive_measurement = create_mock_measurement(
            privacy_function_implemented=True, privacy_function_return_value=20
        )
        actual = MakeInteractive(non_interactive_measurement).privacy_function(d_in=12)
        self.assertEqual(actual, 20)
        non_interactive_measurement.privacy_function.assert_called_once_with(d_in=12)

    def test_correctness(self):
        """MakeInteractive returns expected GetAnswerQueryable."""
        non_interactive_measurement = create_mock_measurement(return_value=2)
        measurement = MakeInteractive(non_interactive_measurement)
        queryable = measurement(None)
        self.assertIsInstance(queryable, GetAnswerQueryable)
        answer = queryable(None)
        self.assertEqual(answer, 2)

    def test_properties(self):
        """MakeInteractive's properties have appropriate values."""
        non_interactive_measurement = create_mock_measurement()
        measurement = MakeInteractive(non_interactive_measurement)
        self.assertEqual(
            measurement.input_domain, non_interactive_measurement.input_domain
        )
        self.assertEqual(
            measurement.input_metric, non_interactive_measurement.input_metric
        )
        self.assertEqual(
            measurement.output_measure, non_interactive_measurement.output_measure
        )
        self.assertTrue(measurement.is_interactive)
        self.assertEqual(measurement.measurement, non_interactive_measurement)


class TestRetirableQueryable(PySparkTest):
    """Tests for :class:`~.RetirableQueryable`."""

    def test_retirable_queryable_returned(self):
        """Returned Queryable is wrapped in RetirableQueryable."""
        returned_queryable = create_mock_queryable()
        queryable = RetirableQueryable(
            create_mock_queryable(return_value=returned_queryable)
        )
        actual = queryable(None)
        self.assertIsInstance(actual, RetirableQueryable)

        self.assertEqual(
            actual._inner_queryable,  # pylint: disable=protected-access
            returned_queryable,
        )

    def test_retire_works_recursively(self):
        """RetireQuery is propagated to all descendants correctly."""
        queryable = RetirableQueryable(
            create_mock_queryable(
                return_value=create_mock_queryable(return_value=create_mock_queryable())
            )
        )
        inner_most_queryable = queryable(None)(None)
        queryable(RetireQuery())
        self.assertTrue(
            inner_most_queryable._is_retired  # pylint: disable=protected-access
        )

    def test_retire_works_when_descendant_is_retired(self):
        """RetirableQueryable can be retired even when a descendant is retired."""
        queryable = RetirableQueryable(
            create_mock_queryable(
                return_value=create_mock_queryable(return_value=create_mock_queryable())
            )
        )
        child_queryable = queryable(None)
        _ = child_queryable(None)
        child_queryable(RetireQuery())  # Retired descendants
        queryable(RetireQuery())

    def test_retired_retirable_raises_error(self):
        """Retired RetirableQueryables raise error when a query is submitted."""
        queryable = RetirableQueryable(create_mock_queryable())
        queryable(RetireQuery())  # Retired descendants
        with self.assertRaisesRegex(ValueError, "Queryable already retired"):
            queryable(None)


class TestSequentialQueryable(PySparkTest):
    """Tests for :class:`~.SequentialQueryable`."""

    def setUp(self):
        """Test setup."""
        self.construct_queryable = lambda budget=10: SequentialQueryable(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            d_in=1,
            output_measure=PureDP(),
            privacy_budget=budget,
            data=np.int64(10),
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        d_in = {"A": 2}
        queryable = SequentialQueryable(
            input_domain=DictDomain({"A": NumpyIntegerDomain()}),
            input_metric=DictMetric({"A": AbsoluteDifference()}),
            d_in=d_in,
            output_measure=PureDP(),
            privacy_budget=10,
            data={"A": np.int64(10)},
        )
        d_in["A"] = 1
        d_in["B"] = 2
        self.assertDictEqual(
            queryable._d_in, {"A": 2}  # pylint: disable=protected-access
        )

    @parameterized.expand([(10, 9), (float("inf"), float("inf"))])
    def test_queryable_budget_is_decreased_correctly(
        self, budget: Any, expected_remaining: Any
    ):
        """SequentialQueryable's internal budget is correctly decreased on query."""
        queryable = self.construct_queryable(budget)
        queryable(
            MeasurementQuery(
                measurement=create_mock_measurement(
                    is_interactive=True,
                    privacy_function_implemented=True,
                    privacy_function_return_value=1,
                    return_value=create_mock_queryable(),
                )
            )
        )
        self.assertEqual(
            queryable._remaining_budget,  # pylint: disable=protected-access
            expected_remaining,
        )

    def test_insufficient_budget_remaining(self):
        """Error is raised when query privacy loss exceeds available budget."""
        queryable = self.construct_queryable()
        queryable(
            MeasurementQuery(
                measurement=create_mock_measurement(
                    is_interactive=True,
                    privacy_function_implemented=True,
                    privacy_function_return_value=1,
                    return_value=create_mock_queryable(),
                )
            )
        )
        with self.assertRaisesRegex(
            ValueError, "Cannot answer query without exceeding available privacy budget"
        ):
            queryable(
                MeasurementQuery(
                    measurement=create_mock_measurement(
                        is_interactive=True,
                        privacy_function_implemented=True,
                        privacy_function_return_value=10,
                    )
                )
            )

    def test_non_interactive_measurement_disallowed(self):
        """SequentialQueryable raises error when measurement is non-interactive."""
        queryable = self.construct_queryable()
        with self.assertRaisesRegex(
            ValueError,
            "SequentialQueryable does not answer non-interactive measurement",
        ):
            queryable(MeasurementQuery(create_mock_measurement()))

    @parameterized.expand(
        [
            (
                NumpyFloatDomain(),
                AbsoluteDifference(),
                PureDP(),
                "Input domain of measurement query does not match the input domain of"
                " SequentialQueryable.",
            ),
            (
                NumpyIntegerDomain(),
                SymmetricDifference(),
                PureDP(),
                "Input metric of measurement query does not match the input metric of"
                " SequentialQueryable.",
            ),
            (
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                RhoZCDP(),
                "Output measure of measurement query does not match the output measure"
                " of SequentialQueryable.",
            ),
        ]
    )
    def test_incompatible_measurement_query(
        self,
        input_domain: Domain,
        input_metric: Metric,
        measure: Measure,
        expected_error_message: str,
    ):
        """SequentialQueryable raises error if query is incompatible."""
        queryable = self.construct_queryable()
        with self.assertRaisesRegex(ValueError, expected_error_message):
            queryable(
                MeasurementQuery(
                    create_mock_measurement(
                        is_interactive=True,
                        input_domain=input_domain,
                        input_metric=input_metric,
                        output_measure=measure,
                    )
                )
            )

    @parameterized.expand([(True,), (False,)])
    def test_measurement_query_with_explicit_d_out(
        self, privacy_function_implemented: bool
    ):
        """SequentialQueryable can run transformations without stability function."""
        queryable = self.construct_queryable()
        queryable(
            MeasurementQuery(
                create_mock_measurement(
                    is_interactive=True,
                    privacy_function_implemented=privacy_function_implemented,
                    privacy_relation_return_value=True,
                    privacy_function_return_value=1,
                    return_value=create_mock_queryable(),
                ),
                d_out=6,
            )
        )
        self.assertEqual(
            queryable._remaining_budget, 4  # pylint: disable=protected-access
        )

    def test_measurement_query_stability_relation_returns_false(self):
        """SequentialQueryable raises error if privacy relation is not True."""
        queryable = self.construct_queryable()
        with self.assertRaisesRegex(
            ValueError,
            "Measurement's privacy relation cannot be satisfied with given d_out"
            r" \(11\)",
        ):
            queryable(
                MeasurementQuery(
                    create_mock_measurement(
                        is_interactive=True,
                        privacy_function_implemented=False,
                        privacy_relation_return_value=False,
                    ),
                    d_out=11,
                )
            )

    @parameterized.expand([(True,), (False,)])
    def test_transformation_query_with_explicit_d_out(
        self, stability_function_implemented: bool
    ):
        """SequentialQueryable can run transformations without stability function."""
        queryable = self.construct_queryable()
        queryable(
            TransformationQuery(
                create_mock_transformation(
                    stability_function_implemented=stability_function_implemented,
                    stability_function_return_value=2,
                    stability_relation_return_value=True,
                ),
                d_out=3,
            )
        )
        self.assertEqual(queryable._d_in, 3)  # pylint: disable=protected-access

    def test_transformation_query_stability_relation_returns_false(self):
        """SequentialQueryable raises error if stability relation is not True."""
        queryable = self.construct_queryable()
        with self.assertRaisesRegex(
            ValueError,
            "Transformation's stability relation cannot be satisfied with given"
            r" d_out \(3\)",
        ):
            queryable(
                TransformationQuery(
                    create_mock_transformation(
                        stability_function_implemented=False,
                        stability_relation_return_value=False,
                    ),
                    d_out=3,
                )
            )

    def test_transformation_query(self):
        """SequentialQueryable processes TransformationQuery correctly."""
        # pylint: disable=protected-access
        queryable = self.construct_queryable()
        transformation = create_mock_transformation(
            return_value=np.float(100.0),
            stability_function_implemented=True,
            output_domain=NumpyFloatDomain(),
            output_metric=SymmetricDifference(),
            stability_function_return_value=20,
        )
        queryable(TransformationQuery(transformation=transformation))
        self.assertEqual(queryable._d_in, 20)
        self.assertEqual(queryable._input_domain, NumpyFloatDomain())
        self.assertEqual(queryable._input_metric, SymmetricDifference())
        self.assertEqual(queryable._data, np.float(100.0))

    @parameterized.expand(
        [
            (
                NumpyIntegerDomain(),
                SymmetricDifference(),
                "Input metric of transformation query does not match the input metric"
                " of SequentialQueryable.",
            ),
            (
                NumpyFloatDomain(),
                AbsoluteDifference(),
                "Input domain of transformation query does not match the input domain"
                " of SequentialQueryable.",
            ),
        ]
    )
    def test_invalid_transformation_query(
        self, input_domain: Domain, input_metric: Metric, error_message: str
    ):
        """SequentialQueryable raises error when TransformationQuery is incompatible."""
        queryable = self.construct_queryable()
        transformation = create_mock_transformation(
            stability_function_implemented=True,
            stability_function_return_value=2,
            input_domain=input_domain,
            input_metric=input_metric,
        )
        with self.assertRaisesRegex(ValueError, error_message):
            queryable(TransformationQuery(transformation=transformation))

    def test_interactive_measurements_return_retirables(self):
        """SequentialQueryable returns a RetirableQueryable when answering queries."""
        queryable = self.construct_queryable()
        answer = queryable(
            MeasurementQuery(
                create_mock_measurement(
                    is_interactive=True,
                    return_value=create_mock_queryable(),
                    privacy_function_implemented=True,
                )
            )
        )
        self.assertIsInstance(answer, RetirableQueryable)

    def test_interleaving_is_disallowed(self):
        """Tests that interleaving is not allowed."""
        queryable = SequentialQueryable(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            d_in=1,
            output_measure=PureDP(),
            privacy_budget=10,
            data=np.int64(10),
        )
        child1_queryable = queryable(
            MeasurementQuery(
                create_mock_measurement(
                    is_interactive=True,
                    privacy_function_implemented=True,
                    return_value=create_mock_queryable(
                        return_value=create_mock_queryable()
                    ),
                )
            )
        )
        grandchild_queryable = child1_queryable(None)
        _ = queryable(
            MeasurementQuery(
                create_mock_measurement(
                    is_interactive=True,
                    return_value=create_mock_queryable(),
                    privacy_function_implemented=True,
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "Queryable already retired"):
            grandchild_queryable(None)


class TestParallelQueryable(PySparkTest):
    """Tests for :class:`~.ParallelQueryable`."""

    def setUp(self):
        """Test setup."""
        self.construct_queryable = lambda: ParallelQueryable(
            data=[np.int64(10), np.int64(1)],
            measurements=[
                create_mock_measurement(
                    is_interactive=True,
                    return_value=create_mock_queryable(
                        return_value=create_mock_queryable()
                    ),
                )
                for _ in range(2)
            ],
        )

    def test_invalid_constructor_arguments(self):
        """ParallelQueryable raises an error when arguments are invalid."""
        with self.assertRaisesRegex(
            ValueError,
            "Length of input data does not match the number of measurements provided",
        ):
            ParallelQueryable(
                data=[np.int64(10), np.int64(1)],
                measurements=[
                    create_mock_measurement(
                        is_interactive=True,
                        return_value=create_mock_queryable(
                            return_value=create_mock_queryable()
                        ),
                    )
                ],
            )

    def test_returned_queryables_are_retirable(self):
        """Index queries are answered with RetirableQueryables."""
        queryable = self.construct_queryable()
        self.assertIsInstance(queryable(IndexQuery(0)), RetirableQueryable)

    def test_sequential_access_enforced(self):
        """ParallelQueryable raises error if access is not in expected order."""
        queryable = self.construct_queryable()
        with self.assertRaisesRegex(ValueError, "Bad Index"):
            queryable(IndexQuery(1))

    def test_interleaving_is_disallowed(self):
        """ParallelQueryable disallows interleaving."""
        queryable = self.construct_queryable()
        child1_queryable = queryable(IndexQuery(0))
        grandchild_queryable = child1_queryable(None)
        _ = queryable(IndexQuery(1))
        with self.assertRaisesRegex(ValueError, "Queryable already retired"):
            grandchild_queryable(None)


class TestGetAnswerQueryable(PySparkTest):
    """Tests for :class:`~.GetAnswerQueryable`."""

    def test_disallows_interactive_measurement(self):
        """GetAnswerQueryable cannot be constructed with an interactive measurement."""
        with self.assertRaisesRegex(ValueError, "Measurement must be non-interactive"):
            GetAnswerQueryable(
                create_mock_measurement(is_interactive=True), data=np.int64(10)
            )

    def test_correctness(self):
        """GetAnswerQueryable returns measurement answer as expected."""
        non_interactive_measurement = create_mock_measurement()
        self.assertEqual(
            GetAnswerQueryable(non_interactive_measurement, np.int64(10))(None),
            non_interactive_measurement(np.int64(10)),
        )


class TestPrivacyAccountant(PySparkTest):
    """Tests for :class:`~.PrivacyAccountant`."""

    def setUp(self):
        """Test Setup."""
        self.data = self.spark.createDataFrame(
            [(1, "X"), (2, "Y"), (3, "Z")], schema=["A", "B"]
        )
        self.measurement = SequentialComposition(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            d_in=1,
            privacy_budget=6,
        )
        self.get_queryable = lambda: self.measurement(self.data)

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        d_in = {"A": 2}
        accountant = PrivacyAccountant(
            queryable=SequentialQueryable(
                input_domain=DictDomain({"A": NumpyIntegerDomain()}),
                input_metric=DictMetric({"A": AbsoluteDifference()}),
                d_in=d_in,
                output_measure=PureDP(),
                privacy_budget=10,
                data={"A": np.int64(10)},
            ),
            input_metric=DictMetric({"A": AbsoluteDifference()}),
            input_domain=DictDomain({"A": NumpyIntegerDomain()}),
            output_measure=PureDP(),
            d_in=d_in,
            privacy_budget=10,
        )
        d_in["A"] = 1
        d_in["B"] = 2
        self.assertDictEqual(accountant.d_in, {"A": 2})

    def test_children_cannot_be_mutated(self):
        """Tests that the list of children of a PrivacyAccountant cannot be mutated."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        children = accountant.split(
            splitting_transformation=create_mock_transformation(
                input_domain=self.measurement.input_domain,
                input_metric=self.measurement.input_metric,
                output_domain=ListDomain(NumpyIntegerDomain(), length=4),
                output_metric=SumOf(AbsoluteDifference()),
                return_value=[np.int64(0) for _ in range(4)],
                stability_function_implemented=True,
            ),
            privacy_budget=2,
        )
        self.assertEqual(children, accountant.children)
        children.append(accountant)
        self.assertNotEqual(children, accountant.children)

        children = accountant.children
        self.assertEqual(children, accountant.children)
        children.append(accountant)
        self.assertNotEqual(children, accountant.children)

    @parameterized.expand(
        [
            (
                None,
                None,
                "PrivacyAccountant cannot be initialized with no parent and no"
                " queryable",
            ),
            (
                Mock(PrivacyAccountant),
                Mock(SequentialQueryable),
                "PrivacyAccountant can be initialized with only parent or only"
                " queryable but not both",
            ),
        ]
    )
    def test_init_invalid_arguments(
        self,
        parent: Optional[PrivacyAccountant],
        queryable: Optional[Queryable],
        error_message: str,
    ):
        """PrivacyAccountant.__init__ raises error when arguments are invalid."""
        with self.assertRaisesRegex(ValueError, error_message):
            PrivacyAccountant(
                queryable=queryable,
                parent=parent,
                input_metric=AbsoluteDifference(),
                input_domain=NumpyIntegerDomain(),
                output_measure=PureDP(),
                d_in=1,
                privacy_budget=1,
            )

    @patch.object(PrivacyAccountant, "__init__", autospec=True, return_value=None)
    def test_launch(self, mock_accountant_init):
        """PrivacyAccountant.launch works as expected."""

        mock_queryable = Mock(spec=SequentialQueryable)
        mock_sequential_composition = Mock(
            spec=SequentialComposition, return_value=mock_queryable
        )
        mock_sequential_composition.d_in = 1
        mock_sequential_composition.privacy_budget = 10
        mock_sequential_composition.input_domain = NumpyIntegerDomain()
        mock_sequential_composition.input_metric = AbsoluteDifference()
        mock_sequential_composition.output_measure = PureDP()

        PrivacyAccountant.launch(
            measurement=mock_sequential_composition, data=self.data
        )
        mock_accountant_init.assert_called_with(
            self=ANY,
            queryable=mock_queryable,
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=PureDP(),
            d_in=1,
            privacy_budget=10,
        )

    def test_privacy_accountant_properties(self):
        """PrivacyAccountant's properties have expected values."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        self.assertEqual(accountant.state, PrivacyAccountantState.ACTIVE)
        self.assertEqual(accountant.input_domain, self.measurement.input_domain)
        self.assertEqual(accountant.input_metric, self.measurement.input_metric)
        self.assertEqual(accountant.output_measure, self.measurement.output_measure)
        self.assertEqual(accountant.privacy_budget, self.measurement.privacy_budget)
        self.assertEqual(accountant.d_in, self.measurement.d_in)
        self.assertEqual(accountant.children, [])

    @parameterized.expand(
        [
            (
                "Transformation's input domain does not match PrivacyAccountant's input"
                " domain",
                create_mock_transformation(
                    input_domain=SparkDataFrameDomain(
                        {"A": SparkIntegerColumnDescriptor()}
                    ),
                    input_metric=SymmetricDifference(),
                    stability_function_implemented=True,
                ),
            ),
            (
                "Transformation's input metric does not match PrivacyAccountant's input"
                " metric",
                create_mock_transformation(
                    input_domain=SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        }
                    ),
                    input_metric=HammingDistance(),
                    stability_function_implemented=True,
                ),
            ),
        ]
    )
    def test_transform_in_place_invalid_arguments(
        self,
        error_message: str,
        transformation: Transformation,
        d_out: Optional[Any] = None,
    ):
        """PrivacyAccountant raises error transformation can not be applied."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        with self.assertRaisesRegex(ValueError, error_message):
            accountant.transform_in_place(transformation=transformation, d_out=d_out)

    def test_transform_in_place(self):
        """PrivacyAccountant.transform_in_place works correctly."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=NumpyIntegerDomain(),
            output_metric=AbsoluteDifference(),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=np.int64(2),
        )
        accountant.transform_in_place(transformation=transformation)
        self.assertEqual(accountant.input_domain, NumpyIntegerDomain())
        self.assertEqual(accountant.input_metric, AbsoluteDifference())
        self.assertEqual(accountant.d_in, 10)
        # pylint: disable=protected-access
        self.assertEqual(accountant._queryable._data, np.int64(2))

    def test_transform_with_explicit_d_out(self):
        """PrivacyAccountant.transform_in_place works with a d_out provided."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=NumpyIntegerDomain(),
            output_metric=AbsoluteDifference(),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=np.int64(2),
        )
        accountant.transform_in_place(transformation=transformation, d_out=10)
        self.assertEqual(accountant.input_domain, NumpyIntegerDomain())
        self.assertEqual(accountant.input_metric, AbsoluteDifference())
        self.assertEqual(accountant.d_in, 10)
        # pylint: disable=protected-access
        self.assertEqual(accountant._queryable._data, np.int64(2))

    @parameterized.expand(
        [
            (
                "Measurement's output measure does not match PrivacyAccountant's output"
                " measure",
                create_mock_measurement(
                    input_domain=SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        }
                    ),
                    input_metric=SymmetricDifference(),
                    output_measure=RhoZCDP(),
                    privacy_function_implemented=True,
                    privacy_function_return_value=5,
                ),
            )
        ]
    )
    def test_measure_invalid_arguments(
        self,
        error_message: str,
        measurement: Measurement,
        error_type: Type = ValueError,
    ):
        """PrivacyAccountant raises error when a measurement cannot be answered."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        with self.assertRaisesRegex(error_type, error_message):
            accountant.measure(measurement=measurement)

    def test_measure(self):
        """PrivacyAccountant.measure answers measurement correctly."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        starting_privacy_budget = accountant.privacy_budget
        mock_measurement = create_mock_measurement(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            privacy_function_implemented=True,
            privacy_function_return_value=5,
            return_value=np.int64(2),
        )
        actual_answer = accountant.measure(measurement=mock_measurement)
        self.assertEqual(actual_answer, np.int64(2))
        self.assertEqual(accountant.privacy_budget, starting_privacy_budget - 5)

    def test_measure_consumes_all_privacy_budget(self):
        """Test that measurements are allowed to consume the entire budget."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        mock_measurement = create_mock_measurement(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            privacy_function_implemented=True,
            privacy_function_return_value=accountant.privacy_budget,
            return_value=np.int64(2),
        )
        actual_answer = accountant.measure(measurement=mock_measurement)
        self.assertEqual(actual_answer, np.int64(2))
        self.assertEqual(accountant.privacy_budget, ExactNumber(0))

    @parameterized.expand(
        [
            (
                "Given d_out does not satisfy the stability relation of given",
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                ListDomain(NumpyIntegerDomain(), length=2),
                SumOf(AbsoluteDifference()),
                PureDP(),
                8,
                False,
                1,
            ),
            (
                "Transformation's input domain does not match PrivacyAccountant's input"
                " domain",
                NumpyStringDomain(),
                AbsoluteDifference(),
                ListDomain(NumpyIntegerDomain(), length=2),
                SumOf(AbsoluteDifference()),
                PureDP(),
                8,
            ),
            (
                "Transformation's input metric does not match PrivacyAccountant's"
                " input metric",
                NumpyIntegerDomain(),
                SymmetricDifference(),
                ListDomain(NumpyIntegerDomain(), length=2),
                SumOf(AbsoluteDifference()),
                PureDP(),
                8,
            ),
            (
                "Splitting transformation's output domain must be ListDomain.",
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                NumpyIntegerDomain(),
                SumOf(AbsoluteDifference()),
                PureDP(),
                8,
            ),
            (
                "Splitting transformation's output domain must specify list length.",
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                ListDomain(NumpyIntegerDomain()),
                SumOf(AbsoluteDifference()),
                PureDP(),
                8,
            ),
            (
                "Splitting transformation's output metric must be"
                f" {SumOf} for output measure PureDP",
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                ListDomain(NumpyIntegerDomain(), length=2),
                RootSumOfSquared(AbsoluteDifference()),
                PureDP(),
                8,
            ),
            (
                "Splitting transformation's output metric must be"
                f" {RootSumOfSquared} for output measure RhoZCDP",
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                ListDomain(NumpyIntegerDomain(), length=2),
                SumOf(AbsoluteDifference()),
                RhoZCDP(),
                8,
            ),
            (
                r"PrivacyAccountant's privacy budget \(10\) is insufficient",
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                ListDomain(NumpyIntegerDomain(), length=2),
                SumOf(AbsoluteDifference()),
                PureDP(),
                11,
            ),
        ]
    )
    def test_split_invalid_arguments(
        self,
        error_message: str,
        input_domain: Domain,
        input_metric: Metric,
        output_domain: Domain,
        output_metric: Metric,
        output_measure: Union[PureDP, RhoZCDP],
        privacy_budget: ExactNumberInput,
        stability_function_implemented: bool = True,
        d_out: Optional[Any] = None,
    ):
        """PrivacyAccountant.split raises errors appropriately."""
        accountant = PrivacyAccountant.launch(
            measurement=SequentialComposition(
                input_domain=NumpyIntegerDomain(),
                input_metric=AbsoluteDifference(),
                output_measure=output_measure,
                d_in=1,
                privacy_budget=10,
            ),
            data=np.int64(10),
        )
        splitting_transformation = create_mock_transformation(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
            stability_function_implemented=stability_function_implemented,
            stability_function_return_value=5,
            stability_relation_return_value=False,
            return_value=[np.int64(2), np.int64(3)],
        )
        with self.assertRaisesRegex(ValueError, error_message):
            accountant.split(
                splitting_transformation=splitting_transformation,
                d_out=d_out,
                privacy_budget=privacy_budget,
            )

    def test_split(self):
        """PrivacyAccountant.split works correctly."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        initial_budget = accountant.privacy_budget
        transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=ListDomain(element_domain=NumpyIntegerDomain(), length=2),
            output_metric=SumOf(AbsoluteDifference()),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=[np.int64(2), np.int64(3)],
        )
        split_budget = 2
        child_accountants = accountant.split(
            splitting_transformation=transformation, privacy_budget=split_budget
        )
        self.assertEqual(accountant.privacy_budget, initial_budget - split_budget)
        self.assertEqual(accountant.state, PrivacyAccountantState.WAITING_FOR_CHILDREN)
        self.assertEqual(len(child_accountants), 2)
        for index, child in enumerate(child_accountants):
            self.assertEqual(
                child.input_domain, transformation.output_domain.element_domain
            )
            self.assertEqual(
                child.input_metric, transformation.output_metric.inner_metric
            )
            self.assertEqual(child.output_measure, accountant.output_measure)
            self.assertEqual(child.d_in, 10)
            self.assertEqual(child.privacy_budget, 2)
            expected_state = (
                PrivacyAccountantState.ACTIVE
                if index == 0
                else PrivacyAccountantState.WAITING_FOR_SIBLING
            )
            self.assertEqual(child.state, expected_state)

    def test_queue_transformation_on_active_accountant(self):
        """queue_transformation runs immediately on active accountant"""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=NumpyIntegerDomain(),
            output_metric=AbsoluteDifference(),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=np.int64(2),
        )
        accountant.queue_transformation(transformation=transformation)
        self.assertEqual(accountant.input_domain, NumpyIntegerDomain())
        self.assertEqual(accountant.input_metric, AbsoluteDifference())
        self.assertEqual(accountant.d_in, 10)
        # pylint: disable=protected-access
        self.assertEqual(accountant._queryable._data, np.int64(2))
        self.assertIsNone(accountant._pending_transformation)

    def test_queue_transformation_on_inactive_accountant(self):
        """queue_transformation queues transformations on inactive account"""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        split_transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=ListDomain(element_domain=NumpyIntegerDomain(), length=2),
            output_metric=SumOf(AbsoluteDifference()),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=[np.int64(2), np.int64(3)],
        )
        split_budget = 2
        child_accountants = accountant.split(
            splitting_transformation=split_transformation, privacy_budget=split_budget
        )
        transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=NumpyIntegerDomain(),
            output_metric=AbsoluteDifference(),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=np.int64(2),
        )
        accountant.queue_transformation(transformation=transformation)
        # values should reflect the pending transformation
        self.assertEqual(accountant.input_domain, NumpyIntegerDomain())
        self.assertEqual(accountant.input_metric, AbsoluteDifference())
        self.assertEqual(accountant.d_in, 10)
        self.assertIsNotNone(
            accountant._pending_transformation  # pylint: disable=protected-access
        )

        for c in child_accountants:
            c.retire()

        # Once the accountant is active again, the transformation should
        # have been run
        self.assertEqual(accountant.state, PrivacyAccountantState.ACTIVE)
        self.assertEqual(accountant.input_domain, NumpyIntegerDomain())
        self.assertEqual(accountant.input_metric, AbsoluteDifference())
        self.assertEqual(accountant.d_in, 10)
        # pylint: disable=protected-access
        self.assertEqual(accountant._queryable._data, np.int64(2))
        self.assertIsNone(accountant._pending_transformation)

    @parameterized.expand(
        [
            (
                "Transformation's input domain does not match PrivacyAccountant's input"
                " domain",
                create_mock_transformation(
                    input_domain=SparkDataFrameDomain(
                        {"A": SparkIntegerColumnDescriptor()}
                    ),
                    input_metric=SymmetricDifference(),
                    stability_function_implemented=True,
                ),
            ),
            (
                "Transformation's input metric does not match PrivacyAccountant's input"
                " metric",
                create_mock_transformation(
                    input_domain=SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        }
                    ),
                    input_metric=HammingDistance(),
                    stability_function_implemented=True,
                ),
            ),
        ]
    )
    def test_queue_transformation_invalid_arguments(
        self,
        error_message: str,
        transformation: Transformation,
        d_out: Optional[Any] = None,
    ) -> None:
        """Test queue_transformation with an invalid transformation."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        split_transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=ListDomain(element_domain=NumpyIntegerDomain(), length=2),
            output_metric=SumOf(AbsoluteDifference()),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=[np.int64(2), np.int64(3)],
        )
        split_budget = 2
        accountant.split(
            splitting_transformation=split_transformation, privacy_budget=split_budget
        )
        self.assertEqual(accountant.state, PrivacyAccountantState.WAITING_FOR_CHILDREN)
        with self.assertRaisesRegex(ValueError, error_message):
            accountant.queue_transformation(transformation=transformation, d_out=d_out)

    @parameterized.expand(
        [
            (
                "Transformation's input domain does not match the output domain"
                " of the last transformation",
                create_mock_transformation(
                    input_domain=SparkDataFrameDomain(
                        {"A": SparkIntegerColumnDescriptor()}
                    ),
                    input_metric=SymmetricDifference(),
                    stability_function_implemented=True,
                ),
            ),
            (
                "Transformation's input metric does not match the output metric"
                " of the last transformation",
                create_mock_transformation(
                    input_domain=SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        }
                    ),
                    input_metric=HammingDistance(),
                    stability_function_implemented=True,
                ),
            ),
        ]
    )
    def test_queue_invalid_transformation_with_transform_in_queue(
        self,
        error_message: str,
        transformation: Transformation,
        d_out: Optional[Any] = None,
    ):
        """Test queue_transformation with invalid arguments and a pending transform."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        split_transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=ListDomain(element_domain=NumpyIntegerDomain(), length=2),
            output_metric=SumOf(AbsoluteDifference()),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=[np.int64(2), np.int64(3)],
        )
        split_budget = 2
        accountant.split(
            splitting_transformation=split_transformation, privacy_budget=split_budget
        )
        self.assertEqual(accountant.state, PrivacyAccountantState.WAITING_FOR_CHILDREN)

        identity_transformation = create_mock_transformation(
            input_domain=accountant.input_domain,
            input_metric=accountant.input_metric,
            output_domain=accountant.input_domain,
            output_metric=accountant.input_metric,
            stability_function_implemented=True,
        )
        accountant.queue_transformation(identity_transformation)
        self.assertIsNotNone(
            accountant._pending_transformation  # pylint: disable=protected-access
        )
        with self.assertRaisesRegex(ValueError, error_message):
            accountant.queue_transformation(transformation=transformation, d_out=d_out)

    def test_force_activate_raises_error_on_invalid_states(self):
        """PrivacyAccountant.force_activate raises error appropriately."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        accountant._state = (  # pylint: disable=protected-access
            PrivacyAccountantState.RETIRED
        )
        with self.assertRaisesRegex(
            RuntimeError, "Can not activate RETIRED PrivacyAccountant"
        ):
            accountant.force_activate()

    def test_force_activate_active(self):
        """PrivacyAccountant.force_activate works when ACTIVE."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        accountant.force_activate()
        self.assertEqual(accountant.state, PrivacyAccountantState.ACTIVE)

    def test_force_activate_waiting_for_siblings(self):
        """PrivacyAccountant.force_activate works when WAITING_FOR_SIBLINGS."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        transformation = create_mock_transformation(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            output_domain=ListDomain(element_domain=NumpyIntegerDomain(), length=4),
            output_metric=SumOf(AbsoluteDifference()),
            stability_function_implemented=True,
            stability_function_return_value=10,
            return_value=[np.int64(2), np.int64(3), np.int64(4), np.int64(5)],
        )
        child_accountants = accountant.split(
            splitting_transformation=transformation, privacy_budget=2
        )

        child_accountants[2].force_activate()

        for i in range(0, 2):
            self.assertEqual(child_accountants[i].state, PrivacyAccountantState.RETIRED)

        self.assertEqual(child_accountants[2].state, PrivacyAccountantState.ACTIVE)
        self.assertEqual(
            child_accountants[3].state, PrivacyAccountantState.WAITING_FOR_SIBLING
        )
        self.assertEqual(accountant.state, PrivacyAccountantState.WAITING_FOR_CHILDREN)

    def test_retire_raises_error_appropriately(self):
        """PrivacyAccountant.retire raises error appropriately."""
        # pylint: disable=protected-access
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        accountant._state = PrivacyAccountantState.WAITING_FOR_CHILDREN
        with self.assertRaisesRegex(
            RuntimeError,
            "Can not retire PrivacyAccountant in WAITING_FOR_CHILDREN state",
        ):
            accountant.retire()

    def test_retire_raises_warning_appropriately(self):
        """PrivacyAccountant.retire raises warning appropriately."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        children = accountant.split(
            splitting_transformation=create_mock_transformation(
                input_domain=self.measurement.input_domain,
                input_metric=self.measurement.input_metric,
                output_domain=ListDomain(NumpyIntegerDomain(), length=4),
                output_metric=SumOf(AbsoluteDifference()),
                return_value=[np.int64(0) for _ in range(4)],
                stability_function_implemented=True,
            ),
            privacy_budget=2,
        )
        with self.assertWarnsRegex(
            RuntimeWarning,
            "Retiring an unused PrivacyAccountant that is"
            " PrivacyAccountantState.WAITING_FOR_SIBLING",
        ):
            children[1].retire()

    def test_retire_accountant_waiting_for_sibling(self):
        """Retiring an unused sibling retires all prior siblings and activates next."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        children = accountant.split(
            splitting_transformation=create_mock_transformation(
                input_domain=self.measurement.input_domain,
                input_metric=self.measurement.input_metric,
                output_domain=ListDomain(NumpyIntegerDomain(), length=4),
                output_metric=SumOf(AbsoluteDifference()),
                return_value=[np.int64(0) for _ in range(4)],
                stability_function_implemented=True,
            ),
            privacy_budget=2,
        )
        assert len(children) == 4
        assert children[0].state == PrivacyAccountantState.ACTIVE
        assert (
            children[1].state
            == children[2].state
            == PrivacyAccountantState.WAITING_FOR_SIBLING
        )
        children[2].retire()
        self.assertTrue(
            all(
                child.state == PrivacyAccountantState.RETIRED for child in children[:-1]
            )
        )
        self.assertEqual(children[3].state, PrivacyAccountantState.ACTIVE)

    def test_retiring_all_children_activates_parent(self):
        """PrivacyAccount transitions from WAITING_FOR_CHILDREN to ACTIVE correctly."""
        accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )
        children = accountant.split(
            splitting_transformation=create_mock_transformation(
                input_domain=self.measurement.input_domain,
                input_metric=self.measurement.input_metric,
                output_domain=ListDomain(NumpyIntegerDomain(), length=2),
                output_metric=SumOf(AbsoluteDifference()),
                return_value=[np.int64(0) for _ in range(2)],
                stability_function_implemented=True,
            ),
            privacy_budget=2,
        )
        assert accountant.state == PrivacyAccountantState.WAITING_FOR_CHILDREN
        for child in children:
            child.retire()

        self.assertEqual(accountant.state, PrivacyAccountantState.ACTIVE)


class TestDecorateQueryable(TestCase):
    """Tests for DecorateQueryable."""

    def setUp(self):
        """Set up class."""
        self.mock_queryable = create_mock_queryable()
        self.measurement = DecorateQueryable(
            measurement=create_mock_measurement(
                input_domain=NumpyIntegerDomain(),
                input_metric=AbsoluteDifference(),
                output_measure=PureDP(),
                is_interactive=True,
                privacy_function_implemented=True,
                privacy_function_return_value=6,
                return_value=self.mock_queryable,
            ),
            preprocess_query=lambda x: x,
            postprocess_answer=lambda x: x,
        )

    @parameterized.expand(get_all_props(DecorateQueryable))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.measurement, prop_name)

    def test_properties(self):
        """SequentialComposition's properties have the expected values."""
        self.assertEqual(self.measurement.input_domain, NumpyIntegerDomain())
        self.assertEqual(self.measurement.input_metric, AbsoluteDifference())
        self.assertEqual(self.measurement.output_measure, PureDP())
        self.assertEqual(self.measurement.is_interactive, True)

    def test_privacy_function(self):
        """SequentialComposition's privacy function is correct."""
        self.assertEqual(self.measurement.privacy_function(1), 6)

    def test_correctness(self):
        """SequentialComposition returns the expected Queryable object."""
        # pylint: disable=protected-access
        actual = self.measurement(np.int64(10))
        self.assertIsInstance(actual, DecoratedQueryable)
        self.assertEqual(actual._preprocess_query, self.measurement.preprocess_query)
        self.assertEqual(
            actual._postprocess_answer, self.measurement.postprocess_answer
        )
        self.assertEqual(actual._queryable, self.mock_queryable)


class TestDecoratedQueryable(TestCase):
    """Tests for DecoratedQueryable."""

    def test_correctness(self):
        """Set up class."""
        mock_queryable = MagicMock()
        mock_queryable.return_value = 1
        decorated_queryable = DecorateQueryable(
            measurement=create_mock_measurement(
                input_domain=NumpyIntegerDomain(),
                input_metric=AbsoluteDifference(),
                output_measure=PureDP(),
                is_interactive=True,
                privacy_function_implemented=True,
                privacy_function_return_value=6,
                return_value=mock_queryable,
            ),
            preprocess_query=lambda x: 2 * x,
            postprocess_answer=lambda x: x + 1,
        )(np.int64(1))
        actual = decorated_queryable(2)
        mock_queryable.assert_called_once_with(4)
        self.assertEqual(actual, 2)


class TestCreateAdaptiveComposition(TestCase):
    """Tests for :func:`~.create_adaptive_composition`."""

    def setUp(self):
        """Test setup."""
        self.queryable = create_adaptive_composition(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            d_in=1,
            privacy_budget=2,
            output_measure=PureDP(),
        )(np.int64(10))

    def test_create_adaptive_composition(self):
        """:func:`~.create_adaptive_composition` works as expected."""
        adaptive_composition = create_adaptive_composition(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            d_in=1,
            privacy_budget=2,
            output_measure=PureDP(),
        )
        self.assertIsInstance(adaptive_composition, DecorateQueryable)
        self.assertIsInstance(adaptive_composition.measurement, SequentialComposition)
        self.assertEqual(
            adaptive_composition.measurement.input_domain, NumpyIntegerDomain()
        )
        self.assertEqual(
            adaptive_composition.measurement.input_metric, AbsoluteDifference()
        )
        self.assertEqual(adaptive_composition.measurement.d_in, 1)
        self.assertEqual(adaptive_composition.measurement.privacy_budget, 2)
        self.assertEqual(adaptive_composition.measurement.output_measure, PureDP())
        self.assertIsInstance(adaptive_composition(np.int64(10)), DecoratedQueryable)

    def test_correctness(self):
        """Queryable works as expected."""
        query = MeasurementQuery(
            create_mock_measurement(
                privacy_function_implemented=True,
                privacy_function_return_value=1,
                return_value=np.int64(22),
            )
        )
        answer = self.queryable(query)
        self.assertEqual(answer, np.int64(22))

    def test_interactive_measurement(self):
        """Can not answer interactive measurement queries."""
        query = MeasurementQuery(create_mock_measurement(is_interactive=True))
        with self.assertRaisesRegex(
            ValueError, "Cannot answer interactive measurement query"
        ):
            self.queryable(query)

    def test_insufficient_budget(self):
        """Raises error on insufficient budget."""
        query1 = MeasurementQuery(
            create_mock_measurement(
                privacy_function_implemented=True, privacy_function_return_value=2
            )
        )
        self.queryable(query1)
        query2 = MeasurementQuery(
            create_mock_measurement(
                privacy_function_implemented=True, privacy_function_return_value=2
            )
        )
        with self.assertRaisesRegex(
            ValueError, "Cannot answer query without exceeding available privacy budget"
        ):
            self.queryable(query2)
