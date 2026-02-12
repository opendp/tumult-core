"""Unit tests for :mod:`~tmlt.core.transformations.converters`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026


import re
from typing import Union

import sympy as sp
from parameterized import parameterized

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.exceptions import DomainColumnError
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.converters import UnwrapIfGroupedBy
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestUnwrapIfGroupedBy(TestComponent):
    """Tests for :class:`~tmlt.core.transformations.converters.UnwrapIfGroupedBy`."""

    @parameterized.expand(get_all_props(UnwrapIfGroupedBy))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        unwrapper = UnwrapIfGroupedBy(
            domain=SparkDataFrameDomain(self.schema_a),
            input_metric=IfGroupedBy(["B"], SumOf(SymmetricDifference())),
        )
        assert_property_immutability(unwrapper, prop_name)

    @parameterized.expand(
        [
            (SumOf(HammingDistance()),),
            (SumOf(SymmetricDifference()),),
            (RootSumOfSquared(HammingDistance()),),
            (RootSumOfSquared(SymmetricDifference()),),
        ]
    )
    def test_properties(self, inner_metric: Union[SumOf, RootSumOfSquared]):
        """Tests that UnwrapIfGroupedBy's properties have expected values."""
        input_metric = IfGroupedBy(["B"], inner_metric)
        domain = SparkDataFrameDomain(self.schema_a)
        unwrapper = UnwrapIfGroupedBy(domain=domain, input_metric=input_metric)
        self.assertEqual(unwrapper.input_metric, input_metric)
        self.assertEqual(unwrapper.output_metric, inner_metric.inner_metric)
        self.assertTrue(unwrapper.input_domain == domain == unwrapper.output_domain)

    @parameterized.expand(
        [
            (IfGroupedBy(["B"], SumOf(SymmetricDifference())), 4, 4, True),
            (IfGroupedBy(["B"], SumOf(SymmetricDifference())), 4, 4 - 1, False),
            (
                IfGroupedBy(["B"], RootSumOfSquared(SymmetricDifference())),
                sp.sqrt(2),
                2,
                True,
            ),
            (IfGroupedBy(["B"], RootSumOfSquared(SymmetricDifference())), 4, 16, True),
            (
                IfGroupedBy(["B"], RootSumOfSquared(SymmetricDifference())),
                4,
                16 - 1,
                False,
            ),
            (IfGroupedBy(["A", "B"], SumOf(SymmetricDifference())), 4, 4, True),
            (IfGroupedBy(["A", "B"], SumOf(SymmetricDifference())), 4, 4 - 1, False),
            (
                IfGroupedBy(["A", "B"], RootSumOfSquared(SymmetricDifference())),
                4,
                16,
                True,
            ),
            (
                IfGroupedBy(["A", "B"], RootSumOfSquared(SymmetricDifference())),
                4,
                16 - 1,
                False,
            ),
        ]
    )
    def test_stability_relation(
        self,
        metric: IfGroupedBy,
        d_in: ExactNumberInput,
        d_out: ExactNumberInput,
        expected: bool,
    ):
        """Tests that UnwrapIfGroupedBy's stability relation is correct."""
        unwrapper = UnwrapIfGroupedBy(
            domain=SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=metric,
        )
        self.assertEqual(unwrapper.stability_relation(d_in, d_out), expected)

    def test_noop(self):
        """Tests that UnwrapIfGroupedBy returns DataFrame unchanged."""
        unwrapper = UnwrapIfGroupedBy(
            domain=SparkDataFrameDomain(self.schema_a),
            input_metric=IfGroupedBy(["B"], SumOf(SymmetricDifference())),
        )
        self.assert_frame_equal_with_sort(
            unwrapper(self.df_a).toPandas(), self.df_a.toPandas()
        )

    @parameterized.expand(
        [
            (
                IfGroupedBy(["B"], SymmetricDifference()),
                ValueError,
                "Inner metric for IfGroupedBy metric must be "
                "SumOf(SymmetricDifference()), or "
                "RootSumOfSquared(SymmetricDifference())",
            ),
            (
                IfGroupedBy(["Z"], SumOf(SymmetricDifference())),
                DomainColumnError,
                "Invalid IfGroupedBy metric: ['Z'] not in domain",
            ),
            (
                IfGroupedBy(["B", "Z"], SumOf(SymmetricDifference())),
                DomainColumnError,
                "Invalid IfGroupedBy metric: ['Z'] not in domain",
            ),
        ]
    )
    def test_invalid_metric(
        self, input_metric, expected_error_type, expected_error_message
    ):
        """Tests the UnwrapIfGroupedby raises an error invalid input metrics."""
        with self.assertRaisesRegex(
            expected_error_type,
            re.escape(expected_error_message),
        ):
            UnwrapIfGroupedBy(
                domain=SparkDataFrameDomain(self.schema_a),
                input_metric=input_metric,
            )
