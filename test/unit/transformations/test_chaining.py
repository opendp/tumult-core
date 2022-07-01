"""Unit tests for :mod:`~tmlt.core.transformations.chaining`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use

import itertools
from unittest.mock import MagicMock, Mock

import pandas as pd
from parameterized import parameterized

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.spark_domains import SparkRowDomain
from tmlt.core.metrics import AbsoluteDifference, Metric, SymmetricDifference
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap,
    RowToRowsTransformation,
)
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    create_mock_transformation,
    get_all_props,
)


class TestChainTT(TestComponent):
    """Test for :class:`~tmlt.core.transformations.chaining.ChainTT`."""

    @parameterized.expand(get_all_props(ChainTT))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        t1 = create_mock_transformation(
            output_domain=NumpyIntegerDomain(), output_metric=AbsoluteDifference()
        )
        t2 = create_mock_transformation(
            input_domain=NumpyIntegerDomain(), input_metric=AbsoluteDifference()
        )
        assert_property_immutability(ChainTT(t1, t2, Mock()), prop_name)

    @parameterized.expand(
        [
            (
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                NumpyFloatDomain(),
                AbsoluteDifference(),
                NumpyFloatDomain(),
                AbsoluteDifference(),
            ),
            (
                NumpyFloatDomain(),
                AbsoluteDifference(),
                NumpyIntegerDomain(),
                AbsoluteDifference(),
                NumpyFloatDomain(),
                AbsoluteDifference(),
            ),
        ]
    )
    def test_properties(
        self,
        input_domain: Domain,
        input_metric: Metric,
        mid_domain: Domain,
        mid_metric: Metric,
        output_domain: Domain,
        output_metric: Metric,
    ):
        """ChainTT's properties have the expected values."""
        transformation1 = create_mock_transformation(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=mid_domain,
            output_metric=mid_metric,
        )
        transformation2 = create_mock_transformation(
            input_domain=mid_domain,
            input_metric=mid_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )
        chained_transformation = ChainTT(
            transformation1, transformation2, hint=lambda _, __: 1
        )
        self.assertEqual(chained_transformation.transformation1, transformation1)
        self.assertEqual(chained_transformation.transformation2, transformation2)
        self.assertEqual(chained_transformation.input_domain, input_domain)
        self.assertEqual(chained_transformation.input_metric, input_metric)
        self.assertEqual(chained_transformation.output_domain, output_domain)
        self.assertEqual(chained_transformation.output_metric, output_metric)

    def test_chained_transformation(self):
        """Tests that chained transformation is correct."""
        duplicate_rows = FlatMap(
            row_transformer=self.duplicate_transformer,
            max_num_rows=2,
            metric=SymmetricDifference(),
        )
        double_a = FlatMap(
            row_transformer=RowToRowsTransformation(
                input_domain=SparkRowDomain(self.schema_a),
                output_domain=ListDomain(SparkRowDomain(self.schema_a_augmented)),
                trusted_f=lambda x: [{"C": x["A"] * 2}],
                augment=True,
            ),
            max_num_rows=1,
            metric=SymmetricDifference(),
        )

        expected_df = pd.DataFrame(
            [[1.2, "X", 2.4], [1.2, "X", 2.4]], columns=["A", "B", "C"]
        )

        actual_df = ChainTT(
            duplicate_rows, double_a, hint=lambda metric_iv, _: metric_iv * 2
        )(self.df_a).toPandas()
        self.assert_frame_equal_with_sort(expected_df, actual_df)

    @parameterized.expand(
        [
            (*params1, *params2, use_hint)
            for params1, params2 in itertools.combinations(
                [
                    (True, 1, True),
                    (True, 2, False),
                    (False, 1, True),
                    (False, 2, False),
                ],
                2,
            )
            for use_hint in [True, False]
        ]
    )
    def test_stability_function_and_relation(
        self,
        stability_function_implemented1: bool,
        stability_function_return_value1: ExactNumberInput,
        stability_relation_return_value1: bool,
        stability_function_implemented2: bool,
        stability_function_return_value2: ExactNumberInput,
        stability_relation_return_value2: bool,
        use_hint: bool,
    ):
        """Tests that the stability function and relation work correctly."""
        mock_transformation1 = create_mock_transformation(
            stability_function_implemented=stability_function_implemented1,
            stability_function_return_value=stability_function_return_value1,
            stability_relation_return_value=stability_relation_return_value1,
        )
        mock_transformation2 = create_mock_transformation(
            stability_function_implemented=stability_function_implemented2,
            stability_function_return_value=stability_function_return_value2,
            stability_relation_return_value=stability_relation_return_value2,
        )
        mock_hint = MagicMock(return_value=(1, 1))
        transformation = ChainTT(
            transformation1=mock_transformation1,
            transformation2=mock_transformation2,
            hint=mock_hint if use_hint else None,
        )
        if not (stability_function_implemented1 and stability_function_implemented2):
            with self.assertRaisesRegex(NotImplementedError, "TEST"):
                transformation.stability_function(1)
        else:
            self.assertEqual(
                transformation.stability_function(1), stability_function_return_value2
            )
        if not stability_function_implemented1 and not use_hint:
            self.assertRaisesRegex(
                ValueError,
                "A hint is needed to check this privacy relation, because the "
                "stability_relation of self.transformation1 raised  a "
                "NotImplementedError: TEST",
            )
        else:
            self.assertEqual(
                transformation.stability_relation(1, 1),
                mock_transformation1.stability_relation(1, 1)
                and mock_transformation2.stability_relation(1, 1),
            )
            if mock_hint.called:
                mock_hint.assert_called_with(1, 1)

    def test_incompatible_domains(self):
        """Tests that chaining fails with incompatible domains."""
        incompatible_duplicate = FlatMap(
            row_transformer=RowToRowsTransformation(
                input_domain=SparkRowDomain(self.schema_b),  # Mismatching domain.
                output_domain=ListDomain(SparkRowDomain(self.schema_a)),
                trusted_f=lambda x: [x, x],
                augment=False,
            ),
            max_num_rows=2,
            metric=SymmetricDifference(),
        )

        with self.assertRaisesRegex(
            ValueError, "Can not chain transformations: Mismatching domains"
        ):
            ChainTT(self.duplicate_transformer, incompatible_duplicate)
