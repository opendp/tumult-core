"""Unit tests for :mod:`~tmlt.core.transformations.identity`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import AbsoluteDifference, SymmetricDifference
from tmlt.core.transformations.identity import Identity
from tmlt.core.utils.testing import TestComponent


class TestIdentityTransformation(TestComponent):
    """Tests for :class:`~tmlt.core.transformations.identity.Identity`."""

    def test_properties(self):
        """Identity's properties have the expected values."""
        domain = NumpyIntegerDomain()
        metric = AbsoluteDifference()
        transformation = Identity(metric=metric, domain=domain)
        self.assertEqual(transformation.input_domain, domain)
        self.assertEqual(transformation.input_metric, metric)
        self.assertEqual(transformation.output_domain, domain)
        self.assertEqual(transformation.output_metric, metric)

    def test_identity(self):
        """Tests that identity transformation works correctly."""
        id_transformation = Identity(
            SymmetricDifference(), SparkDataFrameDomain(self.schema_a)
        )
        self.assertTrue(
            id_transformation.input_metric
            == id_transformation.output_metric
            == SymmetricDifference()
        )
        self.assertTrue(
            id_transformation.input_domain
            == id_transformation.output_domain
            == SparkDataFrameDomain(self.schema_a)
        )

        result = id_transformation(self.df_a)
        self.assert_frame_equal_with_sort(result.toPandas(), self.df_a.toPandas())
        self.assertTrue(id_transformation.stability_relation(1, 1))
        self.assertTrue(id_transformation.stability_relation(1, 2))
        self.assertFalse(id_transformation.stability_relation(4, 2))
