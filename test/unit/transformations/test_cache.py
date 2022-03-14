"""Unit tests for :mod:`~tmlt.core.transformations.cache`."""

# <placeholder: boilerplate>

# pylint: disable=no-self-use

import pandas as pd
from parameterized import parameterized

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.cache import Cache, SingleItemCache, cache
from tmlt.core.transformations.spark_transformations.filter import Filter
from tmlt.core.utils.testing import (
    PySparkTest,
    TestComponent,
    assert_property_immutability,
    create_mock_transformation,
    get_all_props,
)


class TestCacheTransformation(TestComponent):
    """Test for :class:`~tmlt.core.transformations.cache.Cache`."""

    @parameterized.expand(get_all_props(Cache))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = create_mock_transformation()
        assert_property_immutability(Cache(transformation), prop_name)

    def test_properties(self):
        """Cache's properties have the expected values."""
        transformation = create_mock_transformation()
        cached_transformation = Cache(transformation)
        self.assertEqual(
            cached_transformation.input_domain, transformation.input_domain
        )
        self.assertEqual(
            cached_transformation.input_metric, transformation.input_metric
        )
        self.assertEqual(
            cached_transformation.output_domain, transformation.output_domain
        )
        self.assertEqual(
            cached_transformation.output_metric, transformation.output_metric
        )
        self.assertEqual(cached_transformation.transformation, transformation)

    def test_transformation_output_cached(self):
        """Tests that result for transformation is cached correctly."""
        df = self.spark.createDataFrame(
            pd.DataFrame([[1.2, "X"], [0.9, "Y"]], columns=["A", "B"])
        )
        cached_filter = Cache(
            Filter(
                filter_expr="A>1",
                domain=SparkDataFrameDomain(self.schema_a),
                metric=SymmetricDifference(),
            )
        )
        cached_filter(df)
        self.assertIn(
            id(cached_filter), cache()._cache.keys()  # pylint: disable=protected-access
        )


class TestSingleItemCache(PySparkTest):
    """Tests for :class:`~tmlt.core.transformations.cache.SingleItemCache`."""

    def setUp(self):
        """Setup for tests."""
        self.single_item_cache = SingleItemCache()
        self.single_item_cache._cache = {  # pylint: disable=protected-access
            1: ("Key1", "Value1")
        }

    def test_set(self):
        """Tests for SingleItemCache.set()."""
        self.single_item_cache.set(2, "Key2", "Value2")
        self.assertDictEqual(
            {1: ("Key1", "Value1"), 2: ("Key2", "Value2")},
            self.single_item_cache._cache,  # pylint: disable=protected-access
        )

    def test_get(self):
        """Tests for SingleItemCache.get()."""
        self.assertEqual(
            self.single_item_cache.get(1, "Key1"),  # pylint: disable=protected-access
            "Value1",
        )
        with self.assertRaises(KeyError):
            self.single_item_cache.get(2, "Key2")

    def test_delete(self):
        """Tests for SingleItemCache.delete()."""
        self.single_item_cache.delete(1)
        self.assertNotIn(
            1, self.single_item_cache._cache.keys()  # pylint: disable=protected-access
        )
        with self.assertRaises(KeyError):
            self.single_item_cache.delete(2)
