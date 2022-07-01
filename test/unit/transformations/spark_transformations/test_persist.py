"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.persist`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use

from parameterized import parameterized

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.spark_transformations.persist import (
    Persist,
    SparkAction,
    Unpersist,
)
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestPersist(PySparkTest):
    """Tests for Persist transformation."""

    def setUp(self):
        """Test setup."""
        self.transformation = Persist(
            domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
            metric=SymmetricDifference(),
        )

    @parameterized.expand(get_all_props(Persist))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.transformation, prop_name)

    def test_correctness(self):
        """Persist marks DataFrame to be persisted."""
        df = self.spark.createDataFrame([(1,)], schema=["A"])
        assert not df.is_cached
        df = self.transformation(df)
        self.assertTrue(df.is_cached)


class TestUnpersist(PySparkTest):
    """Tests for Unpersist transformation."""

    def setUp(self):
        """Test setup."""
        self.transformation = Unpersist(
            domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
            metric=SymmetricDifference(),
        )

    @parameterized.expand(get_all_props(Unpersist))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.transformation, prop_name)

    def test_correctness(self):
        """Unpersist marks a persisted DataFrame to be garbage collected."""
        df = self.spark.createDataFrame([(1,)], schema=["A"]).persist()
        assert df.is_cached
        df = self.transformation(df)
        self.assertFalse(df.is_cached)


class TestSparkAction(PySparkTest):
    """Tests for SparkAction transformation."""

    def setUp(self):
        """Test setup."""
        self.transformation = SparkAction(
            domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
            metric=SymmetricDifference(),
        )

    @parameterized.expand(get_all_props(SparkAction))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.transformation, prop_name)

    def test_correctness(self):
        """SparkAction makes Spark evaluate and persist a DataFrame immediately."""
        # pylint: disable=protected-access
        df = self.spark.createDataFrame([(1,)], schema=["A"]).persist()
        assert df.is_cached
        assert list(self.spark.sparkContext._jsc.sc().getRDDStorageInfo()) == []
        df = self.transformation(df)
        self.assertEqual(
            len(list(self.spark.sparkContext._jsc.sc().getRDDStorageInfo())), 1
        )
        df.unpersist()
