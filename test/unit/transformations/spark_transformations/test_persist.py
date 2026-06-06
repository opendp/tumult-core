"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.persist`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from parameterized import parameterized

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
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

    def test_format(self):
        """Persist formats as expected."""
        transformation = Persist(
            domain=SparkDataFrameDomain({"A": SparkStringColumnDescriptor()}),
            metric=SymmetricDifference(),
        )
        assert transformation.format() == "Persist"


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

    def test_format(self):
        """Unpersist formats as expected."""
        transformation = Unpersist(
            domain=SparkDataFrameDomain({"A": SparkStringColumnDescriptor()}),
            metric=SymmetricDifference(),
        )
        assert transformation.format() == "Unpersist"


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
        df = self.spark.createDataFrame([(1,)], schema=["A"]).persist()
        assert df.is_cached
        # this will assert that the list is empty
        assert not list(
            self.spark.sparkContext._jsc.sc().getRDDStorageInfo()  # noqa: SLF001
        )
        df = self.transformation(df)
        self.assertEqual(
            len(
                list(
                    self.spark.sparkContext._jsc.sc().getRDDStorageInfo()  # noqa: SLF001
                )
            ),
            1,
        )
        df.unpersist()

    def test_format(self):
        """SparkAction formats as expected."""
        transformation = SparkAction(
            domain=SparkDataFrameDomain({"A": SparkStringColumnDescriptor()}),
            metric=SymmetricDifference(),
        )
        assert transformation.format() == "SparkAction"
