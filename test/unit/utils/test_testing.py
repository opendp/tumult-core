"""Test for :mod:`tmlt.core.utils.testing`"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from operator import add

from pyspark.sql import SparkSession

from tmlt.core.utils.testing import PySparkTest


class TestSparkTestHarness(PySparkTest):
    """Test pyspark testing base class."""

    def test_basic(self):
        """Word count test."""
        test_rdd = self.spark.sparkContext.parallelize(
            ["hello spark", "hello again spark spark"], 2
        )
        results = (
            test_rdd.flatMap(lambda line: line.split())
            .map(lambda word: (word, 1))
            .reduceByKey(add)
            .collect()
        )
        expected_results = [("hello", 2), ("spark", 3), ("again", 1)]
        self.assertEqual(set(results), set(expected_results))

    def test_get_session(self):
        """Tests that *getOrCreate()* connects to test harness SparkSession."""
        spark = SparkSession.builder.getOrCreate()
        self.assertEqual(spark.conf.get("spark.app.name"), "TestSparkTestHarness")
