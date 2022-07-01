"""System tests for :mod:`~tmlt.core.measurements.interactive_measurements`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.interactive_measurements import (
    PrivacyAccountant,
    PrivacyAccountantState,
    SequentialComposition,
)
from tmlt.core.measures import PureDP
from tmlt.core.metrics import SumOf, SymmetricDifference
from tmlt.core.transformations.dictionary import CreateDictFromValue
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.utils.testing import PySparkTest


class TestPrivacyAccountant(PySparkTest):
    """Tests for :class:`~.PrivacyAccountant`."""

    def setUp(self) -> None:
        """Test setup."""
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
        self.accountant = PrivacyAccountant.launch(
            measurement=self.measurement, data=self.data
        )

    def test_queue_transformation_on_inactive_accountant(self) -> None:
        """Test queueing a transformation on an inactive accountant."""
        # Perform a split, to make self.accountant inactive
        split_transformation = PartitionByKeys(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            use_l2=False,
            keys=["A"],
            list_values=[(1,), (2,), (3,)],
        )
        split_budget = 3
        child_accountants = self.accountant.split(
            splitting_transformation=split_transformation, privacy_budget=split_budget
        )

        # Queue the transformation
        transformation = CreateDictFromValue(
            input_domain=self.accountant.input_domain,
            input_metric=self.accountant.input_metric,
            key="data",
        )
        transformed_domain = transformation.output_domain
        transformed_metric = transformation.output_metric
        transformed_d_in = transformation.stability_function(self.accountant.d_in)
        self.accountant.queue_transformation(transformation=transformation)
        # values should reflect the pending transformation
        self.assertEqual(self.accountant.input_domain, transformed_domain)
        self.assertEqual(self.accountant.input_metric, transformed_metric)
        self.assertEqual(self.accountant.d_in, transformed_d_in)
        self.assertIsNotNone(
            self.accountant._pending_transformation  # pylint: disable=protected-access
        )

        for c in child_accountants:
            c.retire()

        # Once the accountant is active again, the transformation should have
        # been run
        self.assertEqual(self.accountant.state, PrivacyAccountantState.ACTIVE)
        # pylint: disable=protected-access
        self.assertEqual(self.accountant._input_domain, transformed_domain)
        self.assertEqual(self.accountant._input_metric, transformed_metric)
        self.assertEqual(self.accountant._d_in, transformed_d_in)
        self.assertIsNone(self.accountant._pending_transformation)
        # pylint: enable=protected-access
