"""Unit tests for :mod:`~tmlt.core.measurements.aggregations`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026
import functools
import random
import unittest
from typing import Any, Callable, Generator, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pytest
import sympy as sp
from parameterized import parameterized, parameterized_class
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.exceptions import (
    DomainMismatchError,
    MetricMismatchError,
    UnsupportedCombinationError,
    UnsupportedMeasureError,
    UnsupportedMetricError,
)
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_average_measurement,
    create_bounds_measurement,
    create_count_distinct_measurement,
    create_count_measurement,
    create_partition_selection_measurement,
    create_quantile_measurement,
    create_standard_deviation_measurement,
    create_sum_measurement,
    create_variance_measurement,
)
from tmlt.core.measurements.converters import PureDPToApproxDP, PureDPToRhoZCDP
from tmlt.core.measurements.postprocess import PostProcess
from tmlt.core.measures import (
    ApproxDP,
    ApproxDPBudget,
    PrivacyBudget,
    PrivacyBudgetInput,
    PureDP,
    RhoZCDP,
)
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.utils.distributions import double_sided_geometric_cmf_exact
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import (
    Case,
    PySparkTest,
    assert_dataframe_equal,
    parametrize,
)

datasets = [
    # Tests with data.
    [("x1", 2, 1), ("x1", 2, 2), ("x2", 4, 3)],
    # Tests with null data.
    [],
]

params = [
    (
        [],
        [StructField("A", StringType())],
    ),
    (
        [("x1",), ("x2",), ("x3",), (None,)],
        [StructField("A", StringType())],
    ),
    (
        [("x1", 2), ("x2", 4), ("x3", 0), (None, None)],
        [StructField("A", StringType()), StructField("B", LongType(), nullable=True)],
    ),
]


# Disabling no-member because groupby_columns are defined in the setup function.


@parameterized_class(
    {
        "data": data,
        "group_keys_list": group_keys_list,
        "struct_fields": structfields,
    }
    for data in datasets
    for group_keys_list, structfields in params
)
class TestGroupByAggregationMeasurements(PySparkTest):
    """Tests for :mod:`tmlt.core.measurements.aggregations`."""

    group_keys_list: List[Tuple[str, ...]]
    struct_fields: List[StructField]
    data: List[Tuple[Any]]

    def setUp(self):
        """Test setup."""
        domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(allow_null=True),
                "C": SparkIntegerColumnDescriptor(),
            }
        )
        self.input_domain = domain
        self.group_keys = self.spark.createDataFrame(
            self.group_keys_list, schema=StructType(self.struct_fields.copy())
        )
        self.sdf = self.spark.createDataFrame(self.data, schema=domain.spark_schema)
        self.groupby_columns = [field.name for field in self.struct_fields]

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                d_out,
                noise_mechanism,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    ["A"], cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(2)),
                (RhoZCDP(), sp.Integer(2)),
                (ApproxDP(), (sp.Integer(2), sp.Integer(0))),
            ]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    # Not marked slow unlike the others to keep one fast groupby test.
    def test_create_count_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_count_measurement works correctly with groupby."""
        count_measurement = create_count_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            count_column="test_count",
        )

        self.assertEqual(count_measurement.input_domain, self.input_domain)
        self.assertEqual(count_measurement.output_measure, output_measure)
        self.assertEqual(count_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = count_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, [*self.groupby_columns, "test_count"])

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                d_out,
                noise_mechanism,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    ["A"], cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(2)),
                (RhoZCDP(), sp.Integer(2)),
                (ApproxDP(), (sp.Integer(2), sp.Integer(0))),
            ]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    @pytest.mark.slow
    def test_create_count_distinct_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_count_distinct_measurement works correctly with groupby."""
        count_distinct_measurement = create_count_distinct_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            count_column="test_count",
        )
        self.assertEqual(count_distinct_measurement.input_domain, self.input_domain)
        self.assertEqual(count_distinct_measurement.output_measure, output_measure)
        self.assertEqual(
            count_distinct_measurement.privacy_function(sp.Integer(1)), d_out
        )
        answer = count_distinct_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, [*self.groupby_columns, "test_count"])

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                d_out,
                noise_mechanism,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    ["A"], cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    @pytest.mark.slow
    def test_create_sum_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_sum_measurement works correctly with groupby."""
        sum_measurement = create_sum_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="C",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            sum_column="sumC",
        )
        self.assertEqual(sum_measurement.input_domain, self.input_domain)
        self.assertEqual(sum_measurement.output_measure, output_measure)
        self.assertEqual(sum_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = sum_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, [*self.groupby_columns, "sumC"])

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                d_out,
                noise_mechanism,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    ["A"], cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    @pytest.mark.slow
    def test_create_average_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_average_measurement works correctly with groupby."""
        average_measurement = create_average_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="C",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            average_column="AVG(C)",
        )
        self.assertEqual(average_measurement.input_domain, self.input_domain)
        self.assertEqual(average_measurement.output_measure, output_measure)
        self.assertEqual(average_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = average_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, [*self.groupby_columns, "AVG(C)"])

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                noise_mechanism,
                d_out,
                output_column,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    ["A"], cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            for output_column in ["XYZ", None]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    @pytest.mark.slow
    def test_create_standard_deviation_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
        noise_mechanism: NoiseMechanism,
        d_out: PrivacyBudgetInput,
        output_column: Optional[str] = None,
    ):
        """Tests that create_standard_deviation_measurement works correctly."""
        standard_deviation_measurement = create_standard_deviation_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="C",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            keep_intermediates=False,
            standard_deviation_column=output_column,
        )
        self.assertEqual(standard_deviation_measurement.input_domain, self.input_domain)
        self.assertEqual(standard_deviation_measurement.output_measure, output_measure)
        self.assertEqual(
            standard_deviation_measurement.privacy_function(sp.Integer(1)), d_out
        )
        answer = standard_deviation_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        if not output_column:
            output_column = "stddev(C)"
        self.assertEqual(answer.columns, [*self.groupby_columns, output_column])
        answer.first()

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                noise_mechanism,
                d_out,
                output_column,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    ["A"], cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            for output_column in ["XYZ", None]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    @pytest.mark.slow
    def test_create_variance_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
        noise_mechanism: NoiseMechanism,
        d_out: PrivacyBudgetInput,
        output_column: Optional[str] = None,
    ):
        """Tests that create_variance_measurement works correctly with groupby."""
        variance_measurement = create_variance_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="C",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            keep_intermediates=False,
            variance_column=output_column,
        )
        self.assertEqual(variance_measurement.input_domain, self.input_domain)
        self.assertEqual(variance_measurement.output_measure, output_measure)
        self.assertEqual(variance_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = variance_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        if not output_column:
            output_column = "var(C)"
        self.assertEqual(answer.columns, [*self.groupby_columns, output_column])
        answer.first()

    @parameterized.expand(
        [
            (input_metric, groupby_output_metric, d_out, output_measure)
            for output_measure, d_out, groupby_output_metric in [
                (PureDP(), sp.Integer(4), SumOf(SymmetricDifference())),
                (RhoZCDP(), sp.Integer(4), RootSumOfSquared(SymmetricDifference())),
                (
                    ApproxDP(),
                    (sp.Integer(4), sp.Integer(0)),
                    SumOf(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    ["A"], cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
        ]
    )
    @pytest.mark.slow
    def test_create_quantile_measurement_with_groupby(
        self,
        input_metric: Union[HammingDistance, SymmetricDifference],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        d_out: PrivacyBudgetInput,
        output_measure: Union[PureDP, RhoZCDP],
    ):
        """Tests that create_quantile_measurement works correctly with groupby."""
        quantile_measurement = create_quantile_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="C",
            quantile=0.5,
            upper=10,
            lower=0,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            quantile_column="MEDIAN(C)",
        )
        self.assertEqual(quantile_measurement.input_domain, self.input_domain)
        self.assertEqual(quantile_measurement.input_metric, input_metric)
        self.assertEqual(quantile_measurement.output_measure, output_measure)
        self.assertEqual(quantile_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = quantile_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, [*self.groupby_columns, "MEDIAN(C)"])
        df = answer.toPandas()
        self.assertTrue(((df["MEDIAN(C)"] <= 10) & (df["MEDIAN(C)"] >= 0)).all())

    @parameterized.expand(
        [
            (input_metric, groupby_output_metric, d_out, output_measure)
            for output_measure, d_out, groupby_output_metric in [
                (PureDP(), sp.Integer(4), SumOf(SymmetricDifference())),
                (RhoZCDP(), sp.Integer(4), RootSumOfSquared(SymmetricDifference())),
                (
                    ApproxDP(),
                    (sp.Integer(4), sp.Integer(0)),
                    SumOf(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                IfGroupedBy(
                    ["A"], cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
        ]
    )
    def test_create_bounds_measurement_with_groupby(
        self,
        input_metric: Union[IfGroupedBy, SymmetricDifference],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        d_out: PrivacyBudgetInput,
        output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    ):
        """Tests that create_bounds_measurement works correctly with groupby."""
        bounds_measurement = create_bounds_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="C",
            threshold=0.9,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            upper_bound_column="upper",
            lower_bound_column="lower",
        )
        self.assertEqual(bounds_measurement.input_domain, self.input_domain)
        self.assertEqual(bounds_measurement.input_metric, input_metric)
        self.assertEqual(bounds_measurement.output_measure, output_measure)
        self.assertEqual(bounds_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = bounds_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(set(answer.columns), {*self.groupby_columns, "lower", "upper"})
        for row in answer.collect():
            assert row.upper > 0
            assert row.lower < 0


@parametrize(
    Case("groupby metric mismatch")(
        input_domain=SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(allow_null=True),
                "C": SparkIntegerColumnDescriptor(),
            }
        ),
        input_metric=IfGroupedBy(["A"], SumOf(SymmetricDifference())),
        groupby_input_metric=IfGroupedBy(["B"], SumOf(SymmetricDifference())),
        groupby_input_domain=SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(allow_null=True),
                "C": SparkIntegerColumnDescriptor(),
            }
        ),
        group_keys=pd.DataFrame({"B": [1]}),
        groupby_use_l2=False,
        error_type=MetricMismatchError,
        error_message="Input metric must match with groupby transformation",
    ),
    Case("groupby domain mismatch")(
        input_domain=SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(allow_null=True),
                "C": SparkIntegerColumnDescriptor(),
            }
        ),
        input_metric=IfGroupedBy(["A"], SumOf(SymmetricDifference())),
        groupby_input_metric=IfGroupedBy(["A"], SumOf(SymmetricDifference())),
        groupby_input_domain=SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(allow_null=True),
                "X": SparkIntegerColumnDescriptor(),
            }
        ),
        group_keys=pd.DataFrame({"A": ["x1"]}),
        groupby_use_l2=False,
        error_type=DomainMismatchError,
        error_message="Input domain must match with groupby transformation",
    ),
)
@parametrize(
    Case("count")(
        create_measurement_method=create_count_measurement,
        extra_args={
            "count_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("count distinct")(
        create_measurement_method=create_count_distinct_measurement,
        extra_args={
            "count_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("sum")(
        create_measurement_method=create_sum_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("average")(
        create_measurement_method=create_average_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("standard deviation")(
        create_measurement_method=create_standard_deviation_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("variance")(
        create_measurement_method=create_variance_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("quantile")(
        create_measurement_method=create_quantile_measurement,
        extra_args={
            "lower": 0,
            "upper": 10,
            "measure_column": "C",
            "quantile": 0.5,
        },
    ),
    Case("bounds")(
        create_measurement_method=create_bounds_measurement,
        extra_args={
            "measure_column": "C",
            "threshold": 0.9,
        },
    ),
)
def test_create_measurement_with_groupby_errors(
    spark,
    create_measurement_method,
    extra_args,
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    groupby_input_domain: SparkDataFrameDomain,
    groupby_input_metric: IfGroupedBy,
    group_keys: pd.DataFrame,
    groupby_use_l2: bool,
    error_type,
    error_message: str,
):
    """Test that the various create_*_measurement methods correctly catch errors."""
    with pytest.raises(error_type, match=error_message):
        create_measurement_method(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=PureDP(),
            d_in=sp.Integer(1),
            d_out=sp.Integer(1),
            groupby_transformation=GroupBy(
                input_domain=groupby_input_domain,
                input_metric=groupby_input_metric,
                use_l2=groupby_use_l2,
                group_keys=spark.createDataFrame(group_keys),
            ),
            **extra_args,
        )


@parametrize(
    Case("average_default_names")(
        measurement_method=create_average_measurement,
        extra_args={},
        expected_output={
            "avg(value)": 2,
            "sod(value)": -9,
            "count": 3,
            "midpoint(value)": 5,
        },
    ),
    Case("variance_default_names")(
        measurement_method=create_variance_measurement,
        extra_args={},
        expected_output={
            "var(value)": 1,
            "sod(value)": -9,
            "sos(value)": -136,
            "count": 3,
            "midpoint(value)": 5,
            "midpoint_of_squared(value)": 50,
        },
    ),
    Case("standard_deviation_default_names")(
        measurement_method=create_standard_deviation_measurement,
        extra_args={},
        expected_output={
            "stddev(value)": 1,
            "sod(value)": -9,
            "sos(value)": -136,
            "count": 3,
            "midpoint(value)": 5,
            "midpoint_of_squared(value)": 50,
        },
    ),
    Case("average_custom_names")(
        measurement_method=create_average_measurement,
        extra_args={
            "average_column": "foo",
        },
        expected_output={
            "foo": 2,
            "sod(value)": -9,
            "count": 3,
            "midpoint(value)": 5,
        },
    ),
    Case("variance_custom_names")(
        measurement_method=create_variance_measurement,
        extra_args={
            "variance_column": "foo",
        },
        expected_output={
            "foo": 1,
            "sod(value)": -9,
            "sos(value)": -136,
            "count": 3,
            "midpoint(value)": 5,
            "midpoint_of_squared(value)": 50,
        },
    ),
    Case("standard_deviation_custom_names")(
        measurement_method=create_standard_deviation_measurement,
        extra_args={
            "standard_deviation_column": "foo",
        },
        expected_output={
            "foo": 1,
            "sod(value)": -9,
            "sos(value)": -136,
            "count": 3,
            "midpoint(value)": 5,
            "midpoint_of_squared(value)": 50,
        },
    ),
)
def test_scalar_intermediates(spark, measurement_method, extra_args, expected_output):
    """Tests the intermediates returned by create_average_measurement."""
    data = spark.createDataFrame(
        [
            (1,),
            (2,),
            (3,),
        ],
        ["value"],
    )

    domain = SparkDataFrameDomain(
        {
            "value": SparkIntegerColumnDescriptor(),
        }
    )

    metric = SymmetricDifference()

    avg = measurement_method(
        domain,
        metric,
        PureDP(),
        float("inf"),
        NoiseMechanism.GEOMETRIC,
        groupby_transformation=None,
        measure_column="value",
        lower=0,
        upper=10,
        keep_intermediates=True,
        **extra_args,
    )

    output = avg(data)

    assert output == expected_output


@parametrize(
    Case("average_default_columns")(
        measurement_method=create_average_measurement,
        extra_args={},
        expected_output=pd.DataFrame(
            [
                {
                    "group": "a",
                    "avg(value)": 1,
                    "sod(value)": -12,
                    "count": 3,
                    "midpoint(value)": 5,
                },
                {
                    "group": "b",
                    "avg(value)": 5,
                    "sod(value)": 0,
                    "count": 3,
                    "midpoint(value)": 5,
                },
            ]
        ),
    ),
    Case("variance_default_columns")(
        measurement_method=create_variance_measurement,
        extra_args={},
        expected_output=pd.DataFrame(
            [
                {
                    "group": "a",
                    "var(value)": 1,
                    "sod(value)": -12,
                    "sos(value)": -145,
                    "count": 3,
                    "midpoint(value)": 5,
                    "midpoint_of_squared(value)": 50,
                },
                {
                    "group": "b",
                    "var(value)": 1,
                    "sod(value)": 0,
                    "sos(value)": -73,
                    "count": 3,
                    "midpoint(value)": 5,
                    "midpoint_of_squared(value)": 50,
                },
            ]
        ),
    ),
    Case("standard_deviation_default_columns")(
        measurement_method=create_standard_deviation_measurement,
        extra_args={},
        expected_output=pd.DataFrame(
            [
                {
                    "group": "a",
                    "stddev(value)": 1,
                    "sod(value)": -12,
                    "sos(value)": -145,
                    "count": 3,
                    "midpoint(value)": 5,
                    "midpoint_of_squared(value)": 50,
                },
                {
                    "group": "b",
                    "stddev(value)": 1,
                    "sod(value)": 0,
                    "sos(value)": -73,
                    "count": 3,
                    "midpoint(value)": 5,
                    "midpoint_of_squared(value)": 50,
                },
            ]
        ),
    ),
    Case("average_custom_columns")(
        measurement_method=create_average_measurement,
        extra_args={
            "average_column": "foo",
        },
        expected_output=pd.DataFrame(
            [
                {
                    "group": "a",
                    "foo": 1,
                    "sod(value)": -12,
                    "count": 3,
                    "midpoint(value)": 5,
                },
                {
                    "group": "b",
                    "foo": 5,
                    "sod(value)": 0,
                    "count": 3,
                    "midpoint(value)": 5,
                },
            ]
        ),
    ),
    Case("variance_custom_columns")(
        measurement_method=create_variance_measurement,
        extra_args={
            "variance_column": "foo",
        },
        expected_output=pd.DataFrame(
            [
                {
                    "group": "a",
                    "foo": 1,
                    "sod(value)": -12,
                    "sos(value)": -145,
                    "count": 3,
                    "midpoint(value)": 5,
                    "midpoint_of_squared(value)": 50,
                },
                {
                    "group": "b",
                    "foo": 1,
                    "sod(value)": 0,
                    "sos(value)": -73,
                    "count": 3,
                    "midpoint(value)": 5,
                    "midpoint_of_squared(value)": 50,
                },
            ]
        ),
    ),
    Case("standard_deviation_custom_columns")(
        measurement_method=create_standard_deviation_measurement,
        extra_args={
            "standard_deviation_column": "foo",
        },
        expected_output=pd.DataFrame(
            [
                {
                    "group": "a",
                    "foo": 1,
                    "sod(value)": -12,
                    "sos(value)": -145,
                    "count": 3,
                    "midpoint(value)": 5,
                    "midpoint_of_squared(value)": 50,
                },
                {
                    "group": "b",
                    "foo": 1,
                    "sod(value)": 0,
                    "sos(value)": -73,
                    "count": 3,
                    "midpoint(value)": 5,
                    "midpoint_of_squared(value)": 50,
                },
            ]
        ),
    ),
)
def test_grouped_intermediates(spark, measurement_method, extra_args, expected_output):
    """Tests the intermediates returned by create_average_measurement."""
    data = spark.createDataFrame(
        [("a", 0), ("a", 1), ("a", 2), ("b", 4), ("b", 5), ("b", 6)],
        ["group", "value"],
    )

    group_keys = spark.createDataFrame([("a",), ("b",)], ["group"])

    domain = SparkDataFrameDomain(
        {
            "group": SparkStringColumnDescriptor(),
            "value": SparkIntegerColumnDescriptor(),
        }
    )

    metric = SymmetricDifference()

    groupby = GroupBy(
        input_domain=domain, input_metric=metric, use_l2=False, group_keys=group_keys
    )

    avg = measurement_method(
        input_domain=domain,
        input_metric=metric,
        output_measure=PureDP(),
        d_out=float("inf"),
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        groupby_transformation=groupby,
        measure_column="value",
        lower=0,
        upper=10,
        keep_intermediates=True,
        **extra_args,
    )

    output = avg(data)

    assert_dataframe_equal(output, expected_output)


class TestAggregationMeasurement(PySparkTest):
    """Tests for :mod:`tmlt.core.measurements.aggregations`."""

    def setUp(self):
        """Test setup."""
        self.input_domain = SparkDataFrameDomain(
            {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        self.sdf = self.spark.createDataFrame([("x1", 2), ("x2", 4)], schema=["A", "B"])

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(2)),
                (RhoZCDP(), sp.Integer(2)),
                (ApproxDP(), (sp.Integer(2), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_count_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_count_measurement works correctly without groupby."""
        count_measurement = create_count_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            output_measure=output_measure,
        )
        self.assertEqual(count_measurement.input_domain, self.input_domain)
        self.assertEqual(count_measurement.input_metric, input_metric)
        self.assertEqual(count_measurement.output_measure, output_measure)
        self.assertEqual(count_measurement.privacy_function(1), d_out)
        answer = count_measurement(self.sdf)
        self.assertIsInstance(answer, (float, int))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(2)),
                (RhoZCDP(), sp.Integer(2)),
                (ApproxDP(), (sp.Integer(2), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_count_distinct_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests create_count_distinct_measurement without groupby."""
        count_distinct_measurement = create_count_distinct_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            output_measure=output_measure,
        )

        self.assertEqual(count_distinct_measurement.input_domain, self.input_domain)
        self.assertEqual(count_distinct_measurement.input_metric, input_metric)
        self.assertEqual(count_distinct_measurement.output_measure, output_measure)
        self.assertEqual(count_distinct_measurement.privacy_function(1), d_out)
        answer = count_distinct_measurement(self.sdf)
        self.assertIsInstance(answer, (int, float))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_sum_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_sum_measurement works correctly without groupby."""
        sum_measurement = create_sum_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            output_measure=output_measure,
        )

        self.assertEqual(sum_measurement.input_domain, self.input_domain)
        self.assertEqual(sum_measurement.input_metric, input_metric)
        self.assertEqual(sum_measurement.output_measure, output_measure)
        self.assertEqual(sum_measurement.privacy_function(1), d_out)
        answer = sum_measurement(self.sdf)
        self.assertIsInstance(answer, (float, int))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_average_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_average_measurement works correctly without groupby."""
        average_measurement = create_average_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            keep_intermediates=False,
            output_measure=output_measure,
        )

        self.assertEqual(average_measurement.input_domain, self.input_domain)
        self.assertEqual(average_measurement.input_metric, input_metric)
        self.assertEqual(average_measurement.output_measure, output_measure)
        self.assertEqual(average_measurement.privacy_function(1), d_out)
        answer = average_measurement(self.sdf)
        self.assertIsInstance(answer, (float, int))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_standard_deviation_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_standard_deviation_measurement works correctly."""
        standard_deviation_measurement = create_standard_deviation_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            keep_intermediates=False,
            output_measure=output_measure,
        )

        self.assertEqual(standard_deviation_measurement.input_domain, self.input_domain)
        self.assertEqual(standard_deviation_measurement.input_metric, input_metric)
        self.assertEqual(standard_deviation_measurement.output_measure, output_measure)
        self.assertEqual(standard_deviation_measurement.privacy_function(1), d_out)
        answer = standard_deviation_measurement(self.sdf)
        self.assertIsInstance(answer, float)

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_variance_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_variance_measurement works correctly without groupby."""
        variance_measurement = create_variance_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            keep_intermediates=False,
            output_measure=output_measure,
        )

        self.assertEqual(variance_measurement.input_domain, self.input_domain)
        self.assertEqual(variance_measurement.input_metric, input_metric)
        self.assertEqual(variance_measurement.output_measure, output_measure)
        self.assertEqual(variance_measurement.privacy_function(1), d_out)
        answer = variance_measurement(self.sdf)
        self.assertIsInstance(answer, (int, float))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
        ]
    )
    def test_create_quantile_measurement_without_groupby(
        self,
        input_metric: Union[HammingDistance, SymmetricDifference],
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
    ):
        """Tests that create_quantile_measurement works correctly without groupby."""
        quantile_measurement = create_quantile_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="B",
            quantile=0.5,
            upper=10,
            lower=0,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=None,
            quantile_column="MEDIAN(B)",
        )
        self.assertEqual(quantile_measurement.input_domain, self.input_domain)
        self.assertEqual(quantile_measurement.input_metric, input_metric)
        self.assertEqual(quantile_measurement.output_measure, output_measure)
        self.assertEqual(quantile_measurement.privacy_function(1), d_out)
        answer = quantile_measurement(self.sdf)
        self.assertIsInstance(answer, float)
        self.assertLessEqual(answer, 10)
        self.assertGreaterEqual(answer, 0)

    @parameterized.expand(
        [
            (float("inf"), 0, 1, 0, 0, None),
            # Test with alternate definition of infinite budget.
            (1, 1, 1, 0, 0, None),
            # Test large value of epsilon succeeds without error.
            (
                10000,
                1
                - double_sided_geometric_cmf_exact(
                    5 - 2, ExactNumber(1) / ExactNumber(10000)
                ),
                1,
                ExactNumber(1) / ExactNumber(10000),
                5,
                None,
            ),
            (
                ExactNumber(1) / ExactNumber(3),
                1 - double_sided_geometric_cmf_exact(7 - 2, 3),
                1,
                3,
                7,
                None,
            ),
            (
                ExactNumber(1) / ExactNumber(17),
                1 - double_sided_geometric_cmf_exact(10 - 2, 17),
                1,
                17,
                10,
                None,
            ),
            (
                ExactNumber(2) / ExactNumber(13),
                2
                * ExactNumber(sp.E) ** (ExactNumber(2) / ExactNumber(13))
                * (1 - double_sided_geometric_cmf_exact(50 - 2, 13)),
                2,
                13,
                50,
                "my_count_column",
            ),
        ]
    )
    def test_create_partition_selection_measurement(
        self,
        epsilon: ExactNumberInput,
        delta: ExactNumberInput,
        d_in: ExactNumberInput,
        expected_alpha: ExactNumberInput,
        expected_threshold: ExactNumberInput,
        count_column: Optional[str] = None,
    ) -> None:
        """Test create_partition_selection_measurement works correctly."""
        measurement = create_partition_selection_measurement(
            input_domain=self.input_domain,
            epsilon=epsilon,
            delta=delta,
            d_in=d_in,
            count_column=count_column,
        )
        self.assertEqual(measurement.alpha, expected_alpha)
        self.assertEqual(measurement.threshold, expected_threshold)
        self.assertEqual(measurement.input_domain, self.input_domain)
        if count_column is not None:
            self.assertEqual(measurement.count_column, count_column)
        # Check that measurement.privacy_function(d_in) = (epsilon, delta)
        measurement_epsilon, measurement_delta = measurement.privacy_function(d_in)
        if ApproxDPBudget((epsilon, delta)).is_finite():
            self.assertEqual(measurement_epsilon, epsilon)
            self.assertEqual(measurement_delta, delta)
        else:
            self.assertFalse(
                ApproxDPBudget((measurement_epsilon, measurement_delta)).is_finite()
            )

    @parameterized.expand(
        [
            (PureDP(), 1, "B", 0.7, 1),
            (RhoZCDP(), 1, "B", 0.8, 2),
            (ApproxDP(), (1, 0), "B", 0.9, 1),
        ]
    )
    def test_create_bound_measurement(
        self,
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
        measure_column: str,
        threshold: float,
        d_in: ExactNumberInput = 1,
    ):
        """Test create_bound_selection_measurement works correctly."""
        measurement = create_bounds_measurement(
            input_domain=self.input_domain,
            input_metric=SymmetricDifference(),
            output_measure=output_measure,
            d_out=d_out,
            measure_column=measure_column,
            threshold=threshold,
            d_in=d_in,
        )
        d_out = PrivacyBudget.cast(output_measure, d_out).value
        if isinstance(measurement, PureDPToRhoZCDP):
            # help mypy
            assert isinstance(measurement.pure_dp_measurement, PostProcess)
            measurement = measurement.pure_dp_measurement
            epsilon = sp.sqrt(ExactNumber(sp.Integer(2) * d_out).expr)
        elif isinstance(measurement, PureDPToApproxDP):
            # help mypy
            assert isinstance(measurement.pure_dp_measurement, PostProcess)
            measurement = measurement.pure_dp_measurement
            assert isinstance(d_out, tuple)
            epsilon = d_out[0]
        else:
            epsilon = d_out
        d_in = ExactNumber(d_in)
        self.assertEqual(measurement.input_domain, self.input_domain)
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.privacy_function(d_in), epsilon)

        answer = measurement(self.sdf)
        self.assertIsInstance(answer, tuple)
        self.assertIsInstance(answer[0], int)
        self.assertIsInstance(answer[1], int)
        self.assertEqual(answer[0], -answer[1])
        self.assertTrue(answer[0] < answer[1])


@parametrize(
    Case("IfGroupedBy without grouping")(
        input_domain=SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(allow_null=True),
                "C": SparkIntegerColumnDescriptor(),
            }
        ),
        input_metric=IfGroupedBy(["A"], SumOf(SymmetricDifference())),
        error_type=UnsupportedMetricError,
        error_message=(
            "Cannot use IfGroupedBy input metric if no groupby_transformation"
            " is provided"
        ),
    )
)
@parametrize(
    Case("count")(
        create_measurement_method=create_count_measurement,
        extra_args={
            "count_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("count distinct")(
        create_measurement_method=create_count_distinct_measurement,
        extra_args={
            "count_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("sum")(
        create_measurement_method=create_sum_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("average")(
        create_measurement_method=create_average_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("standard deviation")(
        create_measurement_method=create_standard_deviation_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("variance")(
        create_measurement_method=create_variance_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "C",
            "noise_mechanism": NoiseMechanism.LAPLACE,
        },
    ),
    Case("quantile")(
        create_measurement_method=create_quantile_measurement,
        extra_args={
            "lower": 0,
            "upper": 10,
            "measure_column": "C",
            "quantile": 0.5,
        },
    ),
    Case("bounds")(
        create_measurement_method=create_bounds_measurement,
        extra_args={
            "measure_column": "C",
            "threshold": 0.9,
        },
    ),
)
def test_create_scalar_measurement_errors(
    create_measurement_method,
    extra_args,
    input_domain: SparkDataFrameDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    error_type,
    error_message: str,
):
    """Test that the various create_*_measurement methods correctly catch errors."""
    with pytest.raises(error_type, match=error_message):
        create_measurement_method(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=PureDP(),
            d_in=sp.Integer(1),
            d_out=sp.Integer(1),
            groupby_transformation=None,
            **extra_args,
        )


@parametrize(
    Case("discrete gaussian")(noise_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN),
    Case("gaussian")(noise_mechanism=NoiseMechanism.GAUSSIAN),
)
@parametrize(
    Case("count")(
        create_measurement_method=create_count_measurement,
        extra_args={},
    ),
    Case("count distinct")(
        create_measurement_method=create_count_distinct_measurement,
        extra_args={},
    ),
    Case("sum")(
        create_measurement_method=create_sum_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "B",
        },
    ),
    Case("average")(
        create_measurement_method=create_average_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "B",
        },
    ),
    Case("standard deviation")(
        create_measurement_method=create_standard_deviation_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "B",
        },
    ),
    Case("variance")(
        create_measurement_method=create_variance_measurement,
        extra_args={
            "lower": sp.Integer(0),
            "upper": sp.Integer(10),
            "measure_column": "B",
        },
    ),
)
def test_unsupported_output_measure_for_noise_mechanism(
    create_measurement_method, extra_args, noise_mechanism: NoiseMechanism
):
    """Gaussian mechanisms reject PureDP as an output measure."""
    input_domain = SparkDataFrameDomain(
        {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
    )
    with pytest.raises(
        UnsupportedMeasureError,
        match="is not supported by noise mechanism",
    ):
        create_measurement_method(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            d_in=sp.Integer(1),
            d_out=sp.Integer(1),
            noise_mechanism=noise_mechanism,
            groupby_transformation=None,
            **extra_args,
        )


@parametrize(
    Case("sum")(create_measurement_method=create_sum_measurement),
    Case("average")(create_measurement_method=create_average_measurement),
    Case("standard deviation")(
        create_measurement_method=create_standard_deviation_measurement
    ),
    Case("variance")(create_measurement_method=create_variance_measurement),
)
def test_non_numeric_measure_column(create_measurement_method):
    """Measure column must be numeric for sum-based aggregations."""
    input_domain = SparkDataFrameDomain(
        {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
    )
    with pytest.raises(ValueError, match="Measure column must be numeric"):
        create_measurement_method(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            d_in=sp.Integer(1),
            d_out=sp.Integer(1),
            noise_mechanism=NoiseMechanism.LAPLACE,
            measure_column="A",
            lower=sp.Integer(0),
            upper=sp.Integer(10),
            groupby_transformation=None,
        )


def test_partition_selection_d_in_less_than_one():
    """Partition selection does not support d_in < 1."""
    with pytest.raises(
        NotImplementedError,
        match="Creating a partition selection measurement with d_in < 1",
    ):
        create_partition_selection_measurement(
            input_domain=SparkDataFrameDomain(
                {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
            ),
            epsilon=sp.Integer(1),
            delta=sp.Rational(1, 10),
            d_in=sp.Rational(1, 2),
        )


def test_bounds_measurement_d_in_less_than_one(spark):
    """Bounds measurement does not support d_in < 1."""
    with pytest.raises(
        NotImplementedError,
        match="Creating a bounds measurement with d_in < 1",
    ):
        create_bounds_measurement(
            input_domain=SparkDataFrameDomain(
                {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
            ),
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            d_out=sp.Integer(1),
            measure_column="B",
            threshold=0.5,
            d_in=sp.Rational(1, 2),
        )


INPUT_DOMAIN = SparkDataFrameDomain(
    {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
)


class TestBadDelta(unittest.TestCase):
    """Tests for :mod:`tmlt.core.measurements.aggregations`."""

    @parameterized.expand(
        [
            (noise_mechanism, d_out, f)
            for noise_mechanism, d_out in [
                (NoiseMechanism.LAPLACE, (sp.Integer(1), sp.Rational(1, 2))),
                (NoiseMechanism.GEOMETRIC, (sp.Integer(1), sp.Rational(1, 2))),
                (NoiseMechanism.GAUSSIAN, (sp.Integer(1), sp.Rational(1, 2))),
                (NoiseMechanism.DISCRETE_GAUSSIAN, (sp.Integer(1), sp.Rational(1, 2))),
                (NoiseMechanism.GAUSSIAN, (sp.Integer(1), sp.Integer(0))),
                (NoiseMechanism.DISCRETE_GAUSSIAN, (sp.Integer(1), sp.Integer(0))),
            ]
            for f in [
                functools.partial(
                    create_count_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_count_distinct_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_sum_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    upper=sp.Integer(10),
                    lower=sp.Integer(0),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_average_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    upper=sp.Integer(10),
                    lower=sp.Integer(0),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_standard_deviation_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    upper=sp.Integer(10),
                    lower=sp.Integer(0),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_variance_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    upper=sp.Integer(10),
                    lower=sp.Integer(0),
                    output_measure=ApproxDP(),
                ),
            ]
        ]
    )
    def test_functions_with_noise_mechanism(
        self, noise_mechanism: NoiseMechanism, d_out: PrivacyBudgetInput, f: Callable
    ) -> None:
        """Test error is raised for invalid delta/noise mechanism combination."""
        with self.assertRaises(UnsupportedCombinationError):
            f(noise_mechanism=noise_mechanism, d_out=d_out)

    @parameterized.expand(
        [
            (d_out, f)
            for d_out in [
                (sp.Integer(1), sp.Rational(1, 2)),
                (sp.Integer(1), sp.Rational(1, 3)),
            ]
            for f in [
                functools.partial(
                    create_bounds_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    threshold=0.5,
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_quantile_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    quantile=0.5,
                    upper=10,
                    lower=0,
                    output_measure=ApproxDP(),
                ),
            ]
        ]
    )
    def test_functions_without_noise_mechanism(
        self, d_out: PrivacyBudgetInput, f: Callable
    ) -> None:
        """Test error is raised for invalid deltas."""
        # Quantile and bounds raise plain ValueError (not UnsupportedCombinationError).
        with self.assertRaises(ValueError) as cm:
            f(d_out=d_out)
        self.assertNotIsInstance(cm.exception, UnsupportedCombinationError)


big_test_size = 1000
datasets = [
    pd.DataFrame({"A": ["x1"], "B": [2]}),
    pd.DataFrame(
        {
            "A": random.choices(["x1", "x2", "x3"], k=big_test_size),
            "B": random.sample(range(big_test_size * 10), k=big_test_size),
        }
    ),
]


@pytest.fixture(
    scope="module", params=datasets, ids=["One Row", f"{big_test_size} Rows"]
)
def spark_data(
    request: pytest.FixtureRequest, spark
) -> Generator[Tuple[DataFrame, pd.DataFrame, StructType], None, None]:
    """Fixture producing matching Spark and Pandas dataframes.

    The fixture parameter is expected to be a Pandas dataframe with the schema
    "A: string, B: long".

    Yields: A 3-tuple (spark_df, pandas_df, schema), where spark_df is a Spark
        dataframe created from pandas_df, and schema is the schema of that
        dataframe.
    """
    spark_df = spark.createDataFrame(request.param)
    df_schema = StructType(
        [
            StructField("A", StringType(), nullable=False),
            StructField("B", LongType(), nullable=False),
        ]
    )
    yield spark_df, request.param, df_schema


def test_std(spark_data):
    """Tests that the Pandas std equals Core's std measurement."""
    spark_df, pd_df, df_schema = spark_data

    input_domain = SparkDataFrameDomain.from_spark_schema(df_schema)
    expected = pd_df["B"].std()

    measurement_output = create_standard_deviation_measurement(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        measure_column="B",
        upper=sp.Integer(big_test_size * 10),
        lower=sp.Integer(0),
        output_measure=PureDP(),
        noise_mechanism=NoiseMechanism.LAPLACE,
        d_in=sp.Integer(1),
        d_out=ExactNumber.from_float(float("inf"), round_up=True),
        keep_intermediates=False,
    )(spark_df)

    if expected > 0:
        assert np.isclose(expected, measurement_output)
    else:
        # The std is null for a single data point.
        assert np.isnan(measurement_output)


def test_groupbystd(spark_data, spark):
    """Tests that the Pandas groupby std equals Core's groupby std measurement."""
    spark_df, pd_df, df_schema = spark_data

    input_domain = SparkDataFrameDomain.from_spark_schema(df_schema)
    expected = pd_df.groupby(["A"]).agg({"B": "std"})
    df_keys = pd.DataFrame({"A": pd_df["A"].unique()})

    group_keys = spark.createDataFrame(df_keys, schema=StructType([df_schema["A"]]))

    measurement_output = create_standard_deviation_measurement(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        measure_column="B",
        upper=sp.Integer(big_test_size * 10),
        lower=sp.Integer(0),
        output_measure=PureDP(),
        noise_mechanism=NoiseMechanism.LAPLACE,
        d_in=sp.Integer(1),
        d_out=ExactNumber.from_float(float("inf"), round_up=True),
        keep_intermediates=False,
        groupby_transformation=GroupBy(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            use_l2=False,
            group_keys=group_keys,
        ),
    )(spark_df)

    expected_sorted = (
        expected.reset_index()
        .sort_values("A")
        .rename(columns={"B": "std"})
        .reset_index(drop=True)
    )
    output_sorted = (
        measurement_output.withColumnRenamed("stddev(B)", "std")
        .toPandas()
        .sort_values("A")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(expected_sorted, output_sorted)


def test_var(spark_data):
    """Tests that the Pandas var equals Core's var measurement."""
    spark_df, pd_df, df_schema = spark_data

    input_domain = SparkDataFrameDomain.from_spark_schema(df_schema)
    expected = pd_df["B"].var()

    measurement_output = create_variance_measurement(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        measure_column="B",
        upper=sp.Integer(big_test_size * 10),
        lower=sp.Integer(0),
        output_measure=PureDP(),
        noise_mechanism=NoiseMechanism.LAPLACE,
        d_in=sp.Integer(1),
        d_out=ExactNumber.from_float(float("inf"), round_up=True),
        keep_intermediates=False,
    )(spark_df)

    if expected > 0:
        assert np.isclose(expected, measurement_output)
    else:
        # The variance is null for a single data point.
        assert np.isnan(measurement_output)


def test_groupbyvar(spark_data, spark):
    """Tests that the Pandas groupby var equals Core's groupby var measurement."""
    spark_df, pd_df, df_schema = spark_data

    input_domain = SparkDataFrameDomain.from_spark_schema(df_schema)
    expected = pd_df.groupby(["A"]).agg({"B": "var"})
    df_keys = pd.DataFrame({"A": pd_df["A"].unique()})

    group_keys = spark.createDataFrame(df_keys, schema=StructType([df_schema["A"]]))

    measurement_output = create_variance_measurement(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        measure_column="B",
        upper=sp.Integer(big_test_size * 10),
        lower=sp.Integer(0),
        output_measure=PureDP(),
        noise_mechanism=NoiseMechanism.LAPLACE,
        d_in=sp.Integer(1),
        d_out=ExactNumber.from_float(float("inf"), round_up=True),
        keep_intermediates=False,
        groupby_transformation=GroupBy(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            use_l2=False,
            group_keys=group_keys,
        ),
    )(spark_df)

    expected_sorted = (
        expected.reset_index()
        .sort_values("A")
        .rename(columns={"B": "var"})
        .reset_index(drop=True)
    )
    output_sorted = (
        measurement_output.withColumnRenamed("var(B)", "var")
        .toPandas()
        .sort_values("A")
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_sorted, output_sorted)


class TestBounds:
    """Correctness tests for class :func:`~.create_bounds_measurement`."""

    @pytest.mark.parametrize(
        "measure_domain,pandas_df,threshold,expected_bound",
        [
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": [1, 2, 2, 3, 3, 8]}),
                0.8,
                4,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": [1.0, 2.0, 2.0, 3.0, 3.0, 8.0]}),
                0.8,
                4.0,
            ),
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": [1, 2, 2, 3, 3, 8]}),
                1.0,
                8,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": [1.0, 2.0, 2.0, 3.0, 3.0, 8.0]}),
                1.0,
                8.0,
            ),
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": [16] * 10}),
                0.95,
                16,
            ),
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": [-50, -30, 0, 10, 30, 30, 30, 50, 60, 70]}),
                0.9,
                64,
            ),
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": [16] * 15 + [500] * 5}),
                0.95,
                512,
            ),
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": [-777] * 10}),
                0.95,
                1024,
            ),
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": [-1] * 2 + [0] * 16 + [1] * 2}),
                0.8,
                1,
            ),
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": [-1] * 3 + [0] * 4 + [1] * 3}),
                0.95,
                1,
            ),
            (
                SparkIntegerColumnDescriptor(),
                pd.DataFrame({"X": []}),
                0.95,
                1.0,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": [16.0] * 10}),
                0.95,
                16.0,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame(
                    {"X": [-50.0, -30.0, 0.0, 10.0, 30.0, 30.0, 30.0, 50.0, 60.0, 70.0]}
                ),
                0.9,
                64.0,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": [16.0] * 15 + [500.0] * 5}),
                0.95,
                512.0,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": [-777.0] * 10}),
                0.95,
                1024.0,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": [-(2**-150)] * 2 + [0.0] * 16 + [2**-150] * 2}),
                0.8,
                2**-100,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": [-(2**-99.5)] * 8 + [0.0] * 10 + [2**-99.5] * 8}),
                0.95,
                2**-99,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": []}),
                0.95,
                2**-100,
            ),
            (
                SparkFloatColumnDescriptor(),
                pd.DataFrame({"X": [2.0**101] * 10}),
                0.95,
                2**100,
            ),
        ],
    )
    @pytest.mark.slow
    def test_create_bounds_no_noise(
        self,
        spark,
        measure_domain: SparkColumnDescriptor,
        pandas_df: pd.DataFrame,
        threshold: float,
        expected_bound: Union[int, float],
    ):
        """Tests that create_bounds_measurement is correct when no noise is added."""
        domain = SparkDataFrameDomain(
            {
                "X": measure_domain,
            }
        )
        spark_df = spark.createDataFrame(pandas_df, schema=domain.spark_schema)

        measurement = create_bounds_measurement(
            input_domain=domain,
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            d_out=float("inf"),
            measure_column="X",
            threshold=threshold,
            d_in=1,
        )
        output = measurement(spark_df)

        assert output[1] == expected_bound
        assert measurement.privacy_function(1) == sp.oo

    @pytest.mark.parametrize(
        (
            "measure_domain,input_metric,pandas_df,"
            "threshold,lower_column,upper_column,expected_df"
        ),
        [
            (
                SparkIntegerColumnDescriptor(),
                SymmetricDifference(),
                pd.DataFrame({"A": ["1"] * 6 + ["2"] * 6, "X": [1, 2, 2, 3, 3, 8] * 2}),
                0.8,
                "lower",
                "upper",
                pd.DataFrame({"A": ["1", "2"], "lower": [-4, -4], "upper": [4, 4]}),
            ),
            (
                SparkFloatColumnDescriptor(),
                IfGroupedBy(["A"], SumOf(SymmetricDifference())),
                pd.DataFrame(
                    {"A": ["1"] * 10 + ["2"] * 10, "X": [16.0] * 10 + [1024.0] * 10}
                ),
                0.95,
                None,
                None,
                pd.DataFrame(
                    {
                        "A": ["1", "2", "3"],
                        "lower_bound(X)": [-16.0, -1024.0, -(2.0**-100)],
                        "upper_bound(X)": [16.0, 1024.0, 2.0**-100],
                    }
                ),
            ),
        ],
    )
    @pytest.mark.slow
    def test_create_bounds_with_groupby_no_noise(
        self,
        spark,
        measure_domain: SparkColumnDescriptor,
        input_metric: Union[SymmetricDifference, IfGroupedBy],
        pandas_df: pd.DataFrame,
        threshold: float,
        lower_column: Optional[str],
        upper_column: Optional[str],
        expected_df: pd.DataFrame,
    ):
        """Tests that create_bounds_measurement is correct when no noise is added."""
        domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(),
                "X": measure_domain,
            }
        )
        spark_df = spark.createDataFrame(pandas_df, schema=domain.spark_schema)
        group_keys = spark.createDataFrame(expected_df).select("A")

        measurement = create_bounds_measurement(
            input_domain=domain,
            input_metric=input_metric,
            output_measure=PureDP(),
            d_out=float("inf"),
            measure_column="X",
            threshold=threshold,
            d_in=1,
            groupby_transformation=GroupBy(
                input_domain=domain,
                input_metric=input_metric,
                use_l2=False,
                group_keys=group_keys,
            ),
            lower_bound_column=lower_column,
            upper_bound_column=upper_column,
        )
        output = measurement(spark_df)

        assert_dataframe_equal(output, expected_df)
        assert measurement.privacy_function(1) == sp.oo
