"""Unit tests for :mod:`~tmlt.core.measurements.pandas_measurements.dataframe`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use
import itertools
import unittest
from typing import Dict, Optional, Union
from unittest.mock import MagicMock

import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark.sql.types import DataType, DoubleType, FloatType, StructField, StructType

from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.measurements.pandas_measurements.dataframe import AggregateByColumn
from tmlt.core.measurements.pandas_measurements.series import Aggregate, NoisyQuantile
from tmlt.core.measures import ApproxDP, Measure, PureDP, RhoZCDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    HammingDistance,
    Metric,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import (
    assert_property_immutability,
    create_mock_measurement,
    get_all_props,
)


def _create_mock_aggregate(
    input_domain: Optional[PandasSeriesDomain] = None,
    input_metric: Optional[Metric] = None,
    output_measure: Optional[Measure] = None,
    output_spark_type: Optional[DataType] = None,
) -> Aggregate:
    """Returns a mock :class:`~.Aggregate` object."""
    if input_domain is None:
        input_domain = PandasSeriesDomain(NumpyIntegerDomain())
    if input_metric is None:
        input_metric = SymmetricDifference()
    if output_measure is None:
        output_measure = PureDP()
    if output_spark_type is None:
        output_spark_type = FloatType()
    mock = create_mock_measurement(
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=output_measure,
    )
    mock.output_spark_type = output_spark_type
    return mock


class TestAggregateByColumn(unittest.TestCase):
    """Tests measurements on pandas dataframes.

    Tests :class:`~tmlt.core.measurements.pandas_measurements.dataframe.Aggregate`.
    """

    @parameterized.expand(get_all_props(AggregateByColumn))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        quantile_measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=1,
        )
        column_to_aggregation: Dict[str, Aggregate] = {
            "B": quantile_measurement,
            "A": quantile_measurement,
        }
        measurement = AggregateByColumn(
            input_domain=PandasDataFrameDomain(
                {
                    "A": PandasSeriesDomain(NumpyIntegerDomain()),
                    "B": PandasSeriesDomain(NumpyIntegerDomain()),
                    "C": PandasSeriesDomain(NumpyStringDomain()),
                }
            ),
            column_to_aggregation=column_to_aggregation,
        )
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """AggregateByColumn's properties have the expected values."""
        quantile_measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=1,
        )
        input_domain = PandasDataFrameDomain(
            {
                "A": PandasSeriesDomain(NumpyIntegerDomain()),
                "B": PandasSeriesDomain(NumpyIntegerDomain()),
                "C": PandasSeriesDomain(NumpyStringDomain()),
            }
        )
        column_to_aggregation: Dict[str, Aggregate] = {
            "B": quantile_measurement,
            "A": quantile_measurement,
        }
        measurement = AggregateByColumn(
            input_domain=input_domain, column_to_aggregation=column_to_aggregation
        )
        self.assertEqual(measurement.input_domain, input_domain)
        self.assertEqual(measurement.input_metric, SymmetricDifference())
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.column_to_aggregation, column_to_aggregation)

    def test_output_schema(self):
        """Test that the output schema is constructed correctly."""
        quantile_measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=1,
        )
        column_to_aggregation: Dict[str, Aggregate] = {
            "B": quantile_measurement,
            "A": quantile_measurement,
        }
        measure = AggregateByColumn(
            input_domain=PandasDataFrameDomain(
                {
                    "A": PandasSeriesDomain(NumpyIntegerDomain()),
                    "B": PandasSeriesDomain(NumpyIntegerDomain()),
                    "C": PandasSeriesDomain(NumpyStringDomain()),
                }
            ),
            column_to_aggregation=column_to_aggregation,
        )

        expected_schema = StructType(
            [StructField("B", DoubleType()), StructField("A", DoubleType())]
        )
        self.assertEqual(expected_schema, measure.output_schema)

    @parameterized.expand(
        [
            (  # no aggregations
                {},
                {"A": PandasSeriesDomain(NumpyIntegerDomain())},
                "No aggregations provided.",
            ),
            (  # mismatching input domains
                {
                    "A": _create_mock_aggregate(
                        input_domain=PandasSeriesDomain(NumpyFloatDomain())
                    )
                },
                {"A": PandasSeriesDomain(NumpyIntegerDomain())},
                "The input domain is not compatible with the input domains of the"
                " aggregation functions.",
            ),
            (  # mismatching column names
                {"A": _create_mock_aggregate()},
                {"B": PandasSeriesDomain(NumpyIntegerDomain())},
                "Column 'A' is not in the input schema.",
            ),
            (  # invalid input metric
                {"A": _create_mock_aggregate(input_metric=AbsoluteDifference())},
                {"A": PandasSeriesDomain(NumpyIntegerDomain())},
                "The input metric of the aggregation function must be either "
                "SymmetricDifference or HammingDistance.",
            ),
            (  # inconsistent input metrics
                {
                    "A": _create_mock_aggregate(input_metric=SymmetricDifference()),
                    "B": _create_mock_aggregate(input_metric=HammingDistance()),
                },
                {
                    "A": PandasSeriesDomain(NumpyIntegerDomain()),
                    "B": PandasSeriesDomain(NumpyIntegerDomain()),
                },
                "All of the aggregation functions must have the same input metric.",
            ),
            (  # invalid output measure
                {"A": _create_mock_aggregate(output_measure=ApproxDP())},
                {"A": PandasSeriesDomain(NumpyIntegerDomain())},
                "The output measure of the aggregation function must be either PureDP "
                "or RhoZCDP.",
            ),
            (  # inconsistent output measures
                {
                    "A": _create_mock_aggregate(output_measure=PureDP()),
                    "B": _create_mock_aggregate(output_measure=RhoZCDP()),
                },
                {
                    "A": PandasSeriesDomain(NumpyIntegerDomain()),
                    "B": PandasSeriesDomain(NumpyIntegerDomain()),
                },
                "All of the aggregation functions must have the same output measure.",
            ),
        ]
    )
    def test_input_validation(
        self,
        column_to_aggregation: Dict[str, Aggregate],
        schema: Dict[str, PandasSeriesDomain],
        expected_error_message: str,
    ):
        """Init correctly validates aggregation functions."""
        with self.assertRaisesRegex(ValueError, expected_error_message):
            AggregateByColumn(
                input_domain=PandasDataFrameDomain(schema),
                column_to_aggregation=column_to_aggregation,
            )

    def test_correctness_quantile(self):
        """Test correctness for a quantile aggregation and infinite budget."""
        df = pd.DataFrame([[28, 23], [26, 22], [27, 24], [29, 25]], columns=["F", "M"])
        qs = [0, 1]
        quantile_measurements = [
            NoisyQuantile(
                PandasSeriesDomain(NumpyIntegerDomain()),
                output_measure=PureDP(),
                quantile=q,
                lower=22,
                upper=29,
                epsilon=sp.oo,
            )
            for q in qs
        ]
        column_to_aggregation: Dict[str, Aggregate] = {
            "F": quantile_measurements[0],
            "M": quantile_measurements[1],
        }
        measurement = AggregateByColumn(
            input_domain=PandasDataFrameDomain(
                {
                    "F": PandasSeriesDomain(NumpyIntegerDomain()),
                    "M": PandasSeriesDomain(NumpyIntegerDomain()),
                }
            ),
            column_to_aggregation=column_to_aggregation,
        )
        output_df = measurement(df)

        self.assertTrue(22 < output_df.loc[0, "F"] <= 26)
        self.assertTrue(25 < output_df.loc[0, "M"] <= 29)
        self.assertEqual(measurement.privacy_function(1), sp.oo)
        self.assertTrue(measurement.privacy_relation(1, sp.oo))

    @parameterized.expand(
        [
            (*params1, *params2, input_metric, use_hint)
            for params1, params2 in itertools.combinations(
                [
                    (False, ExactNumber(1), True),
                    (False, ExactNumber(2), False),
                    (True, ExactNumber(1), True),
                    (True, ExactNumber(2), False),
                ],
                2,
            )
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for use_hint in [True, False]
        ]
    )
    def test_privacy_function_and_relation(
        self,
        privacy_function_implemented1: bool,
        privacy_function_return_value1: ExactNumber,
        privacy_relation_return_value1: bool,
        privacy_function_implemented2: bool,
        privacy_function_return_value2: ExactNumber,
        privacy_relation_return_value2: bool,
        input_metric: Union[SymmetricDifference, HammingDistance],
        use_hint: bool,
    ):
        """Tests that the privacy function and relation work correctly."""
        mock_measurement1 = create_mock_measurement(
            input_domain=PandasSeriesDomain(element_domain=NumpyFloatDomain()),
            input_metric=input_metric,
            privacy_function_implemented=privacy_function_implemented1,
            privacy_function_return_value=privacy_function_return_value1,
            privacy_relation_return_value=privacy_relation_return_value1,
        )
        mock_measurement1.output_spark_type = FloatType()
        mock_measurement2 = create_mock_measurement(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            input_metric=input_metric,
            privacy_function_implemented=privacy_function_implemented2,
            privacy_function_return_value=privacy_function_return_value2,
            privacy_relation_return_value=privacy_relation_return_value2,
        )
        mock_measurement2.output_spark_type = FloatType()
        mock_hint = MagicMock(return_value={"A": 1, "B": 1})
        measurement = AggregateByColumn(
            input_domain=PandasDataFrameDomain(
                {
                    "A": PandasSeriesDomain(NumpyFloatDomain()),
                    "B": PandasSeriesDomain(NumpyIntegerDomain()),
                }
            ),
            column_to_aggregation={"A": mock_measurement1, "B": mock_measurement2},
            hint=mock_hint if use_hint else None,
        )
        if not (privacy_function_implemented1 and privacy_function_implemented2):
            with self.assertRaises(NotImplementedError):
                measurement.privacy_function(1)
        else:
            self.assertEqual(
                measurement.privacy_function(1),
                privacy_function_return_value1 + privacy_function_return_value2,
            )
        if (
            not (privacy_function_implemented1 and privacy_function_implemented2)
            and not use_hint
        ):
            self.assertRaisesRegex(
                ValueError,
                "A hint is needed to check this privacy relation, because the "
                "privacy_relation from one of self.column_to_aggregation.values() "
                "raised a NotImplementedError: TEST",
            )
        else:
            self.assertEqual(
                measurement.privacy_relation(1, 2),
                mock_measurement1.privacy_relation(1, 1)
                and mock_measurement2.privacy_relation(1, 1),
            )
            if mock_hint.called:
                mock_hint.assert_called_with(1, 2)
            self.assertFalse(measurement.privacy_relation(1, "1.99"))
