"""Measurements on Pandas DataFrames."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from abc import abstractmethod
from typing import Callable, Dict, Mapping, Optional, Union, cast

import pandas as pd
from pyspark.sql.types import StructField, StructType
from typeguard import check_type, typechecked

from tmlt.core.domains.pandas_domains import PandasDataFrameDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.pandas_measurements.series import (
    Aggregate as AggregateSeries,
)
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import HammingDistance, SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class Aggregate(Measurement):
    """Aggregate a Pandas DataFrame.

    This measurement requires the output schema be specified as a
    :class:`pyspark.sql.types.StructType` so that it
    can be used as a udf in Spark.
    """

    @typechecked
    def __init__(
        self,
        input_domain: PandasDataFrameDomain,
        input_metric: Union[HammingDistance, SymmetricDifference],
        output_measure: Measure,
        output_schema: StructType,
    ):
        """Constructor.

        Args:
            input_domain: Input domain.
            input_metric: Input metric.
            output_measure: Output measure.
            output_schema: Spark StructType compatible with the output.
        """
        self._output_schema = output_schema
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=False,
        )

    @property
    def input_domain(self) -> PandasDataFrameDomain:
        """Return input domain for the measurement."""
        return cast(PandasDataFrameDomain, super().input_domain)

    @property
    def output_schema(self) -> StructType:
        """Return the output schema."""
        return self._output_schema

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform measurement."""
        ...


class AggregateByColumn(Aggregate):
    """Apply Aggregate measurements to columns of a Pandas DataFrame."""

    @typechecked
    def __init__(
        self,
        input_domain: PandasDataFrameDomain,
        column_to_aggregation: Mapping[str, AggregateSeries],
        hint: Optional[
            Callable[[ExactNumberInput, ExactNumberInput], Dict[str, ExactNumberInput]]
        ] = None,
    ):
        """Constructor.

        Args:
            input_domain: Input domain.
            column_to_aggregation: A dictionary mapping column names to aggregation
                measurements. The provided measurements must all have :class:`~.PureDP`
                or all have :class:`~.RhoZCDP` as their
                :attr:`~.Measurement.output_measure`.
            hint: An optional hint. A hint is only required if one or more of the
                measurement's :meth:`~.Measurement.privacy_function` raise
                NotImplementedError. The hint takes in the same arguments as
                :meth:`~.privacy_relation`., and should return a d_out for each
                aggregation to be composed, where all of the d_outs sum to less than the
                d_out passed into the hint.
        """
        if not column_to_aggregation:
            raise ValueError("No aggregations provided.")
        # Check that the aggregation functions are compatible with DataFrame
        # aggregation.
        for column, aggregation_function in column_to_aggregation.items():
            if column not in input_domain.schema:
                raise ValueError(f"Column '{column}' is not in the input schema.")
            if input_domain.schema[column] != aggregation_function.input_domain:
                raise ValueError(
                    "The input domain is not compatible with the input domains of the"
                    " aggregation functions."
                )

        # Check that all aggregation functions have the same input metric, and that
        # it is either SymmetricDifference or HammingDistance.
        input_metric: Optional[Union[SymmetricDifference, HammingDistance]] = None
        for aggregation_function in column_to_aggregation.values():
            if not isinstance(
                aggregation_function.input_metric,
                (SymmetricDifference, HammingDistance),
            ):
                raise ValueError(
                    "The input metric of the aggregation function must be either"
                    " SymmetricDifference or HammingDistance."
                )
            if input_metric is None:
                input_metric = cast(
                    Union[SymmetricDifference, HammingDistance],
                    aggregation_function.input_metric,
                )
            elif aggregation_function.input_metric != input_metric:
                raise ValueError(
                    "All of the aggregation functions must have the same input metric."
                )
        assert input_metric is not None

        # Check that all aggregation functions have the same output measure, and that
        # it is either PureDP or RhoZCDP.
        output_measure: Optional[Union[PureDP, RhoZCDP]] = None
        for aggregation_function in column_to_aggregation.values():
            if not isinstance(aggregation_function.output_measure, (PureDP, RhoZCDP)):
                raise ValueError(
                    "The output measure of the aggregation function must be either"
                    " PureDP or RhoZCDP."
                )
            if output_measure is None:
                output_measure = cast(
                    Union[PureDP, RhoZCDP], aggregation_function.output_measure
                )
            elif aggregation_function.output_measure != output_measure:
                raise ValueError(
                    "All of the aggregation functions must have the same output "
                    "measure."
                )
        assert output_measure is not None

        # Construct the output schema. Use the ordering of the aggregation columns in
        # column_to_aggregation.
        output_schema = StructType(
            [
                StructField(column, aggregation.output_spark_type)
                for column, aggregation in column_to_aggregation.items()
            ]
        )

        self._column_to_aggregation = dict(column_to_aggregation)
        check_type(
            "column_to_aggregation",
            self._column_to_aggregation,
            Dict[str, AggregateSeries],
        )
        self._hint = hint
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            output_schema=output_schema,
        )

    @property
    def column_to_aggregation(self) -> Dict[str, AggregateSeries]:
        """Returns dictionary from column names to aggregation measurements."""
        return self._column_to_aggregation.copy()

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        Returns the sum of the :meth:`~.Measurement.privacy_function`'s on `d_in` for
        all composed measurements.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If any of the measurements raise
                :class:`NotImplementedError`.
        """
        # Note: This is using sequential composition.
        self.input_metric.validate(d_in)
        d_outs = [
            measurement.privacy_function(d_in)
            for measurement in self.column_to_aggregation.values()
        ]
        return cast(ExactNumber, sum(d_outs))

    @typechecked
    def privacy_relation(self, d_in: ExactNumberInput, d_out: ExactNumberInput) -> bool:
        """Returns True only if outputs are close under close inputs.

        Let `d_outs` be the d_out from the :meth:`~.Measurement.privacy_function`'s of
        all composed measurements or the d_outs from the hint if one of them raises
        :class:`NotImplementedError`.

        And `total_d_out` to be the sum of `d_outs`.

        This returns True if `total_d_out` <= `d_out` (the input argument) and each
        composed measurement satisfies its :meth:`~.Measurement.privacy_relation` from
        `d_in` to its d_out from `d_outs`.

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_measure.
        """
        # Note: This is using sequential composition.
        try:
            return super().privacy_relation(d_in, d_out)
        except NotImplementedError as e:
            if self._hint is None:
                raise ValueError(
                    "A hint is needed to check this privacy relation, because the "
                    "privacy_relation from one of self.column_to_aggregation.values() "
                    f"raised a NotImplementedError: {e}"
                )
        assert self._hint is not None
        d_outs = self._hint(d_in, d_out)
        if set(d_outs) != set(self.column_to_aggregation):
            raise ValueError(
                "The columns produced by the hint function don't match"
                " the columns to be aggregated."
            )
        return self.output_measure.compare(
            sum(ExactNumber(d_out_i) for d_out_i in d_outs.values()), d_out
        ) and all(
            self.column_to_aggregation[column].privacy_relation(d_in, d_out_i)
            for column, d_out_i in d_outs.items()
        )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform the aggregation.

        Args:
            df: The DataFrame to aggregate.
        """
        return pd.DataFrame(
            {
                column_name: [aggregation(df[column_name])]
                for column_name, aggregation in self.column_to_aggregation.items()
            }
        )
