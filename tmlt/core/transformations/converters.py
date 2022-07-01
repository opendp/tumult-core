"""Wrappers for changing a transformation's output metric."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any

from pyspark.sql import DataFrame
from typeguard import typechecked

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class UnwrapIfGroupedBy(Transformation):
    """No-op transformation for switching from IfGroupedBy to its inner metric."""

    @typechecked
    def __init__(self, domain: SparkDataFrameDomain, input_metric: IfGroupedBy):
        """Constructor.

        Args:
            domain: Domain of input DataFrames.
            input_metric: IfGroupedBy metric on input DataFrames.
        """
        if not input_metric.column in domain.schema:
            raise ValueError(
                f"Invalid IfGroupedBy metric: {input_metric.column} not in domain"
            )
        if not isinstance(input_metric.inner_metric, (SumOf, RootSumOfSquared)):
            raise ValueError(
                "Inner metric for IfGroupedBy metric must be "
                "SumOf(SymmetricDifference()), or "
                "RootSumOfSquared(SymmetricDifference())"
            )
        self._is_l2 = isinstance(input_metric.inner_metric, RootSumOfSquared)
        super().__init__(
            input_domain=domain,
            input_metric=input_metric,
            output_domain=domain,
            output_metric=input_metric.inner_metric.inner_metric,
        )

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the transformation.

        If the inner metric of the :class:`~.IfGroupedBy` input metric is a
        :class:`~.SumOf`, returns `d_in`.

        If the inner metric is :class:`~.RootSumOfSquared`, returns `d_in`\*\*2.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        d_in = ExactNumber(d_in)
        if self._is_l2:
            return d_in ** 2
        return d_in

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Returns DataFrame unchanged."""
        return sdf


class HammingDistanceToSymmetricDifference(Transformation):
    """No-op transformation for switching metrics."""

    @typechecked
    def __init__(self, domain: SparkDataFrameDomain):
        """Constructor."""
        super().__init__(
            input_domain=domain,
            input_metric=HammingDistance(),
            output_domain=domain,
            output_metric=SymmetricDifference(),
        )

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Returns 2 * d_in

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(2) * d_in

    def __call__(self, data: Any):
        """Returns unchanged input."""
        return data
