"""Transformations for persisting and un-persisting Spark DataFrames.

#TODO(#1653): Add link to the Spark Abstraction page.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any

from pyspark.sql import DataFrame
from typeguard import typechecked

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import Metric
from tmlt.core.transformations.base import Transformation


class Persist(Transformation):
    """Persists a Spark DataFrame.

    This is an identity transformation that marks the input Spark DataFrame to be
    stored when evaluated by Spark.

    Note:
        This transformation does not eagerly evaluate and store the input DataFrame.
        Spark only stores it when an action (like collect) is performed. If you want
        to persist eagerly, chain this transformation with a :class:`~.SparkAction`.
    """

    @typechecked
    def __init__(self, domain: SparkDataFrameDomain, metric: Metric):
        """Constructor.

        Args:
            metric: Input/Output metric.
            domain: Input/Output domain.
        """
        super().__init__(
            input_domain=domain,
            input_metric=metric,
            output_domain=domain,
            output_metric=metric,
        )

    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        The returned d_out is d_in.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return d_in

    def __call__(self, data: DataFrame) -> DataFrame:
        """Returns input DataFrame."""
        return data.persist()


class Unpersist(Transformation):
    """Unpersists a Spark DataFrame.

    This is an identity transformation that marks a persisted DataFrame to
    be evicted. If the input DataFrame is not persisted, this has no effect.
    """

    @typechecked
    def __init__(self, domain: SparkDataFrameDomain, metric: Metric):
        """Constructor.

        Args:
            metric: Input/Output metric.
            domain: Input/Output domain.
        """
        super().__init__(
            input_domain=domain,
            input_metric=metric,
            output_domain=domain,
            output_metric=metric,
        )

    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        The returned d_out is d_in.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return d_in

    def __call__(self, data: DataFrame) -> DataFrame:
        """Returns input DataFrame."""
        return data.unpersist()


class SparkAction(Transformation):
    r"""Triggers an action on a Spark DataFrame.

    This is intended to be used after :class:`~.Persist` to eagerly
    evaluate and store a :class:`~.Transformation`\ 's output.
    """

    @typechecked
    def __init__(self, domain: SparkDataFrameDomain, metric: Metric):
        """Constructor.

        Args:
            metric: Input/Output metric.
            domain: Input/Output domain.
        """
        super().__init__(
            input_domain=domain,
            input_metric=metric,
            output_domain=domain,
            output_metric=metric,
        )

    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        The returned d_out is d_in.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return d_in

    def __call__(self, data: DataFrame) -> DataFrame:
        """Returns input DataFrame."""
        data.count()
        return data
