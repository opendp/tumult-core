"""Base class for transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from abc import ABC, abstractmethod
from typing import Any, Union, overload

from typeguard import check_type, typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.measurements.base import Measurement
from tmlt.core.metrics import Metric


class Transformation(ABC):
    """Abstract base class for transformations."""

    @typechecked
    def __init__(
        self,
        input_domain: Domain,
        input_metric: Metric,
        output_domain: Domain,
        output_metric: Metric,
    ):
        """Base constructor for transformations."""
        if not input_metric.supports_domain(input_domain):
            raise ValueError(
                f"Input metric {input_metric} and input domain {input_domain} are not"
                " compatible."
            )
        if not output_metric.supports_domain(output_domain):
            raise ValueError(
                f"Output metric {output_metric} and output domain {output_domain} are"
                " not compatible."
            )
        self._input_domain = input_domain
        self._input_metric = input_metric
        self._output_domain = output_domain
        self._output_metric = output_metric

    @property
    def input_domain(self) -> Domain:
        """Return input domain for the measurement."""
        return self._input_domain

    @property
    def input_metric(self) -> Metric:
        """Distance metric on input domain."""
        return self._input_metric

    @property
    def output_domain(self) -> Domain:
        """Return input domain for the measurement."""
        return self._output_domain

    @property
    def output_metric(self) -> Metric:
        """Distance metric on input domain."""
        return self._output_metric

    @typechecked
    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial (add link?) for more information.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If not overridden.
        """
        self.input_metric.validate(d_in)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a stability function"
        )
        return d_in  # pylint: disable=unreachable

    @typechecked
    def stability_relation(self, d_in: Any, d_out: Any) -> bool:
        """Returns True only if close inputs produce close outputs.

        See the privacy and stability tutorial (add link?) for more information.

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_metric.
        """
        min_d_out = self.stability_function(d_in)
        if min_d_out is NotImplemented:
            raise NotImplementedError()
        return self.output_metric.compare(min_d_out, d_out)

    @overload
    def __or__(
        self, other: "Transformation"
    ) -> "Transformation":  # noqa: D105 https://github.com/PyCQA/pydocstyle/issues/525
        ...

    @overload
    def __or__(
        self, other: Measurement
    ) -> Measurement:  # noqa: D105 https://github.com/PyCQA/pydocstyle/issues/525
        ...

    def __or__(self, other):
        """Return this transformation chained with another component."""
        # pylint: disable=import-outside-toplevel
        check_type("other", other, Union[Measurement, Transformation])
        if isinstance(other, Measurement):
            from tmlt.core.measurements.chaining import ChainTM

            return ChainTM(transformation=self, measurement=other)
        assert isinstance(other, Transformation)
        from tmlt.core.transformations.chaining import ChainTT

        return ChainTT(transformation1=self, transformation2=other)

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Perform transformation."""
        ...
