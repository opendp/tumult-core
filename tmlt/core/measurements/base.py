"""Base class for measurements."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from abc import ABC, abstractmethod
from typing import Any

from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.measures import Measure
from tmlt.core.metrics import Metric


class Measurement(ABC):
    """Abstract base class for measurements."""

    @typechecked
    def __init__(
        self,
        input_domain: Domain,
        input_metric: Metric,
        output_measure: Measure,
        is_interactive: bool,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input datasets.
            input_metric: Distance metric for input datasets.
            output_measure: Distance measure for measurement's output.
            is_interactive: Whether the measurement is interactive.
        """
        if not input_metric.supports_domain(input_domain):
            raise ValueError(
                f"Input metric {input_metric} and input domain {input_domain} are not"
                " compatible."
            )
        self._input_domain = input_domain
        self._input_metric = input_metric
        self._output_measure = output_measure
        self._is_interactive = is_interactive

    @property
    def input_domain(self) -> Domain:
        """Return input domain for the measurement."""
        return self._input_domain

    @property
    def input_metric(self) -> Metric:
        """Distance metric on input domain."""
        return self._input_metric

    @property
    def output_measure(self) -> Measure:
        """Distance measure on output."""
        return self._output_measure

    @property
    def is_interactive(self) -> bool:
        """Returns true iff the measurement is interactive."""
        return self._is_interactive

    @typechecked
    def privacy_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the measurement.

        See the privacy and stability tutorial (add link?) for more information.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If not overridden.
        """
        self.input_metric.validate(d_in)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a privacy function"
        )

    @typechecked
    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        """Return True if close inputs produce close outputs.

        See the privacy and stability tutorial (add link?) for more information.

        Args:
            d_in: Distance between inputs under `input_metric`.
            d_out: Distance between outputs under `output_measure`.
        """
        return self.output_measure.compare(self.privacy_function(d_in), d_out)

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Performs measurement."""
        ...
