"""Base class for measurements."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026
from abc import ABC, abstractmethod
from typing import Any, FrozenSet

from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.measures import Measure
from tmlt.core.metrics import Metric, UnsupportedCombinationError
from tmlt.core.utils.format import default_format_attrs, default_format_children


class Measurement(ABC):
    """Abstract base class for measurements."""

    _FORMAT_EXCLUDED_ATTRS: FrozenSet[str] = frozenset(
        {"input_domain", "input_metric", "output_measure", "is_interactive"}
    )
    """Fields hidden from output when formatting this measurement."""

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
            raise UnsupportedCombinationError(
                (input_metric, input_domain),
                (
                    f"Input metric {input_metric} and input domain {input_domain} are"
                    " not compatible."
                ),
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
            d_in: Distance between inputs under ``input_metric``.
            d_out: Distance between outputs under ``output_measure``.
        """
        return self.output_measure.compare(self.privacy_function(d_in), d_out)

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Performs measurement."""

    def format(self) -> str:
        """Return a human-readable multi-line description of this measurement.

        The default implementation assembles :meth:`_format_head` and
        :meth:`_format_children`; subclasses can override either of these
        hooks (or :meth:`format` itself) to customize the rendering.
        """
        head = self._format_head()
        children = self._format_children()
        if not children:
            return head
        return f"{head}\n{children}"

    def _format_head(self) -> str:
        """Render this measurement's head line: class name followed by its attrs."""
        parts = [type(self).__name__]
        parts.extend(
            f"{name}={value}"
            for name, value in default_format_attrs(self, self._FORMAT_EXCLUDED_ATTRS)
        )
        return " ".join(parts)

    def _format_children(self) -> str:
        """Return the rendered block for nested transformations/measurements."""
        return default_format_children(self)
