"""Special exceptions raised by Core in certain situations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Collection, Iterable, Union

import sympy as sp

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

# TYPE_CHECKING is True when MyPy analyzes this file, but False at runtime
# (see: https://docs.python.org/3/library/typing.html)
# This allows importing modules for type-checking only,
# which prevents this file from creating circular imports.
if TYPE_CHECKING:
    import tmlt.core.domains.base
    import tmlt.core.measurements.aggregations
    import tmlt.core.measures
    import tmlt.core.metrics


class OutOfDomainError(Exception):
    """Exception type that indicates a validation error in a Domain.

    Attributes:
        domain: The :class:`~tmlt.core.domains.base.Domain` on which the exception
            was raised.
        value: The value that is not in the domain.
    """

    def __init__(self, domain: "tmlt.core.domains.base.Domain", value: Any, msg: str):
        """Constructor.

        Args:
            domain: The domain on which the exception was raised.
            value: The value that is not in the domain.
            msg: The error message.
        """
        self.domain = domain
        self.value = value
        super().__init__(msg)


class DomainMismatchError(ValueError):
    """Exception type raised when two or more domains should match, but don't.

    Attributes:
        domains: An ``Iterable`` of all the  :class:`~tmlt.core.domains.base.Domain`
            that should match, but do not.
    """

    def __init__(self, domains: Iterable["tmlt.core.domains.base.Domain"], msg: str):
        """Constructor.

        Args:
            domains: The domains that do not match.
            msg: The error message.
        """
        self.domains = domains
        super().__init__(msg)


class UnsupportedDomainError(TypeError):
    """Exception type that indicates that a given domain is not supported.

    Attributes:
        domain: The :class:`~tmlt.core.domains.base.Domain` that is not supported.
    """

    def __init__(self, domain: "tmlt.core.domains.base.Domain", msg: str):
        """Constructor.

        Args:
            domain: The domain that is not supported.
            msg: The error message.
        """
        self.domain = domain
        super().__init__(msg)


class DomainKeyError(Exception):
    """Exception type that indicates that a key is not in the given domain.

    Attributes:
        domain: The :class:`~tmlt.core.domains.base.Domain` on which
            this error was raised.
        key: The key that was not in the domain, or a collection of keys that are
            not in the domain.
    """

    def __init__(self, domain: "tmlt.core.domains.base.Domain", key: Any, msg: str):
        """Constructor.

        Args:
            domain: The domain on which this error was raised.
            key: The key that's not in the domain (or a collection of keys that
                aren't in the domain).
            msg: The error message.
        """
        self.key = key
        self.domain = domain
        super().__init__(domain, key, msg)


class DomainColumnError(Exception):
    """Exception type for when a column is not in the given domain's schema."""

    def __init__(
        self,
        domain: "tmlt.core.domains.base.Domain",
        column: Union[str, Collection[str]],
        msg: str,
    ):
        """Constructor.

        Args:
            domain: The domain on which this error was raised.
            column: The column that's not in the domain's schema, or a collection of
                columns that are not in the domain's schema.
            msg: The error message.
        """
        self.domain = domain
        self.column = column
        super().__init__(msg)


class UnsupportedMetricError(ValueError):
    """Exception raised when a given metric is not supported.

    Attributes:
        metric: The metric that is not supported.
    """

    def __init__(self, metric: "tmlt.core.metrics.Metric", msg: str):
        """Constructor.

        Args:
          metric: The metric that is not supported.
          msg: The error message.
        """
        self.metric = metric
        super().__init__(msg)


class MetricMismatchError(ValueError):
    """Exception raised when two or more metrics should match, but don't.

    Attributes:
        metrics: The metrics that should match, but do not.
    """

    def __init__(self, metrics: Iterable["tmlt.core.metrics.Metric"], msg: str):
        """Constructor.

        Args:
            metrics: The metrics that do not match.
            msg: The error message.
        """
        self.metrics = metrics
        super().__init__(msg)


class UnsupportedCombinationError(ValueError):
    """Raised when a given combination of values is not supported.

    If this exception is raised, any of the values used may be *individually*
    valid, but they are not valid *in this particular combination*.

    For example, when a metric does not support a given domain, this is the
    error that is raised.

    Attributes:
        combination: The combination of values that are not supported.
    """

    def __init__(self, combination: Iterable[Any], msg: str):
        """Constructor.

        Args:
            combination: The combination of values that is not supported.
            msg: The error message.
        """
        self.combination = combination
        super().__init__(msg)


class UnsupportedMeasureError(ValueError):
    """Error raised when a given measure is not supported.

    Attributes:
        measure: The :class:`~tmlt.core.measures.Measure` that is not supported.
    """

    def __init__(self, measure: "tmlt.core.measures.Measure", msg: str):
        """Constructor.

        Args:
            measure: The measure that is not supported.
            msg: The error message.
        """
        self.measure = measure
        super().__init__(msg)


class MeasureMismatchError(ValueError):
    """Error raised when two or more measures should match, but don't.

    Attributes:
        measures: An :class:`~Iterable` with multiple
            :class:`~tmlt.core.measures.Measure` that should match, but don't.
    """

    def __init__(self, measures: Iterable["tmlt.core.measures.Measure"], msg: str):
        """Constructor.

        Args:
            measures: The measures that should match, but do not.
            msg: The error message.
        """
        self.measures = measures
        super().__init__(msg)


class UnsupportedNoiseMechanismError(ValueError):
    """Error raised when the given noise mechanism is not supported.

    Attributes:
        noise_mechanism: The
            :class:`~tmlt.core.measurements.aggregations.NoiseMechanism` that is
            not supported.
    """

    def __init__(
        self,
        noise_mechanism: "tmlt.core.measurements.aggregations.NoiseMechanism",
        msg: str,
    ):
        """Constructor.

        Args:
            noise_mechanism: The noise mechanism that is not supported.
            msg: The error message.
        """
        self.noise_mechanism = noise_mechanism
        super().__init__(msg)


class UnsupportedSympyExprError(ValueError):
    """Error raised when trying to use an unsupported Sympy expression.

    This error is raised when an
    :class:`~tmlt.core.utils.exact_number.ExactNumber` cannot be created from a given
    SymPy expression. This might be raised because the expression corresponds
    to an imaginary number, because the expression has a free variable, or
    because the expression is a kind of Sympy expression that ExactNumber
    does not support (such as sums).

    Attributes:
        expr: the invalid Sympy expression.
    """

    def __init__(self, expr: sp.Expr, msg: str):
        """Constructor.

        Args:
            expr: The invalid Sympy expression.
            msg: The error message.
        """
        self.expr = expr
        super().__init__(msg)
