"""Base class for named, composite transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright the Tumult Core Contributors

from abc import abstractmethod
from typing import Any, final

from typeguard import check_type

from tmlt.core.exceptions import InvalidComponentError
from tmlt.core.transformations.base import Transformation


class DerivedTransformation(Transformation):
    """Base class for transformations with behavior defined by another transformation.

    :class:`.DerivedTransformation` allows the creation of named, parameterizable
    transformation objects whose behavior is defined entirely by another
    transformation that they construct. Its subclasses must implement a
    constructor that builds a transformation and passes it to ``super().__init__``.
    The execution, stability functions, and stability relations of the subclass
    are delegated to that transformation. The subclass may also have additional
    properties, but they are considered metadata.

    ..
        >>> from tmlt.core.domains.collections import DictDomain
        >>> from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
        >>> from tmlt.core.metrics import AbsoluteDifference, DictMetric
        >>> from tmlt.core.transformations.identity import Identity
        >>> from tmlt.core.transformations.dictionary import GetValue

    Example:
        >>> class GetValueChain(DerivedTransformation):
        ...     def __init__(self, input_domain, input_metric, keys):
        ...         xf = Identity(input_metric, input_domain)
        ...         for key in keys:
        ...             xf = xf | GetValue(xf.output_domain, xf.output_metric, key)
        ...         super().__init__(xf)
        >>> xf = GetValueChain(
        ...     DictDomain({"a": DictDomain({"b": NumpyIntegerDomain()})}),
        ...     DictMetric({"a": DictMetric({"b": AbsoluteDifference()})}),
        ...     keys=("a","b"),
        ... )
        >>> xf.output_domain
        NumpyIntegerDomain(size=64)
        >>> xf({"a": {"b": 5}})
        5
    """

    FORMAT_EXCLUDED_ATTRS = Transformation.FORMAT_EXCLUDED_ATTRS | {
        "wrapped_transformation"
    }
    """Fields hidden when formatting this transformation. @nodoc"""

    @abstractmethod
    def __init__(self, wrapped_transformation: Transformation) -> None:
        """Capture the implementation and its stability properties.

        Args:
            wrapped_transformation: Fixed implementation of this transformation.
        """
        wrapped = check_type(wrapped_transformation, Transformation)
        super().__init__(
            input_domain=wrapped.input_domain,
            input_metric=wrapped.input_metric,
            output_domain=wrapped.output_domain,
            output_metric=wrapped.output_metric,
        )
        self.__wrapped_transformation = wrapped

    @property
    @final
    def wrapped_transformation(self) -> Transformation:
        """The transformation defining the derived transformation's behavior."""
        try:
            return self.__wrapped_transformation
        except AttributeError:
            raise InvalidComponentError(
                type(self),
                f"{type(self).__name__}.__init__() must call "
                "super().__init__(wrapped_transformation)",
            ) from None

    @final
    def stability_function(self, d_in: Any) -> Any:
        """Return the wrapped transformation's stability function at ``d_in``."""
        return self.wrapped_transformation.stability_function(d_in)

    @final
    def stability_relation(self, d_in: Any, d_out: Any) -> bool:
        """Evaluate the wrapped transformation's stability relation."""
        return self.wrapped_transformation.stability_relation(d_in, d_out)

    @final
    def __call__(self, data: Any) -> Any:
        """Run the wrapped transformation."""
        return self.wrapped_transformation(data)
