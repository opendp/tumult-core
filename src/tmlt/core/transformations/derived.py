"""Prototype base class for named, composite transformations."""

# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Any, final

from typeguard import check_type

from tmlt.core.transformations.base import Transformation


class DerivedTransformation(Transformation):
    """Wrap a fixed transformation with a name and optional custom properties."""

    FORMAT_EXCLUDED_ATTRS = Transformation.FORMAT_EXCLUDED_ATTRS | {
        "wrapped_transformation"
    }

    @abstractmethod
    def __init__(self, wrapped_transformation: Transformation) -> None:
        """Capture the implementation and its stability properties."""
        wrapped = check_type(wrapped_transformation, Transformation)
        super().__init__(
            input_domain=wrapped.input_domain,
            input_metric=wrapped.input_metric,
            output_domain=wrapped.output_domain,
            output_metric=wrapped.output_metric,
        )
        self._wrapped_transformation = wrapped

    @final
    @property
    def wrapped_transformation(self) -> Transformation:
        """Return the fixed implementation for inspection."""
        try:
            return self._wrapped_transformation
        except AttributeError:
            raise RuntimeError(
                f"{type(self).__name__}.__init__() must call "
                "super().__init__(wrapped_transformation)."
            ) from None

    @final
    def stability_function(self, d_in: Any) -> Any:
        """Delegate to the wrapped transformation."""
        return self.wrapped_transformation.stability_function(d_in)

    @final
    def stability_relation(self, d_in: Any, d_out: Any) -> bool:
        """Delegate even when the wrapped stability function is unavailable."""
        return self.wrapped_transformation.stability_relation(d_in, d_out)

    @final
    def __call__(self, data: Any) -> Any:
        """Run the wrapped transformation."""
        return self.wrapped_transformation(data)
