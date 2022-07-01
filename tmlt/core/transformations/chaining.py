"""Transformations constructed by chaining other transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Callable, Optional

from typeguard import typechecked

from tmlt.core.transformations.base import Transformation


class ChainTT(Transformation):
    """Transformation constructed by chaining two transformations."""

    @typechecked
    def __init__(
        self,
        transformation1: Transformation,
        transformation2: Transformation,
        hint: Optional[Callable[[Any, Any], Any]] = None,
    ):
        """Constructor.

        Args:
            transformation1: Transformation to apply first.
            transformation2: Transformation to apply second.
            hint: An optional function to compute the intermediate metric value
                (after the first transformation, but before the second) for
                :meth:`~.stability_relation`. It takes in the same inputs
                as :meth:`~.stability_relation`, and is only required if
                the transformation's :meth:`~.Transformation.stability_function` raises
                :class:`NotImplementedError`.
        """
        if transformation1.output_domain != transformation2.input_domain:
            raise ValueError("Can not chain transformations: Mismatching domains.")
        if transformation1.output_metric != transformation2.input_metric:
            raise ValueError("Can not chain transformations: Mismatching metrics.")
        super().__init__(
            input_domain=transformation1.input_domain,
            input_metric=transformation1.input_metric,
            output_domain=transformation2.output_domain,
            output_metric=transformation2.output_metric,
        )
        self._transformation1 = transformation1
        self._transformation2 = transformation2
        self._hint = hint

    @typechecked
    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        Returns M.privacy_function(T.stability_function(d_in)).

        where:

        * T1 is the first transformation applied (:attr:`~.transformation1`)
        * T2 is the second transformation applied (:attr:`~.transformation2`)

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If T2.stability_function(T1.stability_function(d_in))
             raises :class:`NotImplementedError`.
        """
        return self.transformation2.stability_function(
            self.transformation1.stability_function(d_in)
        )

    @typechecked
    def stability_relation(self, d_in: Any, d_out: Any) -> bool:
        """Returns True only if outputs are close under close inputs.

        Let d_mid = T1.stability_function(d_in), or hint(d_in, d_out) if
        T1.stability_function raises :class:`NotImplementedError`.

        This returns True only if the following hold:

        (1) T1.stability_relation(d_in, d_mid)
        (2) T2.stability_relation(d_mid, d_out)

        where:

        * T1 is the first transformation applied (:attr:`~.transformation1`)
        * T2 is the second transformation applied (:attr:`~.transformation2`)
        * hint is the hint passed to the constructor.

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_metric.
        """
        self.input_metric.validate(d_in)
        self.output_metric.validate(d_out)
        try:
            d_mid = self.transformation1.stability_function(d_in)
        except NotImplementedError as e:
            if self._hint is None:
                raise ValueError(
                    "A hint is needed to check this privacy relation, because the "
                    "stability_relation of self.transformation1 raised a "
                    f"NotImplementedError: {e}"
                )
            d_mid = self._hint(d_in, d_out)
        return self.transformation1.stability_relation(
            d_in, d_mid
        ) and self.transformation2.stability_relation(d_mid, d_out)

    @property
    def transformation1(self) -> Transformation:
        """Returns the first transformation being applied."""
        return self._transformation1

    @property
    def transformation2(self) -> Transformation:
        """Returns the second transformation being applied."""
        return self._transformation2

    def __call__(self, data: Any) -> Any:
        """Performs transformation1 followed by transformation2."""
        return self._transformation2(self._transformation1(data))
