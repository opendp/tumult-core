"""Identity transformation."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any

from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.metrics import Metric
from tmlt.core.transformations.base import Transformation


class Identity(Transformation):
    """Identity transformation."""

    @typechecked
    def __init__(self, metric: Metric, domain: Domain):
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

    @typechecked
    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        The returned d_out is d_in.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return d_in

    def __call__(self, data: Any) -> Any:
        """Return data."""
        return data
