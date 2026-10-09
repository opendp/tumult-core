"""Small examples for discussing the derived-component prototype."""

# SPDX-License-Identifier: Apache-2.0

from typing import Any

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.measurements.derived import DerivedMeasurement
from tmlt.core.measurements.noise_mechanisms import AddLaplaceNoise
from tmlt.core.measurements.postprocess import PostProcess
from tmlt.core.metrics import AbsoluteDifference, DictMetric
from tmlt.core.transformations.derived import DerivedTransformation
from tmlt.core.transformations.dictionary import create_rename
from tmlt.core.transformations.identity import Identity
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class DictRename(DerivedTransformation):

    def __init__(
        self, input_domain: DictDomain, input_metric: DictMetric, key: Any, new_key: Any
    ) -> None:
        self._key = key
        self._new_key = new_key
        # In practice, we would move the create_rename code into this class,
        # potentially leaving behind a stub for backwards-compatibility.
        super().__init__(create_rename(input_domain, input_metric, key, new_key))

    @property
    def key(self) -> Any:
        return self._key

    @property
    def new_key(self) -> Any:
        return self._new_key


class RoundedLaplace(DerivedMeasurement):

    def __init__(self, scale: ExactNumberInput) -> None:
        self._scale = ExactNumber(scale)
        super().__init__(
            PostProcess(AddLaplaceNoise(NumpyIntegerDomain(), self.scale), round)
        )

    @property
    def scale(self) -> ExactNumber:
        return self._scale

if __name__ == "__main__":
    rename = DictRename(
        DictDomain({"a": NumpyIntegerDomain()}),
        DictMetric({"a": AbsoluteDifference()}),
        "a",
        "b",
    )
    print(rename({"a": 10}))
    print(rename.stability_function({"a": 1}))
    print(rename.format())
    print(rename.wrapped_transformation.format())

    print()

    rounded = RoundedLaplace(scale=2)
    print(rounded.privacy_function(1))
    print(rounded.format())
