"""Module containing supported variants for differential privacy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from abc import ABC, abstractmethod
from typing import Any, Tuple

from typeguard import check_type

from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.validation import validate_exact_number


class Measure(ABC):
    """Base class for output measures."""

    def __eq__(self, other: Any) -> bool:
        """Return True if both measures are equal."""
        return self.__class__ is other.__class__

    @abstractmethod
    def validate(self, value: Any):
        """Raises an error if `value` not a valid distance.

        Args:
            value: A distance between two probability distributions under this measure.
        """
        ...

    @abstractmethod
    def compare(self, value1: Any, value2: Any) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        ...

    def __repr__(self) -> str:
        """Returns string representation."""
        return f"{self.__class__.__name__}()"


class PureDP(Measure):
    """Pure DP measure."""

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a nonnegative real or infinity

        Args:
            value: A distance between two probability distributions under this measure.
        """
        try:
            validate_exact_number(
                value=value,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid PureDP measure value (epsilon) {e}")

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        self.validate(value1)
        self.validate(value2)
        return ExactNumber(value1) <= ExactNumber(value2)


class ApproxDP(Measure):
    """Approximate DP measure."""

    def validate(self, value: Tuple[ExactNumberInput, ExactNumberInput]):
        """Raises an error if `value` not a valid distance.

        * `value` must be a tuple with two values: (epsilon, delta)
        * epsilon must be a nonnegative real or infinity
        * delta must be a real between 0 and 1 (inclusive)

        Args:
            value: A distance between two probability distributions under this measure.
        """
        try:
            check_type("value", value, Tuple[ExactNumberInput, ExactNumberInput])
            validate_exact_number(
                value=value[0],
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )

            validate_exact_number(
                value=value[1],
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
                maximum=1,
                maximum_is_inclusive=True,
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid ApproxDP measure value (epsilon,delta): {e}")

    def compare(
        self,
        value1: Tuple[ExactNumberInput, ExactNumberInput],
        value2: Tuple[ExactNumberInput, ExactNumberInput],
    ) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        self.validate(value1)
        self.validate(value2)
        epsilon1 = ExactNumber(value1[0])
        delta1 = ExactNumber(value1[1])
        epsilon2 = ExactNumber(value2[0])
        delta2 = ExactNumber(value2[1])
        value2_is_infinite = not epsilon2.is_finite or delta2 == 1
        return value2_is_infinite or epsilon1 <= epsilon2 and delta1 <= delta2


class RhoZCDP(Measure):
    """ρ-zCDP measure.

    See the definition of ρ-zCDP in `this <https://arxiv.org/pdf/1605.02065.pdf>`_ paper
    under Definition 1.1.
    """

    def validate(self, value: ExactNumberInput):
        """Raises an error if `value` not a valid distance.

        * `value` must be a nonnegative real or infinity

        Args:
            value: A distance between two probability distributions under this measure.
        """
        try:
            validate_exact_number(
                value=value,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid RhoZCDP measure value (rho): {e}")

    def compare(self, value1: ExactNumberInput, value2: ExactNumberInput) -> bool:
        """Returns True if `value1` is less than or equal to `value2`."""
        self.validate(value1)
        self.validate(value2)
        return ExactNumber(value1) <= ExactNumber(value2)
