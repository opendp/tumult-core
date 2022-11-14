"""Module containing supported variants for differential privacy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, cast

from typeguard import check_type, typechecked

from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.validation import validate_exact_number

PrivacyBudgetInput = Union[ExactNumberInput, Tuple[ExactNumberInput, ExactNumberInput]]
PrivacyBudgetValue = Union[ExactNumber, Tuple[ExactNumber, ExactNumber]]


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


class PrivacyBudget(ABC):
    """An abstract class for representing a privacy budget.

    This class is meant to allow operations on a budget (e.g. subtracting, checking if
    the budget is infinite) without needing to know the type of the budget.
    """

    @classmethod
    @typechecked
    def cast(
        cls, measure: Union[PureDP, ApproxDP, RhoZCDP], value: PrivacyBudgetInput
    ) -> "PrivacyBudget":
        """Return a privacy budget matching the passed measure.

        Args:
            measure: The measure to return a privacy budget for.
            value: The value of the privacy budget.
        """
        if isinstance(measure, PureDP):
            return PureDPBudget(value)
        if isinstance(measure, ApproxDP):
            return ApproxDPBudget(value)
        assert isinstance(measure, RhoZCDP)
        return RhoZCDPBudget(value)

    @property
    @abstractmethod
    def value(self) -> PrivacyBudgetValue:
        """Return the value of the privacy budget."""
        ...

    @abstractmethod
    def is_finite(self) -> bool:
        """Return true iff the budget is finite."""
        ...

    @abstractmethod
    def can_spend_budget(self, other: PrivacyBudgetInput) -> bool:
        """Return true iff we can spend budget `other`.

        Args:
            other: The privacy budget we would like to spend.
        """
        ...

    @abstractmethod
    def subtract(self, other: PrivacyBudgetInput) -> "PrivacyBudget":
        """Return a new budget after subtracting `other`.

        If the budget represented by this class is infinite, return the current budget.

        Args:
            other: The privacy budget to subtract.

        Raises:
            ValueError: If there is not enough privacy budget to subtract other.
        """
        ...

    def __eq__(self, other: Any) -> bool:
        """Check is this instance is equal to `other`.

        Args:
            other: The other instance.
        """
        return self.__class__ is other.__class__ and self.value == other.value


class PureDPBudget(PrivacyBudget):
    """A pure dp budget."""

    def __init__(self, value: PrivacyBudgetInput) -> None:
        """Initialize.

        Args:
            value: The value of the privacy budget.
        """
        PureDP().validate(cast(ExactNumberInput, value))
        self._epsilon = ExactNumber(cast(ExactNumberInput, value))

    @property
    def value(self) -> ExactNumber:
        """Return the value of the privacy budget."""
        return self._epsilon

    @property
    def epsilon(self) -> ExactNumber:
        """The pure dp privacy loss."""
        return self._epsilon

    def is_finite(self) -> bool:
        """Return true iff the budget is finite."""
        return self._epsilon.is_finite

    def can_spend_budget(self, other: PrivacyBudgetInput) -> bool:
        """Return true iff we can spend budget `other`.

        Args:
            other: The privacy budget we would like to spend.
        """
        validated_other = PureDPBudget(other)
        return self._epsilon >= validated_other.epsilon

    def subtract(self, other: PrivacyBudgetInput) -> "PureDPBudget":
        """Return a new budget after subtracting `other`.

        If the budget represented by this class is infinite, return the current budget.

        Args:
            other: The privacy budget to subtract.

        Raises:
            ValueError: If there is not enough privacy budget to subtract other.
        """
        validated_other = PureDPBudget(other)
        if not self.can_spend_budget(other):
            raise ValueError("Cannot subtract a larger budget from a smaller budget.")
        if not self.is_finite():
            return self
        return PureDPBudget(self._epsilon - validated_other.epsilon)


class ApproxDPBudget(PrivacyBudget):
    """An approximate dp budget."""

    def __init__(self, value: PrivacyBudgetInput) -> None:
        """Initialize.

        Args:
            value: The value of the privacy budget.
        """
        ApproxDP().validate(cast(Tuple[ExactNumber, ExactNumber], value))
        typed_value = cast(Tuple[ExactNumberInput, ExactNumberInput], value)
        self._epsilon = ExactNumber(typed_value[0])
        self._delta = ExactNumber(typed_value[1])

    @property
    def value(self) -> Tuple[ExactNumber, ExactNumber]:
        """Return the value of the privacy budget."""
        return (self._epsilon, self._delta)

    @property
    def epsilon(self) -> ExactNumber:
        """The first component of the privacy loss."""
        return self._epsilon

    @property
    def delta(self) -> ExactNumber:
        """The second component of the privacy loss."""
        return self._delta

    def is_finite(self) -> bool:
        """Return true iff the budget is finite."""
        return self._epsilon.is_finite and self._delta < 1

    def can_spend_budget(self, other: PrivacyBudgetInput) -> bool:
        """Return true iff we can spend budget `other`.

        Args:
            other: The privacy budget we would like to spend.
        """
        validated_other = ApproxDPBudget(other)
        return not self.is_finite() or (
            self._epsilon >= validated_other.epsilon
            and self._delta >= validated_other.delta
        )

    def subtract(self, other: PrivacyBudgetInput) -> "ApproxDPBudget":
        """Return a new budget after subtracting `other`.

        If the budget represented by this class is infinite, return the current budget.

        Args:
            other: The privacy budget to subtract.

        Raises:
            ValueError: If there is not enough privacy budget to subtract other.
        """
        validated_other = ApproxDPBudget(other)
        if not self.is_finite():
            return self
        if not self.can_spend_budget(other):
            raise ValueError("Cannot subtract a larger budget from a smaller budget.")
        return ApproxDPBudget(
            (
                self._epsilon - validated_other.epsilon,
                self._delta - validated_other.delta,
            )
        )


class RhoZCDPBudget(PrivacyBudget):
    """A zCDP budget."""

    def __init__(self, value: PrivacyBudgetInput) -> None:
        """Initialize.

        Args:
            value: The value of the privacy budget.
        """
        RhoZCDP().validate(cast(ExactNumberInput, value))
        self._rho = ExactNumber(cast(ExactNumberInput, value))

    @property
    def value(self) -> ExactNumber:
        """Return the value of the privacy budget."""
        return self._rho

    @property
    def rho(self) -> ExactNumber:
        """The zCDP privacy loss."""
        return self._rho

    def is_finite(self) -> bool:
        """Return true iff the budget is finite."""
        return self._rho.is_finite

    def can_spend_budget(self, other: PrivacyBudgetInput) -> bool:
        """Return true iff we can spend budget `other`.

        Args:
            other: The privacy budget we would like to spend.
        """
        validated_other = RhoZCDPBudget(other)
        return self._rho >= validated_other.rho

    def subtract(self, other: PrivacyBudgetInput) -> "RhoZCDPBudget":
        """Return a new budget after subtracting `other`.

        If the budget represented by this class is infinite, return the current budget.

        Args:
            other: The privacy budget to subtract.

        Raises:
            ValueError: If there is not enough privacy budget to subtract other.
        """
        validated_other = RhoZCDPBudget(other)
        if not self.can_spend_budget(other):
            raise ValueError("Cannot subtract a larger budget from a smaller budget.")
        if not self.is_finite():
            return self
        return RhoZCDPBudget(self._rho - validated_other.rho)
