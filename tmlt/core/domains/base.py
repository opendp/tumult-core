"""Base class for input/output domains."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from abc import ABC, abstractmethod
from typing import Any


class OutOfDomainError(Exception):
    """Exception type that indicates a validation error in a Domain."""


class Domain(ABC):
    """Base class for input/output domains."""

    @property
    @abstractmethod
    def carrier_type(self) -> type:
        """Returns the type of elements in the domain."""
        ...

    def validate(self, value: Any):
        """Raises an error if value is not in the domain."""
        if value.__class__ is not self.carrier_type:
            raise OutOfDomainError(
                f"Value must be {self.carrier_type}, instead it is {value.__class__}."
            )

    def __contains__(self, value: Any) -> bool:
        """Returns True if value is in the domain."""
        try:
            self.validate(value)
        except OutOfDomainError:
            return False
        return True
