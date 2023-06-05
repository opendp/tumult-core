"""Base class for input/output domains."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class OutOfDomainError(Exception):
    """Exception type that indicates a validation error in a Domain."""

    def __init__(self, domain: Domain, value: Any, msg: str):
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
    """Exception type raised when two or more domains should match, but don't."""

    def __init__(self, domains: Iterable[Domain], msg: str):
        """Constructor.

        Args:
            domains: The domains that do not match.
            msg: The error message.
        """
        self.domains = domains
        super().__init__(msg)


class UnsupportedDomainError(TypeError):
    """Exception type that indicates that a given domain is not supported."""

    def __init__(self, domain: Domain, msg: str):
        """Constructor.

        Args:
            domain: The domain that is not supported.
            msg: The error message.
        """
        self.domain = domain
        super().__init__(msg)


class Domain(ABC):
    """Base class for input/output domains."""

    @property
    @abstractmethod
    def carrier_type(self) -> type:
        """Returns the type of elements in the domain."""

    def validate(self, value: Any):
        """Raises an error if value is not in the domain."""
        if value.__class__ is not self.carrier_type:
            raise OutOfDomainError(
                self,
                value,
                f"Value must be {self.carrier_type}, instead it is {value.__class__}.",
            )

    def __contains__(self, value: Any) -> bool:
        """Returns True if value is in the domain."""
        try:
            self.validate(value)
        except OutOfDomainError:
            return False
        return True
