"""Base class for input/output domains."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tmlt.core.exceptions import OutOfDomainError
from tmlt.core.utils.misc import get_fullname


class Domain(ABC):
    """Base class for input/output domains."""

    @property
    @abstractmethod
    def carrier_type(self) -> type:
        """Returns the type of elements in the domain."""

    def validate(self, value: Any) -> None:
        """Raises an error if value is not in the domain."""
        if value.__class__ is not self.carrier_type:
            raise OutOfDomainError(
                self,
                value,
                f"Value must be {get_fullname(self.carrier_type)}, instead it is "
                f"{get_fullname(value.__class__)}.",
            )

    def __contains__(self, value: Any) -> bool:
        """Returns True if value is in the domain."""
        try:
            self.validate(value)
        except OutOfDomainError:
            return False
        return True
