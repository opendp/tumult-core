"""Domains for common python collections such as lists and dictionaries."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from dataclasses import dataclass
from typing import Any, Dict, Optional

from typeguard import check_type, typechecked

from tmlt.core.domains.base import Domain, OutOfDomainError


@dataclass(frozen=True, eq=True)
class ListDomain(Domain):
    """Domain of lists of elements of a particular domain."""

    element_domain: Domain
    """Domain of list elements."""

    length: Optional[int] = None
    """Number of elements in lists in the domain. If None, this is unrestricted."""

    def __post_init__(self):
        """Check inputs to constructor."""
        check_type("element_domain", self.element_domain, Domain)
        check_type("length", self.length, Optional[int])

    @property
    def carrier_type(self) -> type:
        """Returns python list type."""
        return list

    def validate(self, value: Any):
        """Raises error if value is not a row with matching schema."""
        super().validate(value)
        for elem in value:
            try:
                self.element_domain.validate(elem)
            except OutOfDomainError as exception:
                raise OutOfDomainError(f"Found invalid value in list: {exception}")


class DictDomain(Domain):
    """Domain of dictionaries."""

    @typechecked
    def __init__(self, key_to_domain: Dict[Any, Domain]):
        """Constructor.

        Args:
            key_to_domain: Mapping from key to domain.
        """
        self._key_to_domain = key_to_domain.copy()

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"{self.__class__.__name__}(key_to_domain={self.key_to_domain})"

    def __eq__(self, other: Any) -> bool:
        """Returns True if both domains are identical."""
        if other.__class__ != self.__class__:
            return False
        return self.key_to_domain == other.key_to_domain

    @property
    def key_to_domain(self) -> Dict[Any, Domain]:
        """Returns dictionary mapping each key in the domain with its domain."""
        return self._key_to_domain.copy()

    @property
    def length(self) -> int:
        """Returns number of keys in the domain."""
        return len(self.key_to_domain)

    def __getitem__(self, key: Any) -> Domain:
        """Returns domain associated with given key."""
        return self.key_to_domain[key]

    @property
    def carrier_type(self) -> type:
        """Returns the type of elements in the domain."""
        return dict

    def validate(self, value: Any):
        """Raises error if value is not in the domain."""
        super().validate(value)
        value_keys, domain_keys = sorted(set(value)), sorted(set(self.key_to_domain))
        if value_keys != domain_keys:
            raise OutOfDomainError(
                "Keys are not as expected, value must match domain.\nValue "
                f"keys: {value_keys}\nDomain keys: {domain_keys}"
            )

        for key in value:
            try:
                self.key_to_domain[key].validate(value[key])
            except OutOfDomainError as exception:
                raise OutOfDomainError(f"Found invalid value at '{key}': {exception}")
