"""Domains for common python collections such as lists and dictionaries."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from typeguard import check_type, typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.exceptions import OutOfDomainError
from tmlt.core.utils.misc import get_fullname


@dataclass(frozen=True, eq=True)
class ListDomain(Domain):
    """Domain of lists of elements of a particular domain."""

    element_domain: Domain
    """Domain of list elements."""

    length: Optional[int] = None
    """Number of elements in lists in the domain. If None, this is unrestricted."""

    def __post_init__(self) -> None:
        """Check inputs to constructor."""
        check_type(self.element_domain, Domain)
        check_type(self.length, Optional[int])
        if self.length is not None and self.length < 0:
            raise ValueError("length must be non-negative")

    @property
    def carrier_type(self) -> type:
        """Returns python list type."""
        return list

    def validate(self, value: Any) -> None:
        """Raises error if value is not a row with matching schema."""
        super().validate(value)
        if self.length is not None and len(value) != self.length:
            raise OutOfDomainError(
                self,
                value,
                f"Expected list of length {self.length}, "
                f"found list of length {len(value)}",
            )
        for elem in value:
            try:
                self.element_domain.validate(elem)
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    self, value, f"Found invalid value in list: {exception}"
                ) from exception


class DictDomain(Domain):
    """Domain of dictionaries."""

    @typechecked
    def __init__(self, key_to_domain: Mapping[Any, Domain]):
        """Constructor.

        Args:
            key_to_domain: Mapping from key to domain.
        """
        self._key_to_domain: Dict[Any, Domain] = dict(key_to_domain.items())
        # TODO(#2727): Remove this check once we update typeguard to ^3.0.0
        for key, domain in self._key_to_domain.items():
            if not isinstance(domain, Domain):
                raise TypeError(
                    f"Expected domain for key '{key}' to be a {get_fullname(Domain)}; "
                    f"got {get_fullname(domain)} instead"
                )

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

    def validate(self, value: Any) -> None:
        """Raises error if value is not in the domain."""
        super().validate(value)
        value_keys, domain_keys = set(value), set(self.key_to_domain)
        if value_keys != domain_keys:
            raise OutOfDomainError(
                self,
                value_keys,
                (
                    "Keys are not as expected, value must match domain.\nValue "
                    f"keys: {value_keys}\nDomain keys: {domain_keys}"
                ),
            )

        for key in value:
            try:
                self.key_to_domain[key].validate(value[key])
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    self, value, f"Found invalid value at '{key}': {exception}"
                ) from exception
