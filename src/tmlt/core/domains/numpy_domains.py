"""Domains for NumPy datatypes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Core Contributors 2025-present

from dataclasses import dataclass
from typing import Any

import numpy as np
from typeguard import check_type

from tmlt.core.domains.base import Domain, OutOfDomainError


class NumpyDomain(Domain):
    """Base class for NumpyDomains."""

    @classmethod
    def from_np_type(cls, dtype: np.dtype) -> "NumpyDomain":
        """Returns a NumPy domain from a Pandas type."""
        pd_to_numpy_domain = {
            np.dtype("int64"): NumpyIntegerDomain(),
            np.dtype("float64"): NumpyFloatDomain(),
            np.dtype("object"): NumpyStringDomain(),
        }
        if dtype not in pd_to_numpy_domain:
            raise ValueError(
                f"Unsupported dtype {dtype}. Integers and floats must be 64-bit; cast"
                " them to int64 or float64 first."
            )
        return pd_to_numpy_domain[dtype]


@dataclass(frozen=True)
class NumpyIntegerDomain(NumpyDomain):
    """Domain of 64-bit NumPy integers."""

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for elements in the domain."""
        return np.int64


@dataclass(frozen=True)
class NumpyFloatDomain(NumpyDomain):
    """Domain of 64-bit NumPy floats."""

    allow_nan: bool = False
    """If True, NaNs are permitted in the domain."""
    allow_inf: bool = False
    """If True, infs are permitted in the domain."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.allow_nan, bool)
        check_type(self.allow_inf, bool)

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for elements in the domain."""
        return np.float64

    def validate(self, value: Any) -> None:
        """Raises error if value is not a member of the domain."""
        super().validate(value)
        if not self.allow_inf and np.isinf(value):
            raise OutOfDomainError(self, value, "Value is infinite.")
        if not self.allow_nan and np.isnan(value):
            raise OutOfDomainError(self, value, "Value is NaN.")


@dataclass(frozen=True)
class NumpyStringDomain(NumpyDomain):
    """Domain of NumPy strings.

    Note:
        This domain does not use :class:`numpy.str_` datatype. Instead, native python
        objects are used since Pandas uses this by default.
    """

    allow_null: bool = False
    """If True, None is allowed."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.allow_null, bool)

    def validate(self, value: Any) -> None:
        """Raises error if value is not in domain."""
        if not isinstance(value, self.carrier_type) and value.__class__ is not str:
            raise OutOfDomainError(
                self, value, f"Value must be {str}, instead it is {value.__class__}."
            )
        if value is None and not self.allow_null:
            raise OutOfDomainError(self, value, "Value is null.")

    @property
    def carrier_type(self) -> type:
        """Returns carrier types for members of NumpyStringDomain."""
        return object
