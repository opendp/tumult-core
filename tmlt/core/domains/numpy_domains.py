"""Domains for NumPy datatypes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

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
            np.dtype("int8"): NumpyIntegerDomain(size=32),
            np.dtype("int16"): NumpyIntegerDomain(size=32),
            np.dtype("int32"): NumpyIntegerDomain(size=32),
            np.dtype("int64"): NumpyIntegerDomain(size=64),
            np.dtype("float32"): NumpyFloatDomain(size=32),
            np.dtype("float64"): NumpyFloatDomain(size=64),
            np.dtype("object"): NumpyStringDomain(),
            np.dtype("bool"): NumpyIntegerDomain(size=32),
        }
        return pd_to_numpy_domain[dtype]


@dataclass(frozen=True)
class NumpyIntegerDomain(NumpyDomain):
    """Domain of NumPy integers."""

    size: int = 64
    """Number of bits a member of the domain occupies. Must be 32 or 64."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("size", self.size, int)
        if self.size not in [32, 64]:
            raise ValueError(f"size must be 32 or 64, not {self.size}")

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for elements in the domain."""
        if self.size == 32:
            return np.int32
        return np.int64


@dataclass(frozen=True)
class NumpyFloatDomain(NumpyDomain):
    """Domain of NumPy floats."""

    allow_nan: bool = False
    """If True, NaNs are permitted in the domain."""
    allow_inf: bool = False
    """If True, infs are permitted in the domain."""
    size: int = 64
    """Number of bits a member of the domain occupies. Must be 32 or 64."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("allow_nan", self.allow_nan, bool)
        check_type("allow_inf", self.allow_inf, bool)
        check_type("size", self.size, int)
        if self.size not in [32, 64]:
            raise ValueError(f"size must be 32 or 64, not {self.size}")

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for elements in the domain."""
        if self.size == 32:
            return np.float32
        return np.float64

    def validate(self, value: Any):
        """Raises error if value is not a member of the domain."""
        super().validate(value)
        if not self.allow_inf and np.isinf(value):
            raise OutOfDomainError("Value is infinite.")
        if not self.allow_nan and np.isnan(value):
            raise OutOfDomainError("Value is NaN.")


@dataclass(frozen=True)
class NumpyStringDomain(NumpyDomain):
    """Domain of NumPy strings.

    Note:
        This domain does not use :class:`numpy.str_` datatype. Instead, native python
        objects are used since Pandas uses this by default.
    """

    allow_null: bool = False
    """If True, None is allowed."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("allow_null", self.allow_null, bool)

    def validate(self, value: Any):
        """Raises error if value is not in domain."""
        if not isinstance(value, self.carrier_type) and value.__class__ is not str:
            raise OutOfDomainError(
                f"Value must be {str}, instead it is {value.__class__}."
            )
        if value is None and not self.allow_null:
            raise OutOfDomainError("Value is null.")

    @property
    def carrier_type(self) -> type:  # pylint: disable=no-self-use
        """Returns carrier types for members of NumpyStringDomain."""
        return object
