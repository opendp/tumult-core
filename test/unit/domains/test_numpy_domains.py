"""Unit tests for :mod:`~tmlt.core.domains.numpy`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from contextlib import nullcontext as does_not_raise
from itertools import combinations_with_replacement
from test.unit.domains.abstract import DomainTests
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import numpy as np
import pytest
from typeguard import TypeCheckError

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.utils.misc import get_fullname

_float_domains = {
    "base_float_domain": NumpyFloatDomain(),
    "size_float_domain": NumpyFloatDomain(size=32),
    "nan_float_domain": NumpyFloatDomain(allow_nan=True),
    "inf_float_domain": NumpyFloatDomain(allow_inf=True),
    "nan_inf_float_domain": NumpyFloatDomain(allow_nan=True, allow_inf=True),
}


class TestNumpyIntegerDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.numpy_domains.NumpyIntegerDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return NumpyIntegerDomain

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            ({}, does_not_raise(), None),
            ({"size": 32}, does_not_raise(), None),
            ({"size": 64}, does_not_raise(), None),
            (
                {"size": 43},
                pytest.raises(ValueError, match=f"size must be {32} or {64}, not {43}"),
                None,
            ),
            (
                {"size": 32.0},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"size": "32"},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
        ],
    )
    def test_construct_component(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The domain is constructed correctly and raises exceptions when initialized with
        invalid inputs.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        super().test_construct_component(
            domain_type, domain_args, expectation, exception_properties
        )

    @pytest.mark.parametrize(
        "domain, other_domain, expected",
        [
            (NumpyIntegerDomain(), NumpyFloatDomain(), False),
            (NumpyIntegerDomain(size=32), NumpyFloatDomain(), False),
            (NumpyIntegerDomain(), NumpyIntegerDomain(), True),
            (NumpyIntegerDomain(size=32), NumpyIntegerDomain(), False),
        ],
    )
    def test_eq(self, domain: Domain, other_domain: Domain, expected: bool):
        """__eq__ works correctly.

        Args:
            domain: The domain to test.
            other_domain: The domain to compare to.
            expected: The expected result of the comparison.
        """
        super().test_eq(domain, other_domain, expected)

    @pytest.mark.skip("No arguments to mutate")
    @pytest.mark.parametrize("domain_args, key, mutator", [])
    def test_mutable_inputs(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        key: str,
        mutator: Callable[[Any], Any],
    ):
        """The mutable inputs to the domain are copied.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        super().test_mutable_inputs(domain_type, domain_args, key, mutator)

    @pytest.mark.parametrize(
        "domain, expected_properties",
        [
            (NumpyIntegerDomain(), {"size": 64, "carrier_type": np.int64}),
            (NumpyIntegerDomain(size=32), {"size": 32, "carrier_type": np.int32}),
            (NumpyIntegerDomain(size=64), {"size": 64, "carrier_type": np.int64}),
        ],
    )
    def test_properties(self, domain: Domain, expected_properties: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            domain: The constructed domain to be tested.
            expected_properties: A dictionary containing all the property:value pairs
                domain is expected to have.
        """
        super().test_properties(domain, expected_properties)

    @pytest.mark.parametrize(
        "domain", [NumpyIntegerDomain(), NumpyIntegerDomain(size=32)]
    )
    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        super().test_property_immutability(domain)

    @pytest.mark.parametrize(
        "domain, candidate, expectation, exception_properties",
        [
            (NumpyIntegerDomain(), np.int64(1), does_not_raise(), None),
            (
                NumpyIntegerDomain(),
                np.int32(1),
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(np.int64)}, instead it is "
                    f"{get_fullname(np.int32)}",
                ),
                {"domain": NumpyIntegerDomain(), "value": np.int32(1)},
            ),
            (
                NumpyIntegerDomain(),
                np.float64(1),
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(np.int64)}, instead it is "
                    f"{get_fullname(np.float64)}",
                ),
                {"domain": NumpyIntegerDomain(), "value": np.float64(1)},
            ),
            (
                NumpyIntegerDomain(size=32),
                np.int64(1),
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(np.int32)}, instead it is "
                    f"{get_fullname(np.int64)}",
                ),
                {"domain": NumpyIntegerDomain(size=32), "value": np.int64(1)},
            ),
            (
                NumpyIntegerDomain(size=32),
                1,
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(np.int32)}, instead it is "
                    f"{get_fullname(int)}",
                ),
                {"domain": NumpyIntegerDomain(size=32), "value": 1},
            ),
            (NumpyIntegerDomain(size=32), np.int32(1), does_not_raise(), None),
            (
                NumpyIntegerDomain(size=32),
                np.float64(1),
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(np.int32)}, instead it is "
                    f"{get_fullname(np.float64)}",
                ),
                {"domain": NumpyIntegerDomain(size=32), "value": np.float64(1)},
            ),
        ],
    )
    def test_validate(
        self,
        domain: Domain,
        candidate: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Validate works correctly.

        Args:
            domain: The domain to test.
            candidate: The value to validate using domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        super().test_validate(domain, candidate, expectation, exception_properties)

    @pytest.mark.parametrize(
        "dtype, expected, expectation",
        [
            (np.dtype(np.int64), NumpyIntegerDomain(), does_not_raise()),
            (np.dtype(np.int32), NumpyIntegerDomain(size=32), does_not_raise()),
            (
                np.dtype([("f1", np.int64)]),
                None,
                pytest.raises(KeyError, match=f"{np.dtype([('f1', np.int64)])}"),
            ),
        ],
    )
    def test_from_np_type(
        self,
        domain_type: Type[NumpyIntegerDomain],
        dtype: np.dtype,
        expected: NumpyIntegerDomain,
        expectation: ContextManager[None],
    ):
        """from_np_type works correctly.

        Args:
            domain_type: The type of domain to be constructed.
            dtype: The dtype to test.
            expected: The expected domain to be constructed.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert domain_type.from_np_type(dtype) == expected


class TestNumpyFloatDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.numpy_domains.NumpyFloatDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return NumpyFloatDomain

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            ({}, does_not_raise(), None),
            ({"size": 32}, does_not_raise(), None),
            ({"size": 64}, does_not_raise(), None),
            (
                {"size": 128},
                pytest.raises(
                    ValueError, match=f"size must be {32} or {64}, not {128}"
                ),
                None,
            ),
            ({"allow_inf": True, "allow_nan": True}, does_not_raise(), None),
            (
                {"allow_inf": True, "allow_nan": None},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"allow_inf": "False", "allow_nan": True},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"allow_inf": True, "allow_nan": 0.1},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"allow_inf": True, "allow_nan": True, "size": "tom"},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"allow_inf": True, "allow_nan": True, "size": np.int64(32)},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"allow_inf": True, "allow_nan": True, "size": 128},
                pytest.raises(
                    ValueError, match=f"size must be {32} or {64}, not {128}"
                ),
                None,
            ),
        ],
    )
    def test_construct_component(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The domain is constructed correctly and raises exceptions when initialized with
        invalid inputs.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        super().test_construct_component(
            domain_type, domain_args, expectation, exception_properties
        )

    @pytest.mark.parametrize(
        "domain, other_domain, expected",
        [
            (_float_domains[base_key], _float_domains[other_key], base_key == other_key)
            for (base_key, other_key) in combinations_with_replacement(
                _float_domains.keys(), 2
            )
        ],
    )
    def test_eq(self, domain: Domain, other_domain: Domain, expected: bool):
        """__eq__ works correctly.

        Args:
            domain: The domain to test.
            other_domain: The domain to compare to.
            expected: The expected result of the comparison.
        """
        super().test_eq(domain, other_domain, expected)

    @pytest.mark.skip("No arguments to mutate")
    @pytest.mark.parametrize("domain_args, key, mutator", [])
    def test_mutable_inputs(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        key: str,
        mutator: Callable[[Any], Any],
    ):
        """The mutable inputs to the domain are copied.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        super().test_mutable_inputs(domain_type, domain_args, key, mutator)

    @pytest.mark.parametrize(
        "domain, expected_properties",
        [
            (
                NumpyFloatDomain(),
                {
                    "size": 64,
                    "allow_nan": False,
                    "allow_inf": False,
                    "carrier_type": np.float64,
                },
            ),
            (
                NumpyFloatDomain(size=32),
                {
                    "size": 32,
                    "allow_nan": False,
                    "allow_inf": False,
                    "carrier_type": np.float32,
                },
            ),
            (
                NumpyFloatDomain(allow_nan=True),
                {
                    "size": 64,
                    "allow_nan": True,
                    "allow_inf": False,
                    "carrier_type": np.float64,
                },
            ),
            (
                NumpyFloatDomain(allow_inf=True),
                {
                    "size": 64,
                    "allow_nan": False,
                    "allow_inf": True,
                    "carrier_type": np.float64,
                },
            ),
        ],
    )
    def test_properties(self, domain: Domain, expected_properties: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            domain: The constructed domain to be tested.
            expected_properties: A dictionary containing all the property:value pairs
                domain is expected to have.
        """
        super().test_properties(domain, expected_properties)

    @pytest.mark.parametrize(
        "domain",
        [
            NumpyFloatDomain(),
            NumpyFloatDomain(size=32),
            NumpyFloatDomain(allow_nan=True),
            NumpyFloatDomain(allow_inf=True),
            NumpyFloatDomain(allow_nan=True, allow_inf=True),
        ],
    )
    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        super().test_property_immutability(domain)

    @pytest.mark.parametrize(
        "domain, candidate, expectation, exception_properties",
        [
            # 32 bit float cases
            (
                NumpyFloatDomain(size=32),
                to_validate,
                pytest.raises(OutOfDomainError, match=match),
                {"domain": NumpyFloatDomain(size=32), "value": to_validate},
            )
            for to_validate, match in [
                (
                    np.float64(1.0),
                    f"Value must be {get_fullname(np.float32)}, instead it is "
                    f"{get_fullname(np.float64)}.",
                ),
                (np.float32(float("inf")), "Value is infinite."),
                (np.float32(-float("inf")), "Value is infinite."),
                # nan != nan, so we ignore the value
                # (np.float32(float("nan")), "Value is NaN."),
                (
                    1,
                    f"Value must be {get_fullname(np.float32)}, instead it is "
                    f"{get_fullname(int)}.",
                ),
            ]
        ]
        + [
            # 64 should not accept 32
            (
                domain,
                np.float32(1.0),
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(np.float64)}, instead it is "
                    f"{get_fullname(np.float32)}.",
                ),
                {"domain": domain, "value": np.float32(1.0)},
            )
            for domain in [
                NumpyFloatDomain(),
                NumpyFloatDomain(allow_nan=True),
                NumpyFloatDomain(allow_inf=True),
                NumpyFloatDomain(allow_nan=True, allow_inf=True),
            ]
        ]
        + [
            # Only allow inf if allow_inf is True
            (
                domain,
                np.float64(float("inf")),
                pytest.raises(OutOfDomainError, match="Value is infinite."),
                {"domain": domain, "value": np.float64(float("inf"))},
            )
            for domain in [NumpyFloatDomain(), NumpyFloatDomain(allow_nan=True)]
        ]
        + [
            # Same thing should happen for negative infinities
            (
                domain,
                np.float64(-float("inf")),
                pytest.raises(OutOfDomainError, match="Value is infinite."),
                {"domain": domain, "value": np.float64(-float("inf"))},
            )
            for domain in [NumpyFloatDomain(), NumpyFloatDomain(allow_nan=True)]
        ]
        + [
            # Only allow nan if allow_nan is True
            (
                domain,
                np.float64(float("nan")),
                pytest.raises(OutOfDomainError, match="Value is NaN."),
                # nan != nan, so we ignore the value
                {"domain": domain},  # , "value": np.float64(float("nan"))},
            )
            for domain in [NumpyFloatDomain(), NumpyFloatDomain(allow_inf=True)]
        ]
        + [
            # Should reject non float values
            (
                domain,
                1,
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(np.float64)}, instead it is "
                    f"{get_fullname(int)}.",
                ),
                {"domain": domain, "value": 1},
            )
            for domain in [
                NumpyFloatDomain(),
                NumpyFloatDomain(allow_nan=True),
                NumpyFloatDomain(allow_inf=True),
                NumpyFloatDomain(allow_nan=True, allow_inf=True),
            ]
        ]
        + [
            # Should reject regular python floats
            (
                domain,
                1.0,
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(np.float64)}, instead it is "
                    f"{get_fullname(float)}.",
                ),
                {"domain": domain, "value": 1.0},
            )
            for domain in [
                NumpyFloatDomain(),
                NumpyFloatDomain(allow_nan=True),
                NumpyFloatDomain(allow_inf=True),
                NumpyFloatDomain(allow_nan=True, allow_inf=True),
            ]
        ],
    )
    def test_validate(
        self,
        domain: Domain,
        candidate: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Validate works correctly.

        Args:
            domain: The domain to test.
            candidate: The value to validate using domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        super().test_validate(domain, candidate, expectation, exception_properties)

    @pytest.mark.parametrize(
        "dtype, expected, expectation",
        [
            (np.dtype(np.float64), NumpyFloatDomain(), does_not_raise()),
            (np.dtype(np.float32), NumpyFloatDomain(size=32), does_not_raise()),
            (
                np.dtype([("f1", np.int64)]),
                None,
                pytest.raises(KeyError, match=f"{np.dtype([('f1', np.int64)])}"),
            ),
        ],
    )
    def test_from_np_type(
        self,
        domain_type: Type[NumpyFloatDomain],
        dtype: np.dtype,
        expected: NumpyFloatDomain,
        expectation: ContextManager[None],
    ):
        """from_np_type works correctly.

        Args:
            domain_type: The type of domain to be constructed.
            dtype: The dtype to test.
            expected: The expected domain to be constructed.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert domain_type.from_np_type(dtype) == expected


class TestNumpyStringDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.numpy_domains.NumpyStringDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return NumpyStringDomain

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {"allow_null": 23},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            )
        ],
    )
    def test_construct_component(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The domain is constructed correctly and raises exceptions when initialized with
        invalid inputs.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        super().test_construct_component(
            domain_type, domain_args, expectation, exception_properties
        )

    @pytest.mark.parametrize(
        "domain, other_domain, expected",
        [
            (
                NumpyStringDomain(allow_null=True),
                NumpyStringDomain(allow_null=True),
                True,
            ),
            (
                NumpyStringDomain(allow_null=True),
                NumpyStringDomain(allow_null=False),
                False,
            ),
            (
                NumpyStringDomain(allow_null=False),
                NumpyStringDomain(allow_null=False),
                True,
            ),
            (
                NumpyStringDomain(allow_null=False),
                ListDomain(NumpyIntegerDomain()),
                False,
            ),
        ],
    )
    def test_eq(self, domain: Domain, other_domain: Domain, expected: bool):
        """__eq__ works correctly.

        Args:
            domain: The domain to test.
            other_domain: The domain to compare to.
            expected: The expected result of the comparison.
        """
        super().test_eq(domain, other_domain, expected)

    @pytest.mark.skip("No arguments to mutate")
    @pytest.mark.parametrize("domain_args, key, mutator", [])
    def test_mutable_inputs(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        key: str,
        mutator: Callable[[Any], Any],
    ):
        """The mutable inputs to the domain are copied.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        super().test_mutable_inputs(domain_type, domain_args, key, mutator)

    @pytest.mark.parametrize(
        "domain, expected_properties",
        [
            (NumpyStringDomain(), {"allow_null": False, "carrier_type": object}),
            (
                NumpyStringDomain(allow_null=True),
                {"allow_null": True, "carrier_type": object},
            ),
        ],
    )
    def test_properties(self, domain: Domain, expected_properties: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            domain: The constructed domain to be tested.
            expected_properties: A dictionary containing all the property:value pairs
                domain is expected to have.
        """
        super().test_properties(domain, expected_properties)

    @pytest.mark.parametrize(
        "domain", [NumpyStringDomain(), NumpyStringDomain(allow_null=True)]
    )
    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        super().test_property_immutability(domain)

    @pytest.mark.parametrize(
        "domain, candidate, expectation, exception_properties",
        [
            (NumpyStringDomain(allow_null=True), None, does_not_raise(), None),
            (
                NumpyStringDomain(allow_null=False),
                None,
                pytest.raises(OutOfDomainError, match="Value is null."),
                {"domain": NumpyStringDomain(allow_null=False), "value": None},
            ),
            (NumpyStringDomain(allow_null=False), "ABC", does_not_raise(), None),
            (NumpyStringDomain(allow_null=True), "ABC", does_not_raise(), None),
            (NumpyStringDomain(allow_null=False), 123, does_not_raise(), None),
            (NumpyStringDomain(allow_null=False), np.int64(4), does_not_raise(), None),
        ],
    )
    def test_validate(
        self,
        domain: Domain,
        candidate: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Validate works correctly.

        Args:
            domain: The domain to test.
            candidate: The value to validate using domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        super().test_validate(domain, candidate, expectation, exception_properties)

    @pytest.mark.parametrize(
        "dtype, expected, expectation",
        [
            # pylint: disable=line-too-long
            (np.dtype(np.object0), NumpyStringDomain(), does_not_raise()),  # type: ignore[attr-defined]
            # pylint: enable=line-too-long
            (
                np.dtype([("f1", np.int64)]),
                None,
                pytest.raises(KeyError, match=f"{np.dtype([('f1', np.int64)])}"),
            ),
        ],
    )
    def test_from_np_type(
        self,
        domain_type: Type[NumpyStringDomain],
        dtype: np.dtype,
        expected: NumpyStringDomain,
        expectation: ContextManager[None],
    ):
        """from_np_type works correctly.

        Args:
            domain_type: The type of domain to be constructed.
            dtype: The dtype to test.
            expected: The expected domain to be constructed.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert domain_type.from_np_type(dtype) == expected
