"""Unit tests for :mod:`~tmlt.core.domains.pandas_domains`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from contextlib import nullcontext as does_not_raise
from itertools import combinations_with_replacement
from test.unit.domains.abstract import DomainTests
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import numpy as np
import pandas as pd
import pytest
from typeguard import TypeCheckError

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.utils.misc import get_fullname

_pandas_series_domains = {
    "int32_series_domain": PandasSeriesDomain(
        element_domain=NumpyIntegerDomain(size=32)
    ),
    "int64_series_domain": PandasSeriesDomain(
        element_domain=NumpyIntegerDomain(size=64)
    ),
    "float32_inf_series_domain": PandasSeriesDomain(
        element_domain=NumpyFloatDomain(size=32, allow_inf=True)
    ),
}


class TestPandasSeriesDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.pandas_domains.PandasSeriesDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return PandasSeriesDomain

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            ({"element_domain": NumpyIntegerDomain()}, does_not_raise(), None),
            (
                {"element_domain": ListDomain(NumpyFloatDomain())},
                pytest.raises(TypeCheckError),
                None,
            ),
            ({}, pytest.raises(TypeError), None),
            ({"element_domain": "not a domain"}, pytest.raises(TypeCheckError), None),
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
                _pandas_series_domains[base_key],
                _pandas_series_domains[other_key],
                base_key == other_key,
            )
            for (base_key, other_key) in combinations_with_replacement(
                _pandas_series_domains.keys(), 2
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
                PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                {
                    "element_domain": NumpyIntegerDomain(size=32),
                    "carrier_type": pd.Series,
                },
            ),
            (
                PandasSeriesDomain(NumpyFloatDomain(size=32, allow_inf=True)),
                {
                    "element_domain": NumpyFloatDomain(size=32, allow_inf=True),
                    "carrier_type": pd.Series,
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
            PandasSeriesDomain(NumpyIntegerDomain(size=32)),
            PandasSeriesDomain(NumpyIntegerDomain(size=64)),
            PandasSeriesDomain(NumpyFloatDomain(size=32, allow_inf=True)),
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
            # Success cases
            (
                PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                pd.Series([1, 2], dtype=np.dtype("int32")),
                does_not_raise(),
                None,
            ),
            (
                PandasSeriesDomain(NumpyIntegerDomain()),
                pd.Series([1, 2], dtype=np.dtype("int64")),
                does_not_raise(),
                None,
            ),
            (
                PandasSeriesDomain(NumpyFloatDomain(size=32, allow_inf=True)),
                pd.Series([1, 2], dtype=np.dtype("float32")),
                does_not_raise(),
                None,
            ),
            # Failure cases
            (
                PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                pd.Series([1, 2], dtype=np.dtype("int64")),
                pytest.raises(OutOfDomainError),
                {
                    "domain": PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                    "value": pd.Series([1, 2], dtype=np.dtype("int64")),
                },
            ),
            (
                PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                "not a series",
                pytest.raises(OutOfDomainError),
                {
                    "domain": PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                    "value": "not a series",
                },
            ),
            (
                PandasSeriesDomain(NumpyIntegerDomain()),
                pd.Series([1, 2], dtype=np.dtype("int32")),
                pytest.raises(OutOfDomainError),
                {
                    "domain": PandasSeriesDomain(NumpyIntegerDomain()),
                    "value": pd.Series([1, 2], dtype=np.dtype("int32")),
                },
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
        with expectation as exception:
            domain.validate(candidate)
        if exception_properties is None or len(exception_properties) == 0:
            return
        # Help out mypy
        assert isinstance(exception, pytest.ExceptionInfo)
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception.value, prop), f"Expected prop was missing: {prop}"
            actual_value = getattr(exception.value, prop)
            if isinstance(actual_value, pd.Series):
                assert actual_value.equals(
                    expected_value
                ), f"Expected {prop} to be {expected_value}, got {actual_value}"
            else:
                assert (
                    actual_value == expected_value
                ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @pytest.mark.parametrize(
        "dtype, expected, expectation",
        # pylint: disable=line-too-long
        [
            (
                np.dtype(dtype),
                PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                does_not_raise(),
            )
            for dtype in [np.int8, np.int16, np.int32, np.bool8]  # type: ignore[attr-defined]
        ]
        # pylint: enable=line-too-long
        + [
            (
                np.dtype(np.float32),
                PandasSeriesDomain(NumpyFloatDomain(size=32)),
                does_not_raise(),
            ),
            (
                np.dtype(np.float64),
                PandasSeriesDomain(NumpyFloatDomain(size=64)),
                does_not_raise(),
            ),
            (
                np.dtype(np.object0),  # type: ignore[attr-defined]
                PandasSeriesDomain(NumpyStringDomain()),
                does_not_raise(),
            ),
            (
                np.dtype(np.int64),
                PandasSeriesDomain(NumpyIntegerDomain(size=64)),
                does_not_raise(),
            ),
            # pylint: disable=line-too-long
            (np.dtype([("f1", np.int16)]), None, pytest.raises(KeyError)),  # type: ignore
            # pylint: enable=line-too-long
        ],
    )
    def test_from_numpy_type(
        self,
        domain_type: Type[PandasSeriesDomain],
        dtype: np.dtype,
        expected: PandasSeriesDomain,
        expectation: ContextManager[None],
    ):
        """The domain can be constructed from numpy types.

        Args:
            domain_type: The type of domain to be constructed.
            dtype: The numpy type to test.
            expected: The expected result of the conversion.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert domain_type.from_numpy_type(dtype) == expected


class TestPandasDataFrameDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.pandas_domains.PandasDataFrameDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return PandasDataFrameDomain

    @pytest.fixture(scope="class")
    def domain(self) -> PandasDataFrameDomain:
        """Get a base PandasDataFrameDomain."""
        return PandasDataFrameDomain(
            schema={
                "A": PandasSeriesDomain(NumpyIntegerDomain()),
                "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                "C": PandasSeriesDomain(NumpyStringDomain()),
            }
        )

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            ({"schema": invalid_schema}, pytest.raises(TypeCheckError), None)
            for invalid_schema in [
                {"A": "not a domain"},
                {"A": NumpyIntegerDomain(), "B": "not a domain"},
                {"A": 123},
                {"A": NumpyIntegerDomain},
                {23: PandasSeriesDomain(NumpyStringDomain())},
            ]
        ]
        + [
            ({"schema": valid_schema}, does_not_raise(), None)
            for valid_schema in [
                {"A": PandasSeriesDomain(NumpyIntegerDomain())},
                {
                    "A": PandasSeriesDomain(NumpyIntegerDomain()),
                    "B": PandasSeriesDomain(NumpyFloatDomain()),
                },
                {},
            ]
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
        "other_domain, expected",
        [
            (
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain()),
                        "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                        "C": PandasSeriesDomain(NumpyStringDomain()),
                    }
                ),
                True,
            ),
            # wrong column type
            (
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                        "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                        "C": PandasSeriesDomain(NumpyStringDomain()),
                    }
                ),
                False,
            ),
            # columns out of order
            (
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain()),
                        "C": PandasSeriesDomain(NumpyStringDomain()),
                        "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                    }
                ),
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

    @pytest.mark.parametrize(
        "domain_args, key, mutator",
        [
            (
                {"schema": {"A": PandasSeriesDomain(NumpyIntegerDomain())}},
                "schema",
                lambda x: x.update({"A": PandasSeriesDomain(NumpyFloatDomain())}),
            )
        ],
    )
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
                PandasDataFrameDomain(schema=schema),
                {"schema": schema, "carrier_type": pd.DataFrame},
            )
            for schema in [
                {"A": PandasSeriesDomain(NumpyIntegerDomain())},
                {
                    "A": PandasSeriesDomain(NumpyIntegerDomain()),
                    "B": PandasSeriesDomain(NumpyFloatDomain()),
                },
                {
                    "A": PandasSeriesDomain(NumpyIntegerDomain()),
                    "B": PandasSeriesDomain(NumpyFloatDomain()),
                    "C": PandasSeriesDomain(NumpyStringDomain()),
                },
            ]
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

    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """

    @pytest.mark.parametrize(
        "candidate, expectation, exception_properties",
        [
            (
                pd.DataFrame({"A": [1, 2], "B": [1.0, 2.0], "C": ["1", "2"]}),
                does_not_raise(),
                None,
            ),
            # wrong column type
            (
                pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": ["1", "2"]}),
                pytest.raises(
                    OutOfDomainError,
                    match="Found invalid value in column 'B': Found invalid value in "
                    f"Series: Value must be {get_fullname(np.float64)}, instead "
                    f"it is {get_fullname(np.int64)}.",
                ),
                {
                    "domain": PandasDataFrameDomain(
                        schema={
                            "A": PandasSeriesDomain(NumpyIntegerDomain()),
                            "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                            "C": PandasSeriesDomain(NumpyStringDomain()),
                        }
                    ),
                    "value": pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": ["1", "2"]}),
                },
            ),
            # missing column
            (
                pd.DataFrame({"A": [1, 2], "B": [1.0, 2.0]}),
                pytest.raises(
                    OutOfDomainError,
                    match="Columns are not as expected. DataFrame and Domain "
                    r"must contain the same columns in the same order.\nDataFrame "
                    r"columns: \['A', 'B'\]\n"
                    r"Domain columns: \['A', 'B', 'C'\]",
                ),
                {
                    "domain": PandasDataFrameDomain(
                        schema={
                            "A": PandasSeriesDomain(NumpyIntegerDomain()),
                            "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                            "C": PandasSeriesDomain(NumpyStringDomain()),
                        }
                    ),
                    "value": pd.DataFrame({"A": [1, 2], "B": [1.0, 2.0]}),
                },
            ),
            # columns out of order
            (
                pd.DataFrame({"A": [1, 2], "C": ["1", "2"], "B": [1.0, 2.0]}),
                pytest.raises(
                    OutOfDomainError,
                    match="Columns are not as expected. DataFrame and Domain "
                    r"must contain the same columns in the same order.\nDataFrame "
                    r"columns: \['A', 'C', 'B'\]\nDomain columns: \['A', 'B', 'C'\]",
                ),
                {
                    "domain": PandasDataFrameDomain(
                        schema={
                            "A": PandasSeriesDomain(NumpyIntegerDomain()),
                            "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                            "C": PandasSeriesDomain(NumpyStringDomain()),
                        }
                    ),
                    "value": pd.DataFrame(
                        {"A": [1, 2], "C": ["1", "2"], "B": [1.0, 2.0]}
                    ),
                },
            ),
            # wrong type
            (
                pd.Series([1, 2, 3]),
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(pd.DataFrame)}, instead it is "
                    f"{get_fullname(pd.Series)}.",
                ),
                {
                    "domain": PandasDataFrameDomain(
                        schema={
                            "A": PandasSeriesDomain(NumpyIntegerDomain()),
                            "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                            "C": PandasSeriesDomain(NumpyStringDomain()),
                        }
                    ),
                    "value": pd.Series([1, 2, 3]),
                },
            ),
            # duplicated columns
            (
                pd.DataFrame(
                    [["A", "A", "B", 1.1], ["V", "V", "E", 1.2], ["A", "A", "V", 1.3]],
                    columns=["A", "A", "B", "C"],
                ),
                pytest.raises(
                    OutOfDomainError, match=r"Some columns are duplicated, \['A'\]"
                ),
                {
                    "domain": PandasDataFrameDomain(
                        schema={
                            "A": PandasSeriesDomain(NumpyIntegerDomain()),
                            "B": PandasSeriesDomain(NumpyFloatDomain(allow_inf=True)),
                            "C": PandasSeriesDomain(NumpyStringDomain()),
                        }
                    ),
                    "value": pd.DataFrame(
                        [
                            ["A", "A", "B", 1.1],
                            ["V", "V", "E", 1.2],
                            ["A", "A", "V", 1.3],
                        ],
                        columns=["A", "A", "B", "C"],
                    ),
                },
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
        with expectation as exception:
            domain.validate(candidate)
        if exception_properties is None or len(exception_properties) == 0:
            return
        # Help out mypy
        assert isinstance(exception, pytest.ExceptionInfo)
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception.value, prop), f"Expected prop was missing: {prop}"
            actual_value = getattr(exception.value, prop)
            if isinstance(actual_value, (pd.Series, pd.DataFrame)):
                assert actual_value.equals(
                    expected_value
                ), f"Expected {prop} to be {expected_value}, got {actual_value}"
            else:
                assert (
                    actual_value == expected_value
                ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @pytest.mark.parametrize(
        "dtypes, expected, expectation",
        [
            # Success cases
            (
                {"A": np.dtype("int32"), "B": np.dtype("float32")},
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                        "B": PandasSeriesDomain(NumpyFloatDomain(size=32)),
                    }
                ),
                does_not_raise(),
            ),
            (
                {"A": np.dtype("int64"), "B": np.dtype("bool")},
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain(size=64)),
                        "B": PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                    }
                ),
                does_not_raise(),
            ),
        ]
        + [
            # Failure cases
            (
                {"A": np.dtype("int32"), "B": np.dtype([("f1", np.int16)])},
                None,
                pytest.raises(KeyError),
            )
        ],
    )
    def test_from_numpy_types(
        self,
        dtypes: Dict[str, np.dtype],
        expected: PandasDataFrameDomain,
        expectation: ContextManager[None],
    ):
        """from_numpy_types constructs the correct domain.

        Args:
            dtypes: The dtypes to construct the domain from.
            expected: The expected domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert PandasDataFrameDomain.from_numpy_types(dtypes) == expected
