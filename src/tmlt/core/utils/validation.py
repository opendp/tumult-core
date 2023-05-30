"""Utilities for checking the inputs to components."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from __future__ import annotations

import datetime
from typing import List, Mapping, Optional, Union

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


# You might want to put a type alias here for domains.
# Then you will discover that Sphinx AutoAPI doesn't support type aliases
# the way you want it to.
# Don't fall down this rabbit hole.
def validate_groupby_domains(
    groupby_domains: Mapping[
        str,
        Union[
            List[str],
            List[Optional[str]],
            List[int],
            List[Optional[int]],
            List[datetime.date],
            List[Optional[datetime.date]],
        ],
    ],
    input_domain: SparkDataFrameDomain,
):
    """Raises error if groupby domains are invalid.

    In particular, this passes only if:

        - each column has a non-empty domain AND
        - for each column, all values in its domain are valid values w.r.t the
          input domain AND
        - for each column, each value in its domain is distinct.
    """
    for column, domain in groupby_domains.items():
        if not domain:
            raise ValueError(f"Domain for '{column}' is empty!")
        col_desc = input_domain.schema[column]
        unique_keys = set()
        for key in domain:
            if key in unique_keys:
                raise ValueError(f"Domain for '{column}' contains duplicates.")
            unique_keys.add(key)
            if not col_desc.valid_py_value(key):
                raise ValueError(
                    f"Groupby key '{key}' is invalid for column's '{column}' "
                    f"domain {col_desc}."
                )


def validate_exact_number(
    value: ExactNumberInput,
    allow_nonintegral: bool = True,
    minimum: Optional[ExactNumberInput] = None,
    minimum_is_inclusive: bool = True,
    maximum: Optional[ExactNumberInput] = None,
    maximum_is_inclusive: bool = True,
) -> None:
    """Raises a :class:`ValueError` if `value` fails the specified conditions.

    Examples:
        Verify that a number is integral

        ..
            >>> from fractions import Fraction
            >>> import sympy as sp

        >>> validate_exact_number(
        ...     value=1,
        ...     allow_nonintegral=False,
        ... )
        >>> validate_exact_number(
        ...     value=Fraction(1, 2),
        ...     allow_nonintegral=False,
        ... )
        Traceback (most recent call last):
        ValueError: 1/2 is not an integer

        Verify that a number is between 0 and 1 (inclusive)

        >>> validate_exact_number(
        ...     value=1,
        ...     minimum=0,
        ...     maximum=1,
        ... )
        >>> validate_exact_number(
        ...     value=-1,
        ...     minimum=0,
        ...     maximum=1,
        ... )
        Traceback (most recent call last):
        ValueError: -1 is not greater than or equal to 0
        >>> validate_exact_number(
        ...     value=2,
        ...     minimum=0,
        ...     maximum=1,
        ... )
        Traceback (most recent call last):
        ValueError: 2 is not less than or equal to 1

        Verify that a number is a finite integer

        >>> validate_exact_number(
        ...     value=-123,
        ...     allow_nonintegral=False,
        ...     minimum=-float("inf"),
        ...     minimum_is_inclusive=False,
        ...     maximum=float("inf"),
        ...     maximum_is_inclusive=False,
        ... )
        >>> validate_exact_number(
        ...     value="0.5",
        ...     allow_nonintegral=False,
        ...     minimum=-float("inf"),
        ...     minimum_is_inclusive=False,
        ...     maximum=float("inf"),
        ...     maximum_is_inclusive=False,
        ... )
        Traceback (most recent call last):
        ValueError: 0.5 is not an integer
        >>> validate_exact_number(
        ...     value=sp.oo,
        ...     allow_nonintegral=False,
        ...     minimum=-float("inf"),
        ...     minimum_is_inclusive=False,
        ...     maximum=float("inf"),
        ...     maximum_is_inclusive=False,
        ... )
        Traceback (most recent call last):
        ValueError: oo is not strictly less than inf
        >>> validate_exact_number(
        ...     value=-sp.oo,
        ...     allow_nonintegral=False,
        ...     minimum=-float("inf"),
        ...     minimum_is_inclusive=False,
        ...     maximum=float("inf"),
        ...     maximum_is_inclusive=False,
        ... )
        Traceback (most recent call last):
        ValueError: -oo is not strictly greater than -inf

    Args:
        value: A :class:`sympy.Expr` to validate.
        allow_nonintegral: If False, raises an error if `value` is not integral,
            unless it is infinity.
        minimum: An optional lower bound.
        minimum_is_inclusive: If False, `value` is not allowed to be equal to
            `minimum`. Defaults to True.
        maximum: An optional upper bound.
        maximum_is_inclusive: If False, `value` being equal to `maximum` is not
            allowed. Defaults to True.
    """
    exact_value = ExactNumber(value)
    if not (allow_nonintegral or exact_value.is_integer or not exact_value.is_finite):
        raise ValueError(f"{value} is not an integer")
    if minimum is not None:
        exact_minimum = ExactNumber(minimum)
        if minimum_is_inclusive:
            if exact_value < exact_minimum:
                raise ValueError(f"{value} is not greater than or equal to {minimum}")
        else:
            if exact_value <= exact_minimum:
                raise ValueError(f"{value} is not strictly greater than {minimum}")
    if maximum is not None:
        exact_maximum = ExactNumber(maximum)
        if maximum_is_inclusive:
            if exact_value > exact_maximum:
                raise ValueError(f"{value} is not less than or equal to {maximum}")
        else:
            if exact_value >= exact_maximum:
                raise ValueError(f"{value} is not strictly less than {maximum}")
