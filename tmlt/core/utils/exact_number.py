"""Utilities for using exact representations of real numbers.

This module contains :class:`~.ExactNumber` and :data:`~.ExactNumberInput`.

:class:`~.ExactNumber`'s are used as inputs for components that privacy guarantees
depend on. They are used because they can be exactly represented and manipulated using
`SymPy <https://www.sympy.org/en/index.html>`_, which avoids errors such as privacy
violations due to numerical imprecision, as well as more mundane user errors from the
limitations of floating point arithmetic. See
:ref:`Handling Real Numbers <real-numbers>` for more information.

:data:`~.ExactNumberInput`'s are values which can be automatically converted into
:class:`~.ExactNumber`'s. They can be any of the following:

Any :class:`int` or :class:`fractions.Fraction`:
    >>> ExactNumber(5)
    5
    >>> ExactNumber(-1000000000000000)
    -1000000000000000
    >>> ExactNumber(Fraction(1, 10))
    1/10
    >>> ExactNumber(Fraction(127, 128))
    127/128

Any :class:`sympy.Expr` that meets all of the following criteria:
    No free symbols

    >>> x = sp.symbols("x")
    >>> x_plus_1 = x + 1
    >>> ExactNumber(x_plus_1)
    Traceback (most recent call last):
    ValueError: x + 1 contains free symbols
    >>> ExactNumber(x_plus_1.subs(x, 3))
    4

    No undefined functions

    >>> f = sp.core.function.Function('f')
    >>> ExactNumber(f(2) + 1)
    Traceback (most recent call last):
    ValueError: f(2) + 1 has an undefined function
    >>> f = sp.Lambda(x, x**2)
    >>> ExactNumber(f(2) + 1)
    5

    No floating point values

    >>> ExactNumber(sp.Float(3.14))
    Traceback (most recent call last):
    ValueError: 3.14000000000000 is represented using floating point precision
    >>> ExactNumber(sp.Float(3.14) * (sp.Integer(2) ** sp.pi))
    Traceback (most recent call last):
    ValueError: 3.14*2**pi is invalid: 3.14000000000000 is represented using floating point precision
    >>> ExactNumber(sp.Rational("3.14") * (sp.Float(3) ** sp.pi))
    Traceback (most recent call last):
    ValueError: 157*3.0**pi/50 is invalid: Base of 3.0**pi is invalid: 3.00000000000000 is represented using floating point precision

    Is a real number (or +/- infinity)

    >>> i = sp.sqrt(sp.Integer(-1))
    >>> ExactNumber(i)
    Traceback (most recent call last):
    ValueError: I has an imaginary component
    >>> ExactNumber(1 + 2*i)
    Traceback (most recent call last):
    ValueError: 1 + 2*I has an imaginary component
    >>> ExactNumber(i**2)
    -1
    >>> ExactNumber(sp.oo)
    oo
    >>> ExactNumber(-sp.oo)
    -oo
    >>> ExactNumber(1 + sp.oo*i)
    Traceback (most recent call last):
    ValueError: 1 + oo*I is invalid: oo*I is invalid: I has an imaginary component
    >>> ExactNumber(sp.oo + i)
    Traceback (most recent call last):
    ValueError: oo + I is invalid: I has an imaginary component

Any :class:`str` that can be:
    1. Exactly interpreted as a Rational number or
    2. Converted to a valid :class:`sympy.Expr` by
       :func:`sympy.parsing.sympy_parser.parse_expr`

    >>> ExactNumber("0.5")
    1/2
    >>> ExactNumber("0.123456789")
    123456789/1000000000
    >>> ExactNumber("2 + 7**2")
    51
    >>> ExactNumber("2 * pi**2")
    2*pi**2
    >>> ExactNumber("sqrt(5/3)")
    sqrt(15)/3
    >>> ExactNumber("pi + I")
    Traceback (most recent call last):
    ValueError: pi + I has an imaginary component
    >>> ExactNumber("x + 1")
    Traceback (most recent call last):
    ValueError: x + 1 contains free symbols

`float('inf')` and `-float('inf')` are allowed:
    >>> ExactNumber(float('inf'))
    oo
    >>> ExactNumber(-float('inf'))
    -oo
    >>> ExactNumber(3.5)
    Traceback (most recent call last):
    ValueError: Expected +/-float('inf'), not 3.5
    <BLANKLINE>
    Floating point values typically do not exactly represent the value they are intended to represent, and so are not automatically converted. See tmlt.core.utils.exact_number.from_float for more information.

Finally :class:`~.ExactNumber`'s are allowed:
    >>> ExactNumber(ExactNumber(3))
    3
    >>> ExactNumber(ExactNumber("0.5"))
    1/2

:class:`~.ExactNumber`'s support many common mathematical operations, and can be used in
combination with :data:`~.ExactNumberInput`'s.

Examples:
    >>> ExactNumber(1) + ExactNumber("0.5")
    3/2
    >>> ExactNumber(sp.Integer(7)) - 3
    4
    >>> ExactNumber(5) ** 2
    25
    >>> -2 ** ExactNumber(Fraction(1, 2))
    -sqrt(2)
    >>> 2 / ExactNumber(6)
    1/3
"""  # pylint: disable=line-too-long

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from fractions import Fraction
from typing import Any, Union

import sympy as sp
from typeguard import typechecked


@typechecked
def _verify_expr_is_an_exact_number(expr: sp.Expr) -> None:
    """Raises an error if `expr` is not an exact real number or +/- infinity."""
    if expr.free_symbols:
        raise ValueError(f"{expr} contains free symbols")
    # is_number means no free symbols, and no undefined functions
    if not expr.is_number:
        raise ValueError(f"{expr} has an undefined function")
    if expr.is_finite and not expr.is_real:
        raise ValueError(f"{expr} has an imaginary component")
    if expr in (sp.oo, -sp.oo):
        return
    if isinstance(expr, (sp.Integer, sp.Rational)):
        return
    if isinstance(expr, sp.NumberSymbol):
        return
    if isinstance(expr, (sp.Mul, sp.Add)):
        left_expr, right_expr = expr.as_two_terms()
        try:
            _verify_expr_is_an_exact_number(left_expr)
            _verify_expr_is_an_exact_number(right_expr)
        except ValueError as e:
            raise ValueError(f"{expr} is invalid: {e}")
        return
    if isinstance(expr, (sp.Pow, sp.exp)):
        try:
            _verify_expr_is_an_exact_number(expr.base)
        except ValueError as e:
            raise ValueError(f"Base of {expr} is invalid: {e}")
        try:
            _verify_expr_is_an_exact_number(expr.exp)
        except ValueError as e:
            raise ValueError(f"Exponent of {expr} is invalid: {e}")
        return
    if isinstance(expr, sp.log):
        if len(expr.args) != 1:
            raise ValueError(f"Logarithm {expr} has more than one term")
        try:
            _verify_expr_is_an_exact_number(expr.args[0])
        except ValueError as e:
            raise ValueError(f"Invalid Logarithm {expr}: {e}")
        return
    if isinstance(expr, sp.Float):
        raise ValueError(f"{expr} is represented using floating point precision")
    raise ValueError(f"Invalid SymPy expression: {expr}")


@typechecked
def _to_sympy(value: "ExactNumberInput") -> sp.Expr:
    """Returns a :class:`sympy.Expr` representing the input value.

    Raises:
        ValueError: If `value` cannot be converted to an :class:`sympy.Expr` or the
            resulting expression is not an exact real number or +/- infinity.
    """
    expr: sp.Expr
    if isinstance(value, ExactNumber):
        return value.expr
    elif isinstance(value, float):
        if value == float("inf"):
            expr = sp.oo
        elif value == -float("inf"):
            expr = -sp.oo
        else:
            raise ValueError(
                f"Expected +/-float('inf'), not {value}"
                "\n\nFloating point values typically do not exactly represent the "
                "value they are intended to represent, and so are not automatically "
                "converted. See tmlt.core.utils.exact_number.from_float for more "
                "information."
            )
    elif isinstance(value, int):
        expr = sp.Integer(value)
    elif isinstance(value, str):
        try:
            expr = sp.Rational(value)
        except (TypeError, ValueError):
            # evaluate=False keeps the structure of expr close to the string that was
            # passed in, which is nicer for the exceptions. If it passes all of the
            # checks, we can simplify before returning.
            expr = sp.parsing.sympy_parser.parse_expr(value, evaluate=False)
    elif isinstance(value, Fraction):
        expr = sp.Rational(value.numerator, value.denominator)
    else:
        assert isinstance(value, sp.Expr)
        expr = value

    _verify_expr_is_an_exact_number(expr)
    expr = sp.simplify(expr)
    return expr


class ExactNumber:
    """An exact representation of a real number or +/- infinity.

    See :mod:`~tmlt.core.utils.exact_number` for more information and examples.
    """

    def __init__(self, value: "ExactNumberInput"):
        """Constructor.

        Args:
            value: An :data:`~.ExactNumberInput` that represents a real number or +/-
                infinity.
        """
        self._expr = _to_sympy(value)

    @property
    def expr(self) -> sp.Expr:
        """Returns a :class:`sympy.Expr` representation."""
        return self._expr

    @property
    def is_integer(self) -> bool:
        """Returns whether self represents an integer."""
        return bool(self.expr.is_integer)

    @property
    def is_finite(self) -> bool:
        """Returns whether self represents a finite number."""
        return bool(self.expr.is_finite)

    def to_float(self, round_up: bool) -> float:
        """Returns self as a float.

        Args:
            round_up: If True, the returned value is greater than or equal to self.
                Otherwise, it is less than or equal to self.
        """
        if self.is_integer:
            return int(self.expr)
        evaluated_val = float(self.expr.evalf())
        nudge = 1e-15 if round_up else -1e-15

        def compare(x: float, y: sp.Expr) -> bool:
            return x < y if round_up else x > y

        while bool(compare(evaluated_val, self.expr)):
            evaluated_val += nudge
            nudge *= 10
        return evaluated_val

    @staticmethod
    @typechecked
    def from_float(value: float, round_up: bool) -> "ExactNumber":
        """Returns an :class:`~.ExactNumber` from a :class:`float`.

        .. warning::

            Floating point values do not have the same algebraic properties as real
            numbers (For example, operations such as addition and multiplication are not
            associative or distributive). It is strongly recommended to use exact
            representations where possible and to only use this method when an exact
            representation is no longer needed.

        Args:
            value: A :class:`float` to convert to an :class:`~.ExactNumber`.
            round_up: If True, returned value is greater than or equal to `value`.
                Otherwise, it is less than or equal to `value`.
        """
        if float(value).is_integer():
            return ExactNumber(int(value))
        if value in [float("inf"), -float("inf")]:
            return ExactNumber(value)
        expr = sp.Rational(*value.as_integer_ratio())
        nudge = sp.Pow(10, -15) if round_up else -sp.Pow(10, -15)

        def compare(x: sp.Expr, y: sp.Rational) -> bool:
            return x < y if round_up else x > y

        while compare(expr, value):
            expr += nudge
            nudge *= 10
        return ExactNumber(expr)

    def __abs__(self) -> "ExactNumber":
        """Returns absolute value of self."""
        return ExactNumber(sp.Abs(self.expr))

    def __neg__(self) -> "ExactNumber":
        """Returns the additive inverse of self."""
        return -1 * self.expr

    def __truediv__(self, other: "ExactNumberInput") -> "ExactNumber":
        """Returns quotient from dividing self by other."""
        other = ExactNumber(other)
        if other == 0:
            raise ZeroDivisionError("division by zero")
        return ExactNumber(self.expr / other.expr)

    def __rtruediv__(self, other: "ExactNumberInput") -> "ExactNumber":
        """Returns quotient from dividing other by self."""
        other = ExactNumber(other)
        if self == 0:
            raise ZeroDivisionError("division by zero")
        return ExactNumber(other.expr / self.expr)

    def __mul__(self, other: "ExactNumberInput") -> "ExactNumber":
        """Returns product of self and other."""
        other = ExactNumber(other)
        return ExactNumber(self.expr * other.expr)

    def __add__(self, other: "ExactNumberInput") -> "ExactNumber":
        """Returns sum of self and other."""
        other = ExactNumber(other)
        return ExactNumber(self.expr + other.expr)

    def __sub__(self, other: "ExactNumberInput") -> "ExactNumber":
        """Returns difference of self and other."""
        other = ExactNumber(other)
        return ExactNumber(self.expr - other.expr)

    def __rsub__(self, other: "ExactNumberInput") -> "ExactNumber":
        """Returns difference of other and self."""
        other = ExactNumber(other)
        return ExactNumber(other.expr - self.expr)

    def __pow__(self, other: "ExactNumberInput") -> "ExactNumber":
        """Returns power obtained by raising self to other."""
        other = ExactNumber(other)
        return ExactNumber(self.expr ** other.expr)

    def __rpow__(self, other: "ExactNumberInput") -> "ExactNumber":
        """Returns power obtained by raising other to self."""
        other = ExactNumber(other)
        return ExactNumber(other.expr ** self.expr)

    def __eq__(self, other: Any) -> bool:
        """Returns True if other and self represent the same value."""
        try:
            other = ExactNumber(other)
        except ValueError:
            return False
        return bool(sp.Eq(self.expr, other.expr))

    def __lt__(self, other: Any) -> bool:
        """Returns True if self is strictly less than other."""
        try:
            other = ExactNumber(other)
        except ValueError:
            return False
        return bool(self.expr < other.expr)

    def __le__(self, other: Any) -> bool:
        """Returns True if self is less than or equal to other."""
        try:
            other = ExactNumber(other)
        except ValueError:
            return False
        return bool(self.expr <= other.expr)

    def __gt__(self, other: Any) -> bool:
        """Returns True if self is strictly greater than other."""
        try:
            other = ExactNumber(other)
        except ValueError:
            return False
        return bool(self.expr > other.expr)

    def __ge__(self, other: Any) -> bool:
        """Returns True if self is greater than or equal to other."""
        try:
            other = ExactNumber(other)
        except ValueError:
            return False
        return bool(self.expr >= other.expr)

    def __repr__(self) -> str:
        """Returns a string representation."""
        return repr(self.expr)

    def __hash__(self) -> int:
        """Returns a hash."""
        return hash(self.expr)

    __radd__ = __add__

    __rmul__ = __mul__


ExactNumberInput = Union[ExactNumber, float, int, str, Fraction, sp.Expr]
"""A type alias for exact representations of real numbers.

See :mod:`.exact_number` for more information.
"""
