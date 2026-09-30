"""Utilities for working with arbs."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Core Contributors 2025-present

import math

from flint import arb, ctx


def to_only_float(n: arb, prec: int = 64) -> float:
    """Returns the only floating point number contained in ``n``.

    If more than one float lies in the interval represented by ``n``,
    this raises an error.
    """
    if not n.is_nan() and n.is_finite():
        with ctx.workprec(prec):
            l_man, l_exp = n.lower().man_exp()
            u_man, u_exp = n.upper().man_exp()
            lower_float = math.ldexp(int(l_man), int(l_exp))
            upper_float = math.ldexp(int(u_man), int(u_exp))
            if lower_float == upper_float:
                return lower_float
    if not n.is_finite() and n.is_exact():
        if n.mid() > 0:
            return float("inf")
        return -float("inf")
    raise ValueError("Arb contains more than one float.")
