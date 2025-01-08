"""Probability functions for distributions commonly used in differential privacy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from functools import lru_cache
from typing import Union, overload

import numpy as np
import sympy as sp
from scipy.stats import norm

from tmlt.core.utils.arb import (
    Arb,
    arb_add,
    arb_const_pi,
    arb_div,
    arb_erfc,
    arb_exp,
    arb_max,
    arb_min,
    arb_mul,
    arb_product,
    arb_sqrt,
    arb_sub,
    arb_sum,
    arb_union,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


@overload
def double_sided_geometric_pmf(k: int, alpha: float) -> float:
    ...


@overload
def double_sided_geometric_pmf(k: np.ndarray, alpha: float) -> np.ndarray:
    ...


@overload
def double_sided_geometric_pmf(k: int, alpha: np.ndarray) -> np.ndarray:
    ...


def double_sided_geometric_pmf(
    k: Union[int, np.ndarray], alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""Returns the pmf for a double-sided geometric distribution at k.

    For :math:`k \in \mathbb{Z}`

    .. math::

        f(k)=
        \frac
            {e^{1 / \alpha} - 1}
            {e^{1 / \alpha} + 1}
        \cdot
        e^{\frac{-\mid k \mid}{\alpha}}

    A double sided geometric distribution is the difference between two geometric
    distributions (It can be sampled by sampling two values from a geometric
    distribution, and taking their difference).

    See section 4.1 in :cite:`BalcerV18` or scipy.stats.geom for more information (Note
    that the parameter :math:`p` used in scipy.stats.geom is related to :math:`\alpha`
    through :math:`p = 1 - e^{-1 / \alpha}`).

    Args:
        k: The values to calculate the pmf for.
        alpha: The scale of the geometric distribution.
    """
    return (
        (np.exp(1 / alpha) - 1) / (np.exp(1 / alpha) + 1) * np.exp(-np.abs(k) / alpha)
    )


@overload
def double_sided_geometric_cmf(k: int, alpha: float) -> float:
    ...


@overload
def double_sided_geometric_cmf(k: np.ndarray, alpha: float) -> np.ndarray:
    ...


def double_sided_geometric_cmf(
    k: Union[int, np.ndarray], alpha: float
) -> Union[float, np.ndarray]:
    r"""Returns the cmf for a double-sided geometric distribution at k.

    For :math:`k \in \mathbb{Z}`

    .. math::

        F(k) = \begin{cases}
        \frac{e^{1 / \alpha}}{e^{1 / \alpha} + 1} \cdot e^\frac{k}{\alpha} &
        \text{if k} \le 0 \\
        1 - \frac{1}{e^{1 / \alpha} + 1}\cdot e^{\frac{-k}{\alpha}} &
        \text{otherwise} \\
        \end{cases}

    See :func:`double_sided_geometric_pmf` for more information.

    Args:
        k: The values to calculate the pmf for.
        alpha: The scale of the geometric distribution.
    """
    if isinstance(k, int):
        return double_sided_geometric_cmf(np.array([k]), alpha)[0]
    return np.where(
        k <= 0,
        np.exp(1 / alpha) / (np.exp(1 / alpha) + 1) * np.exp(k / alpha),
        1 - np.exp(-k / alpha) / (np.exp(1 / alpha) + 1),
    )


def double_sided_geometric_cmf_exact(
    k: ExactNumberInput, alpha: ExactNumberInput
) -> ExactNumber:
    """Returns exact value of the cmf for a double-sided geometric distribution at k.

    See :func:`double_sided_geometric_cmf` for more information.

    Args:
        k: The values to calculate the pmf for.
        alpha: The scale of the geometric distribution.
    """
    k_expr = ExactNumber(k).expr
    alpha_expr = ExactNumber(alpha).expr
    if k_expr <= 0:
        p = (
            sp.exp(1 / alpha_expr)
            / (sp.exp(1 / alpha_expr) + 1)
            * sp.exp(k_expr / alpha_expr)
        )
    else:
        p = 1 - 1 / (sp.exp(k_expr / alpha_expr) * (sp.exp(1 / alpha_expr) + 1))
    return ExactNumber(p)


@overload
def double_sided_geometric_inverse_cmf(p: float, alpha: float) -> int:
    ...


@overload
def double_sided_geometric_inverse_cmf(p: np.ndarray, alpha: float) -> np.ndarray:
    ...


def double_sided_geometric_inverse_cmf(
    p: Union[float, np.ndarray], alpha: float
) -> Union[int, np.ndarray]:
    r"""Returns the inverse cmf of a double-sided geometric distribution at p.

    In other words, it returns the smallest k s.t. CMF(k) >= p.

    For :math:`p \in [0, 1]`

    .. math::

        F(k) = \begin{cases}
        \alpha\ln(p\cdot\frac{e^{\frac{1}{\alpha}} + 1}{e^{\frac{1}{\alpha}}}) &
        \text{if p} \le 0.5 \\
        -\alpha\ln((e^{\frac{1}{\alpha}} + 1)(1 - p)) &
        \text{otherwise} \\
        \end{cases}

    See :func:`double_sided_geometric_pmf` for more information.
    """
    if isinstance(p, float):
        return double_sided_geometric_inverse_cmf(np.array([p]), alpha)[0]
    return np.where(
        p <= 0.5,
        np.ceil(alpha * np.log(p * (np.exp(1 / alpha) + 1) / np.exp(1 / alpha))),
        np.ceil(-alpha * np.log((np.exp(1 / alpha) + 1) * (1 - p))),
    )


def double_sided_geometric_inverse_cmf_exact(
    p: ExactNumberInput, alpha: ExactNumberInput
) -> ExactNumber:
    """Return exact value of inverse cmf of double-sided geometric distribution at p.

    See :func:`double_sided_geometric_inverse_cmf` for more information.

    Args:
        p: The values to calculate the inverse cmf for.
        alpha: The scale of the geometric distribution.
    """
    p_expr = ExactNumber(p).expr
    alpha_expr = ExactNumber(alpha).expr
    if p_expr < ExactNumber("0.5"):
        k = alpha_expr * sp.log(
            p_expr * (sp.exp(1 / alpha_expr) + 1) / sp.exp(1 / alpha_expr)
        )
    else:
        k = -alpha_expr * sp.log((sp.exp(1 / alpha_expr) + 1) * (1 - p_expr))
    return ExactNumber(sp.ceiling(k))


def _discrete_gaussian_unnormalized_pmf(k: int, sigma_squared: Arb, prec: int) -> Arb:
    r"""Returns the unnormalized pmf for a discrete gaussian distribution at k.

    :math:`e^\frac{-k^2}{2\sigma^2}`

    Notice that this is the numerator of the pmf for a discrete gaussian distribution.
    See :func:`~._discrete_gaussian_normalizing_constant` for more information.
    """
    return arb_exp(
        arb_div(
            Arb.from_int(-(k**2)), arb_mul(Arb.from_int(2), sigma_squared, prec), prec
        ),
        prec,
    )


@lru_cache(maxsize=128)
def _discrete_gaussian_unnormalized_mass_from_k_to_n(
    k: int, n: int, sigma_squared: Arb, prec: int
) -> Arb:
    """Returns the unnormalized mass for a discrete gaussian distribution from k to n.

    Includes both k and n.

    See :func:`~._discrete_gaussian_normalizing_constant` for more information.
    """
    return arb_sum(
        [
            _discrete_gaussian_unnormalized_pmf(i, sigma_squared, prec)
            for i in range(k, n + 1)
        ],
        prec,
    )


def _discrete_gaussian_unnormalized_mass_from_k_to_inf(
    k: int, sigma_squared: Arb, prec: int
) -> Arb:
    r"""Returns the unnormalized mass of a discrete gaussian distribution from k to inf.

    Includes k.

    Uses integral approximation.

    The integral of the unnormalized pmf from n to infinity is:
        :math:`\sqrt{\frac{\pi}{2}}\sigma\text{erfc}(\frac{n}{\sqrt{2}\sigma})`

    The lower bound is the integral from k to infinity.
    The upper bound is the integral from k-1 to infinity.

    See :func:`~._discrete_gaussian_normalizing_constant` for more information.
    """
    sigma = arb_sqrt(sigma_squared, prec)

    def integral(n: int) -> Arb:
        return arb_product(
            [
                arb_sqrt(arb_div(arb_const_pi(prec), Arb.from_int(2), prec), prec),
                sigma,
                arb_erfc(
                    arb_div(
                        Arb.from_int(n),
                        arb_mul(arb_sqrt(Arb.from_int(2), prec), sigma, prec),
                        prec,
                    ),
                    prec,
                ),
            ],
            prec,
        )

    lower = integral(k)
    upper = integral(k - 1)
    return arb_union(lower, upper, prec)


def _discrete_gaussian_unnormalized_mass_from_k_to_n_fast(
    k: int, n: int, sigma_squared: Arb, prec: int
) -> Arb:
    """Returns the unnormalized mass for a discrete gaussian distribution from k to n.

    Includes both k and n.

    Uses integral approximation. See
    :func:`_discrete_gaussian_unnormalized_mass_from_x_to_inf` for more information.
    """
    return arb_sub(
        _discrete_gaussian_unnormalized_mass_from_k_to_inf(k, sigma_squared, prec),
        _discrete_gaussian_unnormalized_mass_from_k_to_inf(n + 1, sigma_squared, prec),
        prec,
    )


def _discrete_gaussian_normalizing_constant(
    sigma_squared: Arb, n_terms: int, prec: int
) -> Arb:
    """Returns the normalizing factor for discrete gaussian noise.

    The normalizing factor is the sum of the unnormalized pmf for all integers.

    The terms for integers between -n_terms and n_terms are calculated exactly. The
    rest are approximated using the integral approximation.

    Notice this is the denominator of the pmf for a discrete gaussian distribution.
    See :func:`~.discrete_gaussian_pmf` for more information.
    """
    mass_at_0 = _discrete_gaussian_unnormalized_pmf(0, sigma_squared, prec)
    mass_from_1_to_n_terms = _discrete_gaussian_unnormalized_mass_from_k_to_n(
        1, n_terms, sigma_squared, prec
    )
    mass_from_n_terms_plus_1_to_inf = (
        _discrete_gaussian_unnormalized_mass_from_k_to_inf(
            n_terms + 1, sigma_squared, prec
        )
    )
    # all mass from -inf to inf
    return arb_sum(
        [
            mass_from_n_terms_plus_1_to_inf,  # -inf to -(n_terms + 1)
            mass_from_1_to_n_terms,  # -n_terms to -1
            mass_at_0,  # 0
            mass_from_1_to_n_terms,  # 1 to n_terms
            mass_from_n_terms_plus_1_to_inf,  # (n_terms + 1) to inf
        ],
        prec,
    )


def _discrete_gaussian_unnormalized_cmf(
    k: int, sigma_squared: Arb, n_terms: int, prec: int
) -> Arb:
    """Returns the unnormalized cmf for a discrete gaussian distribution at k.

    The unnormalized cmf is the sum of the unnormalized pmf for all integers from -inf
    to k.

    All terms for integers between -n_terms and n_terms are calculated exactly. The rest
    are approximated using the integral approximation.

    Only works for k >= 0.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    assert n_terms >= 0
    mass_at_0 = _discrete_gaussian_unnormalized_pmf(0, sigma_squared, prec)
    mass_from_1_to_n_terms = _discrete_gaussian_unnormalized_mass_from_k_to_n(
        1, n_terms, sigma_squared, prec
    )
    mass_from_n_terms_plus_1_to_inf = (
        _discrete_gaussian_unnormalized_mass_from_k_to_inf(
            n_terms + 1, sigma_squared, prec
        )
    )
    # multiple cases, handled from k=0 to inf
    # all cases have terms from -inf to 0
    result = arb_sum(
        [
            mass_from_n_terms_plus_1_to_inf,  # -inf to -(n_terms + 1)
            mass_from_1_to_n_terms,  # -n_terms to -1
            mass_at_0,  # 0
        ],
        prec,
    )
    if k == 0:
        return result
    elif k <= n_terms:  # k is in the range [1, n_terms]
        # add terms from 1 to k, by explicitly calculating them
        return arb_add(
            result,  # -inf to 0
            _discrete_gaussian_unnormalized_mass_from_k_to_n(  # 1 to k
                1, k, sigma_squared, prec
            ),
            prec,
        )
    else:
        assert k > n_terms  # k is in the range [n_terms + 1, inf)
        return arb_sum(
            [
                result,  # -inf to 0
                mass_from_1_to_n_terms,  # 1 to n_terms
                _discrete_gaussian_unnormalized_mass_from_k_to_n_fast(
                    n_terms + 1, k, sigma_squared, prec  # n_terms + 1 to k
                ),
            ],
            prec,
        )


def _discrete_gaussian_pmf(k: int, sigma_squared: Arb, n_terms: int, prec: int) -> Arb:
    """Returns the pmf for a discrete gaussian distribution at k.

    See :func:`~.discrete_gaussian_pmf` for more information.
    """
    return arb_div(
        _discrete_gaussian_unnormalized_pmf(k, sigma_squared, prec),
        _discrete_gaussian_normalizing_constant(sigma_squared, n_terms, prec),
        prec,
    )


def _discrete_gaussian_cmf(k: int, sigma_squared: Arb, n_terms: int, prec: int) -> Arb:
    """Returns the cmf for a discrete gaussian distribution at k.

    See :func:`~.discrete_gaussian_cmf` for more information.
    """
    if k < 0:  # eliminates half of the cases
        return arb_sub(
            Arb.from_int(1),
            _discrete_gaussian_cmf(-k - 1, sigma_squared, n_terms, prec),
            prec,
        )
    result = arb_div(
        _discrete_gaussian_unnormalized_cmf(k, sigma_squared, n_terms, prec),
        _discrete_gaussian_normalizing_constant(sigma_squared, n_terms, prec),
        prec,
    )
    # clamp to [0, 1]
    result = arb_min(result, Arb.from_int(1), prec)
    result = arb_max(result, Arb.from_int(0), prec)
    return result


@overload
def discrete_gaussian_pmf(k: int, sigma_squared: float) -> float:
    ...


@overload
def discrete_gaussian_pmf(k: np.ndarray, sigma_squared: float) -> np.ndarray:
    ...


def discrete_gaussian_pmf(
    k: Union[int, np.ndarray], sigma_squared: float
) -> Union[float, np.ndarray]:
    r"""Returns the pmf for a discrete gaussian distribution at k.

    For :math:`k \in \mathbb{Z}`

    .. math::
        :label: discrete_gaussian_pmf

        f(k) = \frac
        {e^{-k^2/2\sigma^2}}
        {
            \sum_{n\in \mathbb{Z}}
            e^{-n^2/2\sigma^2}
        }

    See :cite:`Canonne0S20` for more information. The formula above is based on
    Definition 1 in the paper. TEST PLEASE IGNORE

    .. note:

        The performance of this function degrades roughly linearly with the square root
        of ``sigma_squared``.

    Args:
        k: The value to evaluate the pmf at.
        sigma_squared: The variance of the discrete gaussian distribution.
    """
    if sigma_squared <= 0:
        raise ValueError("sigma_squared must be > 0")
    if isinstance(k, np.ndarray):
        results = np.empty_like(k, dtype=float)
        for i, k_i in enumerate(k):
            results[i] = discrete_gaussian_pmf(k_i, sigma_squared)
        return results
    # this is based on experiments for how many terms are needed. Technically you can
    # get a way with fewer for larger values of sigma (like 7 standard
    # deviations instead of 10 for sigma=10^4, but this works for all values of sigma).
    # see https://gitlab.com/tumult-labs/tumult/-/issues/2358#note_1418996578 for more
    # information.
    n_terms = int(np.sqrt(sigma_squared) * 10) + 1
    sigma_squared_arb = Arb.from_float(sigma_squared)
    prec = 100
    while True:
        try:
            return _discrete_gaussian_pmf(
                k, sigma_squared_arb, n_terms, prec
            ).to_float()
        except ValueError:
            prec *= 2
            n_terms *= 2


@overload
def discrete_gaussian_cmf(k: int, sigma_squared: float) -> float:
    ...


@overload
def discrete_gaussian_cmf(k: np.ndarray, sigma_squared: float) -> np.ndarray:
    ...


def discrete_gaussian_cmf(
    k: Union[int, np.ndarray], sigma_squared: float
) -> Union[float, np.ndarray]:
    """Returns the cmf for a discrete gaussian distribution at k.

    See :eq:`discrete_gaussian_pmf` for the probability mass function.

    .. note:

        The performance of this function degrades roughly linearly with the square root
        of ``sigma_squared``.

    Args:
        k: The value to evaluate the cmf at.
        sigma_squared: The variance of the discrete gaussian distribution.
    """
    if sigma_squared <= 0:
        raise ValueError("sigma_squared must be > 0")
    if isinstance(k, np.ndarray):
        return np.vectorize(discrete_gaussian_cmf)(k, sigma_squared)
    # this is based on experiments for how many terms are needed. Technically you can
    # get a way with fewer for larger values of sigma (like 7 standard
    # deviations instead of 10 for sigma=10^4, but this works for all values of sigma).
    # see https://gitlab.com/tumult-labs/tumult/-/issues/2358#note_1418996578 for more
    # information.
    n_terms = int(np.sqrt(sigma_squared) * 10) + 1
    prec = 100
    while True:
        try:
            return _discrete_gaussian_cmf(
                k, Arb.from_float(sigma_squared), n_terms, prec
            ).to_float()
        except ValueError:
            prec *= 2
            n_terms *= 2


@overload
def discrete_gaussian_inverse_cmf(p: float, sigma_squared: float) -> int:
    ...


@overload
def discrete_gaussian_inverse_cmf(p: np.ndarray, sigma_squared: float) -> np.ndarray:
    ...


@overload
def discrete_gaussian_inverse_cmf(p: Arb, sigma_squared: Arb) -> int:
    ...


def discrete_gaussian_inverse_cmf(
    p: Union[float, np.ndarray, Arb], sigma_squared: Union[float, Arb]
) -> Union[int, np.ndarray]:
    """Returns the inverse cmf for a discrete gaussian distribution at p.

    In other words, it returns the smallest k s.t. CMF(k) >= p.

    .. note:

        The performance of this function degrades roughly linearly with the square root
        of ``sigma_squared``.

    Args:
        p: The value to evaluate the inverse cmf at.
        sigma_squared: The variance of the discrete gaussian distribution.
    """
    if isinstance(p, np.ndarray):
        return np.vectorize(discrete_gaussian_inverse_cmf)(p, sigma_squared)

    if isinstance(p, float):
        p = Arb.from_float(p)
    elif isinstance(p, int):
        p = Arb.from_int(p)

    if not p.is_exact():
        raise ValueError(
            "p must be exact. If you want to use an approximate value, call this"
            " on p.lower() and p.upper() instead."
        )
    if not Arb.from_int(0) < p < Arb.from_int(1):
        raise ValueError("p must be strictly between 0 and 1")

    if isinstance(sigma_squared, float):
        sigma_squared = Arb.from_float(sigma_squared)
    elif isinstance(sigma_squared, int):
        sigma_squared = Arb.from_int(sigma_squared)

    if not sigma_squared.is_exact():
        raise ValueError(
            "sigma_squared must be exact. If you want to use an approximate value, call"
            " this on sigma_squared.lower() and sigma_squared.upper() instead."
        )
    if sigma_squared <= Arb.from_int(0):
        raise ValueError("sigma_squared must be > 0")

    # Calculating the cmf is expensive, so we start with low precision, and gradually
    # increase it as needed until we get the correct answer.

    # find initial value for lo, hi
    # Can get a very good initial guess by using the inverse of the normal distribution
    guess = int(norm.ppf(p.to_float(), scale=np.sqrt(sigma_squared.to_float())))
    distance = 0
    n_terms = 10
    prec = 30
    while True:
        lo = guess - distance - 1
        hi = guess + distance
        lo_p_arb = _discrete_gaussian_cmf(lo, sigma_squared, n_terms, prec)
        hi_p_arb = _discrete_gaussian_cmf(hi, sigma_squared, n_terms, prec)
        if lo_p_arb < p < hi_p_arb:
            # [lo + 1, hi] contains k
            break
        if lo_p_arb > p or hi_p_arb < p:
            if distance == 0:
                distance = 1
            else:
                distance *= 2
        else:
            # not enough precision to determine if k is in [lo + 1, hi]
            prec += 20
            n_terms *= 2

    # now do binary search
    while hi - lo > 1:
        mid = (hi + lo) // 2
        mid_cmf = _discrete_gaussian_cmf(mid, sigma_squared, n_terms, prec)
        if mid_cmf < p:
            # answer is in [mid + 1, hi]
            lo = mid
        elif mid_cmf > p:
            # answer is in [lo + 1, mid]
            hi = mid
        else:
            # not enough precision to determine if mid > p or mid < p
            n_terms *= 2
            prec += 20
    return hi
