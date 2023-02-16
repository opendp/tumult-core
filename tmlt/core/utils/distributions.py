"""Probability functions for distributions commonly used in differential privacy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from functools import lru_cache
from typing import overload

import numpy as np
import sympy as sp

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
    k, alpha
):  # pylint: disable=missing-type-doc, missing-return-type-doc
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
    k, alpha
):  # pylint: disable=missing-type-doc, missing-return-type-doc
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
    p, alpha
):  # pylint: disable=missing-type-doc, missing-return-type-doc
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


@lru_cache(None)
def _discrete_gaussian_normalizing_constant(sigma_squared: float):
    """Returns the normalizing factor for :func:`discrete_gaussian_pmf`."""
    # This method for calculating the upper/lower bound to consider is used in
    # https://github.com/IBM/discrete-gaussian-differential-privacy/blob/master/testing-kolmogorov-discretegaussian.py
    # They say
    # "Compute the normalizing constant from -bounds to bounds instead of -inf to inf:
    # negligible error, as bounds is at least 50 standard deviations from 0,
    # and the tails decay exponentially."
    bound = max(10000, int(sigma_squared * 50))
    # +1 is the contribution from y = 0
    # The factor of 2 is because y/-y contribute the same for y != 0
    return 1 + 2 * np.sum(np.exp(-np.arange(1, bound) ** 2 / (2 * sigma_squared)))


@overload
def discrete_gaussian_pmf(k: int, sigma_squared: float) -> float:
    ...


@overload
def discrete_gaussian_pmf(k: np.ndarray, sigma_squared: float) -> np.ndarray:
    ...


def discrete_gaussian_pmf(k, sigma_squared):
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
    Definition 1.

    The implementation also referenced the paper's implementation, which can be found at
    https://github.com/IBM/discrete-gaussian-differential-privacy.
    """
    return np.exp(
        -(k ** 2) / (2 * sigma_squared)
    ) / _discrete_gaussian_normalizing_constant(sigma_squared)


@overload
def discrete_gaussian_cmf(k: int, sigma_squared: float) -> float:
    ...


@overload
def discrete_gaussian_cmf(k: np.ndarray, sigma_squared: float) -> np.ndarray:
    ...


def discrete_gaussian_cmf(k, sigma_squared):
    """Returns the cmf for a discrete gaussian distribution at k.

    See :eq:`discrete_gaussian_pmf` for the probability mass function.
    """
    if isinstance(k, int):
        return discrete_gaussian_cmf(np.array([k]), sigma_squared)[0]
    # _discrete_gaussian_normalizing_constant explains calculating the bound this way
    lower_bound = min(-10000, -int(sigma_squared * 50), np.min(k))
    unnormalized_cmf = np.cumsum(
        np.exp(
            -(np.abs(np.arange(lower_bound, np.max(k) + 1) ** 2)) / (2 * sigma_squared)
        )
    )
    return unnormalized_cmf[k - lower_bound] / _discrete_gaussian_normalizing_constant(
        sigma_squared
    )


@overload
def discrete_gaussian_inverse_cmf(p: float, sigma_squared: float) -> int:
    ...


@overload
def discrete_gaussian_inverse_cmf(p: np.ndarray, sigma_squared: float) -> np.ndarray:
    ...


def discrete_gaussian_inverse_cmf(p, sigma_squared):
    """Returns the inverse cmf for a discrete gaussian distribution at p.

    In other words, it returns the smallest k s.t. CMF(k) >= p.

    This is not terribly performant. Don't use it in places where it's
    going to get called a lot.
    """
    if isinstance(p, np.ndarray):
        return np.vectorize(discrete_gaussian_inverse_cmf)(p, sigma_squared)

    # The discrete gaussian distribution is symmetrical with a mean of 0, so CMF(p) =
    # -CMF(1.0 - p). We use this to always calculate the inverse CMF on a probability
    # <= 0.5, which reduces the number of cases we need to cover.
    adjusted_p = p if p <= 0.5 else 1.0 - p

    unnormalized_cmf = adjusted_p * _discrete_gaussian_normalizing_constant(
        sigma_squared
    )

    def unnormalized_pmf(k):
        return np.exp(-(k ** 2) / (2 * sigma_squared))

    # _discrete_gaussian_normalizing_constant explains calculating the bound this way
    lower_bound = min(-10000, -int(sigma_squared * 50))

    # if k < lower bound, cmf(k) returns pmf(k), so we can binary search for k.
    if unnormalized_pmf(lower_bound) > unnormalized_cmf:
        hi = lower_bound
        lo = lower_bound * 2
        while unnormalized_pmf(lo) > unnormalized_cmf:
            hi = lo
            lo *= 2
        while hi != lo:
            mid = hi / 2 + lo / 2
            if unnormalized_pmf(mid) == unnormalized_cmf:
                return mid if p <= 0.5 else -mid
            elif unnormalized_pmf(mid) < unnormalized_cmf:
                lo = mid
            else:
                hi = mid
        return hi if p <= 0.5 else -hi

    k = (
        np.searchsorted(
            np.cumsum(unnormalized_pmf(np.arange(lower_bound, 0))), unnormalized_cmf
        )
        + lower_bound
    )

    return k if p <= 0.5 else -k
