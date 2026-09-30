"""Floating-point safe utility functions for per-record diffential privacy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024-2025, and the Tumult Core Contributors 2025-present

from flint import arb, ctx

from tmlt.core.random.continuous_gaussian import gaussian_inverse_cdf
from tmlt.core.random.inverse_cdf import construct_inverse_sampler


def fourth_root_transformation_mechanism(
    x: float, offset: float, sigma: float
) -> float:
    """Fourth root transformation mechanism."""
    x_arb = arb(x)
    sigma_arb = arb(sigma)
    offset_arb = arb(offset)

    def inverse_cdf(p: arb, prec: int) -> arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        with ctx.workprec(prec):
            u_arb = (x_arb + offset_arb) ** 0.25
            gaussian_sample = gaussian_inverse_cdf(
                u=u_arb, sigma_squared=sigma_arb**2, p=p, prec=prec
            )
            return gaussian_sample**4 - offset_arb

    return construct_inverse_sampler(inverse_cdf=inverse_cdf)()


def square_root_transformation_mechanism(
    x: float, offset: float, sigma: float
) -> float:
    """Square root transformation mechanism."""
    x_arb = arb(x)
    offset_arb = arb(offset)
    sigma_arb = arb(sigma)

    def inverse_cdf(p: arb, prec: int) -> arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        with ctx.workprec(prec):
            u_arb = (x_arb + offset_arb).sqrt()
            gaussian_sample = gaussian_inverse_cdf(
                u=u_arb, sigma_squared=sigma_arb**2, p=p, prec=prec
            )
            return gaussian_sample**2 - offset_arb

    return construct_inverse_sampler(inverse_cdf=inverse_cdf)()


def log_transformation_mechanism(x: float, offset: float, sigma: float) -> float:
    """Log transformation mechanism."""
    x_arb = arb(x)
    offset_arb = arb(offset)
    sigma_arb = arb(sigma)

    def inverse_cdf(p: arb, prec: int) -> arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        with ctx.workprec(prec):
            u_arb = (x_arb + offset_arb).log()
            gaussian_sample = gaussian_inverse_cdf(
                u=u_arb, sigma_squared=sigma_arb**2, p=p, prec=prec
            )
            return gaussian_sample.exp() - offset_arb

    return construct_inverse_sampler(inverse_cdf=inverse_cdf)()


def square_root_gaussian_inverse_cdf(x: arb, sigma: arb, prec: int) -> arb:
    r"""Inverse CDF for a special case of the generalized Gaussian distribution.

    In particular, this function returns the inverse CDF of the generalized Gaussian
    distribution when the shape parameter is ``1/2``:

    .. math::

        \begin{equation}
            \text{CDF}^{-1}(x) =
            \begin{cases}
                0 &  x = \frac{1}{2} \\
                \sigma\left[-W\left(\frac{2x-2}{e}\right)-1\right]^2 &  x > \frac{1}{2} \\
                -\sigma\left[-W\left(\frac{-2x}{e}\right)-1\right]^2  & x < \frac{1}{2}
            \end{cases}
        \end{equation}

    """  # noqa: E501
    if x == 0.5:
        return arb(0)

    with ctx.workprec(prec):
        e_arb = arb(1).exp()

        if x > 0.5:
            lambertw_arg = (2 * x - 2) / e_arb
            lambertw_branch = 0 if lambertw_arg >= 0 else -1
            lambert_term = lambertw_arg.lambertw(branch=lambertw_branch)
            return sigma * (lambert_term + 1) ** 2

        if x < 0.5:
            lambertw_arg = -2 * x / e_arb
            lambertw_branch = 0 if lambertw_arg >= 0 else -1
            lambert_term = lambertw_arg.lambertw(branch=lambertw_branch)
            return sigma.neg() * (lambert_term + 1) ** 2

        # NOTE: It is possible that none of the above conditions are true.
        # In this case, we return the interval (-inf, inf). The inverse CDF
        # sampler should re-try with more precision.
        return arb(mid=0, rad=float("inf"))


def square_root_gaussian_mechanism(sigma: float) -> float:
    """Samples a float from the generalized Gaussian distribution."""
    sigma_arb = arb(sigma)
    return construct_inverse_sampler(
        inverse_cdf=lambda p, prec: square_root_gaussian_inverse_cdf(p, sigma_arb, prec)
    )()


def _phi(x: arb, prec: int) -> arb:
    """CDF for the unit Gaussian distribution N(0, 1)."""
    with ctx.workprec(prec):
        erf_arg = x / arb(2).sqrt()
        return 0.5 * (1 + erf_arg.erf())


def _phi_inv(p: arb, prec: int) -> arb:
    """Inverse CDF for the unit Gaussian distribution N(0, 1)."""
    with ctx.workprec(prec):
        return arb(2).sqrt() * (2 * p - 1).erfinv()


def exponential_polylogarithmic_inverse_cdf(
    x: arb, d: arb, a: arb, sigma: arb, prec: int
) -> arb:
    r"""Inverse CDF for the exponential polylogarithmic distribution.

    In particular, this function computes the inverse CDF as defined below:

    .. math::

        y =
            \begin{cases}
                - \sigma \exp \left[\left([2d]^{-1/2}\Phi^{-1}\left[\left(\left[1-\Phi\left(\frac{\ln(a)-(2d)^{-1}}{(2d)^{-1/2}}\right)\right][1- 2x] \right) + \Phi\left(\frac{\ln(a)-(2d)^{-1}}{(2d)^{-1/2}}\right) \right]\right) + (2d)^{-1} \right] + \sigma a& x <\frac{1}{2}
                \\
                \sigma \exp \left[\left([2d]^{-1/2}\Phi^{-1}\left[\left(\left[1-\Phi\left(\frac{\ln(a)-(2d)^{-1}}{(2d)^{-1/2}}\right)\right][2x - 1]\right) + \Phi\left(\frac{\ln(a)-(2d)^{-1}}{(2d)^{-1/2}}\right) \right]\right) + (2d)^{-1}\right] - \sigma a& x >\frac{1}{2}
                \\
                0 & x = \frac{1}{2}
            \end{cases}

    """  # noqa: E501
    if x == 0.5:
        return arb(0)

    with ctx.workprec(prec):
        minus_sigma = sigma.neg()
        two_d = 2 * d

        one_minux_2_x = (2 * x - 1).neg()

        phi_arg = (a.log() - 1 / two_d) / (1 / two_d.sqrt())
        phi_term = _phi(phi_arg, prec=prec)

        if x < 0.5:
            return (
                minus_sigma
                * (
                    1
                    / two_d.sqrt()
                    * _phi_inv((1 - phi_term) * one_minux_2_x + phi_term, prec=prec)
                    + 1 / two_d
                ).exp()
                + sigma * a
            )

        if x > 0.5:
            return (
                sigma
                * (
                    1
                    / two_d.sqrt()
                    * _phi_inv((1 - phi_term) * (2 * x - 1) + phi_term, prec=prec)
                    + 1 / two_d
                ).exp()
                - sigma * a
            )
        # NOTE: It is possible that none of the above conditions are true.
        # In this case, we return the interval (-inf, inf). The inverse CDF
        # sampler should re-try with more precision.
        return arb(mid=0, rad=float("inf"))


def exponential_polylogarithmic_mechanism(
    d: float, a: float, sigma: float, step_size: int = 63
) -> float:
    """Samples a float from the exponential polylogarithmic distribution."""
    d_arb = arb(d)
    a_arb = arb(a)
    sigma_arb = arb(sigma)
    return construct_inverse_sampler(
        inverse_cdf=lambda p, prec: exponential_polylogarithmic_inverse_cdf(
            p, d_arb, a_arb, sigma_arb, prec
        ),
        step_size=step_size,
    )()
