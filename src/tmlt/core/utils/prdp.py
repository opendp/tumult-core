"""Floating-point safe utility functions for per-record diffential privacy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

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
    one_fourth = arb(1 / 4)
    two = arb(2)
    four = arb(4)

    def inverse_cdf(p: arb, prec: int) -> arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        with ctx.workprec(prec):
            u_arb = (x_arb + offset_arb) ** (one_fourth)
            sigma_squared_arb = sigma_arb ** (two)
            gaussian_sample = gaussian_inverse_cdf(
                u=u_arb, sigma_squared=sigma_squared_arb, p=p, prec=prec
            )
            return gaussian_sample ** (four) - offset_arb

    return construct_inverse_sampler(inverse_cdf=inverse_cdf)()


def square_root_transformation_mechanism(
    x: float, offset: float, sigma: float
) -> float:
    """Square root transformation mechanism."""
    x_arb = arb(x)
    offset_arb = arb(offset)
    sigma_arb = arb(sigma)
    two = arb(2)

    def inverse_cdf(p: arb, prec: int) -> arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        with ctx.workprec(prec):
            u_arb = (x_arb + offset_arb).sqrt()
            sigma_squared_arb = sigma_arb ** (two)
            gaussian_sample = gaussian_inverse_cdf(
                u=u_arb, sigma_squared=sigma_squared_arb, p=p, prec=prec
            )
            return gaussian_sample ** (two) - offset_arb

    return construct_inverse_sampler(inverse_cdf=inverse_cdf)()


def log_transformation_mechanism(x: float, offset: float, sigma: float) -> float:
    """Log transformation mechanism."""
    x_arb = arb(x)
    offset_arb = arb(offset)
    sigma_arb = arb(sigma)
    two = arb(2)

    def inverse_cdf(p: arb, prec: int) -> arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        with ctx.workprec(prec):
            u_arb = (x_arb + offset_arb).log()
            sigma_squared_arb = sigma_arb ** (two)
            gaussian_sample = gaussian_inverse_cdf(
                u=u_arb, sigma_squared=sigma_squared_arb, p=p, prec=prec
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
    if x == arb(0.5):
        return arb(0)

    zero = arb(0)
    half = arb(0.5)
    one = arb(1)
    two = arb(2)
    with ctx.workprec(prec):
        e_arb = one.exp()

        if x > half:
            lambertw_arg = ((arb(2) * x) - two) / e_arb
            lambertw_branch = 0 if lambertw_arg >= zero else -1
            lambert_term = lambertw_arg.lambertw(branch=lambertw_branch)
            return sigma * (lambert_term + one) ** (two)

        if x < half:
            lambertw_arg = (arb(2).neg() * x) / e_arb
            lambertw_branch = 0 if lambertw_arg >= zero else -1
            lambert_term = lambertw_arg.lambertw(branch=lambertw_branch)
            return sigma.neg() * (lambert_term + one) ** (two)

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
    half = arb(0.5)
    with ctx.workprec(prec):
        erf_arg = x / arb(2).sqrt()
        return half * (arb(1) + erf_arg.erf())


def _phi_inv(p: arb, prec: int) -> arb:
    """Inverse CDF for the unit Gaussian distribution N(0, 1)."""
    with ctx.workprec(prec):
        return arb(2).sqrt() * ((arb(2) * p) - arb(1)).erfinv()


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
    if x == arb(0.5):
        return arb(0)

    with ctx.workprec(prec):
        minus_sigma = sigma.neg()
        half = arb(0.5)
        one = arb(1)
        two_d = arb(2) * d

        two_x_minus_1 = (arb(2) * x) - one
        one_minux_2_x = two_x_minus_1.neg()

        log_a = a.log()
        sqrt_2d = two_d.sqrt()
        one_div_sqrt_2d = one / sqrt_2d
        one_div_2d = one / two_d

        sigma_times_a = sigma * a

        phi_arg = (log_a - one_div_2d) / one_div_sqrt_2d
        phi_term = _phi(phi_arg, prec=prec)
        one_minus_phi_term = one - phi_term

        if x < half:
            return (
                minus_sigma
                * (
                    (
                        one_div_sqrt_2d
                        * _phi_inv(
                            ((one_minus_phi_term * one_minux_2_x) + phi_term),
                            prec=prec,
                        )
                    )
                    + one_div_2d
                ).exp()
            ) + sigma_times_a

        if x > half:
            return (
                sigma
                * (
                    (
                        one_div_sqrt_2d
                        * _phi_inv(
                            ((one_minus_phi_term * two_x_minus_1) + phi_term),
                            prec=prec,
                        )
                    )
                    + one_div_2d
                ).exp()
            ) - sigma_times_a
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
