"""Floating-point safe utility functions for per-record diffential privacy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from tmlt.core.random.continuous_gaussian import gaussian_inverse_cdf
from tmlt.core.random.inverse_cdf import construct_inverse_sampler
from tmlt.core.utils.arb import (
    Arb,
    arb_add,
    arb_div,
    arb_erf,
    arb_erfinv,
    arb_exp,
    arb_lambertw,
    arb_log,
    arb_mul,
    arb_neg,
    arb_pow,
    arb_sqrt,
    arb_sub,
)


def fourth_root_transformation_mechanism(
    x: float, offset: float, sigma: float
) -> float:
    """Fourth root transformation mechanism."""
    x_arb = Arb.from_float(x)
    sigma_arb = Arb.from_float(sigma)
    offset_arb = Arb.from_float(offset)
    one_fourth = Arb.from_float(1 / 4)
    two = Arb.from_int(2)
    four = Arb.from_int(4)

    def inverse_cdf(p: Arb, prec: int) -> Arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        u_arb = arb_pow(arb_add(x_arb, offset_arb, prec=prec), one_fourth, prec=prec)
        sigma_squared_arb = arb_pow(sigma_arb, two, prec=prec)
        gaussian_sample = gaussian_inverse_cdf(
            u=u_arb, sigma_squared=sigma_squared_arb, p=p, prec=prec
        )
        return arb_sub(arb_pow(gaussian_sample, four, prec=prec), offset_arb, prec=prec)

    return construct_inverse_sampler(inverse_cdf=inverse_cdf)()


def square_root_transformation_mechanism(
    x: float, offset: float, sigma: float
) -> float:
    """Square root transformation mechanism."""
    x_arb = Arb.from_float(x)
    offset_arb = Arb.from_float(offset)
    sigma_arb = Arb.from_float(sigma)
    two = Arb.from_int(2)

    def inverse_cdf(p: Arb, prec: int) -> Arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        u_arb = arb_sqrt(arb_add(x_arb, offset_arb, prec=prec), prec=prec)
        sigma_squared_arb = arb_pow(sigma_arb, two, prec=prec)
        gaussian_sample = gaussian_inverse_cdf(
            u=u_arb, sigma_squared=sigma_squared_arb, p=p, prec=prec
        )
        return arb_sub(arb_pow(gaussian_sample, two, prec=prec), offset_arb, prec=prec)

    return construct_inverse_sampler(inverse_cdf=inverse_cdf)()


def log_transformation_mechanism(x: float, offset: float, sigma: float) -> float:
    """Log transformation mechanism."""
    x_arb = Arb.from_float(x)
    offset_arb = Arb.from_float(offset)
    sigma_arb = Arb.from_float(sigma)
    two = Arb.from_int(2)

    def inverse_cdf(p: Arb, prec: int) -> Arb:
        """Inverse CDF for the post-processed Gaussian distribution."""
        u_arb = arb_log(arb_add(x_arb, offset_arb, prec=prec), prec=prec)
        sigma_squared_arb = arb_pow(sigma_arb, two, prec=prec)
        gaussian_sample = gaussian_inverse_cdf(
            u=u_arb, sigma_squared=sigma_squared_arb, p=p, prec=prec
        )
        return arb_sub(
            arb_exp(gaussian_sample, prec=prec),
            offset_arb,
            prec=prec,
        )

    return construct_inverse_sampler(inverse_cdf=inverse_cdf)()


def square_root_gaussian_inverse_cdf(x: Arb, sigma: Arb, prec: int) -> Arb:
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

    """  # pylint: disable=line-too-long,useless-suppression
    if x == Arb.from_float(0.5):
        return Arb.from_int(0)

    zero = Arb.from_int(0)
    half = Arb.from_float(0.5)
    one = Arb.from_int(1)
    two = Arb.from_int(2)
    e_arb = arb_exp(one, prec=prec)

    if x > half:
        lambertw_arg = arb_div(
            arb_sub(arb_mul(Arb.from_int(2), x, prec=prec), two, prec=prec),
            e_arb,
            prec=prec,
        )
        lambertw_branch = 0 if lambertw_arg >= zero else 1
        lambert_term = arb_lambertw(lambertw_arg, branch=lambertw_branch, prec=prec)
        return arb_mul(
            sigma,
            arb_pow(arb_add(lambert_term, one, prec=prec), two, prec=prec),
            prec=prec,
        )

    if x < half:
        lambertw_arg = arb_div(
            arb_mul(arb_neg(Arb.from_int(2)), x, prec=prec), e_arb, prec=prec
        )
        lambertw_branch = 0 if lambertw_arg >= zero else 1
        lambert_term = arb_lambertw(lambertw_arg, branch=lambertw_branch, prec=prec)
        return arb_mul(
            arb_neg(sigma),
            arb_pow(arb_add(lambert_term, one, prec=prec), two, prec=prec),
            prec=prec,
        )

    # NOTE: It is possible that none of the above conditions are true.
    # In this case, we return the interval (-inf, inf). The inverse CDF
    # sampler should re-try with more precision.
    return Arb.from_midpoint_radius(mid=0, rad=float("inf"))


def square_root_gaussian_mechanism(sigma: float) -> float:
    """Samples a float from the generalized Gaussian distribution."""
    sigma_arb = Arb.from_float(sigma)
    return construct_inverse_sampler(
        inverse_cdf=lambda p, prec: square_root_gaussian_inverse_cdf(p, sigma_arb, prec)
    )()


def _phi(x: Arb, prec: int) -> Arb:
    """CDF for the unit Gaussian distribution N(0, 1)."""
    half = Arb.from_float(0.5)
    erf_arg = arb_div(x, arb_sqrt(Arb.from_int(2), prec=prec), prec=prec)
    return arb_mul(
        half,
        arb_add(Arb.from_int(1), arb_erf(erf_arg, prec=prec), prec=prec),
        prec=prec,
    )


def _phi_inv(p: Arb, prec: int) -> Arb:
    """Inverse CDF for the unit Gaussian distribution N(0, 1)."""
    return arb_mul(
        arb_sqrt(Arb.from_int(2), prec=prec),
        arb_erfinv(
            arb_sub(arb_mul(Arb.from_int(2), p, prec=prec), Arb.from_int(1), prec=prec),
            prec=prec,
        ),
        prec=prec,
    )


def exponential_polylogarithmic_inverse_cdf(
    x: Arb, d: Arb, a: Arb, sigma: Arb, prec: int
) -> Arb:
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

    """  # pylint: disable=line-too-long,useless-suppression
    if x == Arb.from_float(0.5):
        return Arb.from_int(0)

    minus_sigma = arb_neg(sigma)
    half = Arb.from_float(0.5)
    one = Arb.from_int(1)
    two_d = arb_mul(Arb.from_int(2), d, prec=prec)

    two_x_minus_1 = arb_sub(arb_mul(Arb.from_int(2), x, prec=prec), one, prec=prec)
    one_minux_2_x = arb_neg(two_x_minus_1)

    log_a = arb_log(a, prec=prec)
    sqrt_2d = arb_sqrt(two_d, prec=prec)
    one_div_sqrt_2d = arb_div(one, sqrt_2d, prec=prec)
    one_div_2d = arb_div(one, two_d, prec=prec)

    sigma_times_a = arb_mul(sigma, a, prec=prec)

    phi_arg = arb_div(arb_sub(log_a, one_div_2d, prec=prec), one_div_sqrt_2d, prec=prec)
    phi_term = _phi(phi_arg, prec=prec)
    one_minus_phi_term = arb_sub(one, phi_term, prec=prec)

    if x < half:
        return arb_add(
            arb_mul(
                minus_sigma,
                arb_exp(
                    arb_add(
                        arb_mul(
                            one_div_sqrt_2d,
                            _phi_inv(
                                arb_add(
                                    arb_mul(
                                        one_minus_phi_term, one_minux_2_x, prec=prec
                                    ),
                                    phi_term,
                                    prec=prec,
                                ),
                                prec=prec,
                            ),
                            prec=prec,
                        ),
                        one_div_2d,
                        prec=prec,
                    ),
                    prec=prec,
                ),
                prec=prec,
            ),
            sigma_times_a,
            prec=prec,
        )

    if x > half:
        return arb_sub(
            arb_mul(
                sigma,
                arb_exp(
                    arb_add(
                        arb_mul(
                            one_div_sqrt_2d,
                            _phi_inv(
                                arb_add(
                                    arb_mul(
                                        one_minus_phi_term, two_x_minus_1, prec=prec
                                    ),
                                    phi_term,
                                    prec=prec,
                                ),
                                prec=prec,
                            ),
                            prec=prec,
                        ),
                        one_div_2d,
                        prec=prec,
                    ),
                    prec=prec,
                ),
                prec=prec,
            ),
            sigma_times_a,
            prec=prec,
        )
    # NOTE: It is possible that none of the above conditions are true.
    # In this case, we return the interval (-inf, inf). The inverse CDF
    # sampler should re-try with more precision.
    return Arb.from_midpoint_radius(mid=0, rad=float("inf"))


def exponential_polylogarithmic_mechanism(
    d: float, a: float, sigma: float, step_size: int = 63
) -> float:
    """Samples a float from the exponential polylogarithmic distribution."""
    d_arb = Arb.from_float(d)
    a_arb = Arb.from_float(a)
    sigma_arb = Arb.from_float(sigma)
    return construct_inverse_sampler(
        inverse_cdf=lambda p, prec: exponential_polylogarithmic_inverse_cdf(
            p, d_arb, a_arb, sigma_arb, prec
        ),
        step_size=step_size,
    )()
