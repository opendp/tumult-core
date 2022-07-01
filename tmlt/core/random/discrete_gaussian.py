"""Module for discrete Gaussian sampling."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# This file is derived from a work authored by Thomas Steinke dgauss@thomas-steinke.net,
# copyrighted by IBM Corp. 2020, licensed under Apache 2.0, and available at
# https://github.com/IBM/discrete-gaussian-differential-privacy.

# This module contains code modified from the file `discretegauss.py` from the authors'
# repository at commit `cb190d2a990a78eff6e21159203bc888e095f01b`.

import math
import random
from fractions import Fraction
from typing import Optional, Union

from typeguard import typechecked
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class SupportsRandRange(Protocol):
    """Protocol class defining randrange."""

    def randrange(self, high: int) -> int:
        """Returns an integer sampled uniformly from [0, high)."""
        ...


def _sample_uniform(m: int, rng: SupportsRandRange) -> int:
    """Returns an integer sampled uniformly from [0, m).

    Args:
        m: Upper bound (exclusive) for sampling.
        rng: Random Number Generator for sampling.
    """
    if m <= 0:
        raise ValueError(f"Expected a positive integer, not {m}.")
    return rng.randrange(m)


def _sample_bernoulli(p: Fraction, rng: SupportsRandRange) -> int:
    """Returns a sample from the Bernoulli(p) distribution.

    Args:
        p: Probability of obtaining 1.
        rng: Random Number Generator for sampling.
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability must be in [0, 1], not {p}")
    m = _sample_uniform(p.denominator, rng)
    if m < p.numerator:
        return 1
    else:
        return 0


def _sample_bernoulli_exp1(x: Fraction, rng: SupportsRandRange) -> int:
    """Returns a sample from Bernoulli(exp(-x)) distribution.

    Args:
        x: Fraction satisfying 0 <= x <= 1.
        rng: Random Number Generator for sampling.
    """
    if not 0 <= x <= 1:
        raise ValueError(f"Expected x in [0, 1], not {x}.")
    k = 1
    while True:
        if _sample_bernoulli(x / k, rng) == 1:
            k = k + 1
        else:
            break
    return k % 2


def _sample_bernoulli_exp(x: Fraction, rng: SupportsRandRange) -> int:
    """Returns a sample from a Bernoulli(exp(-x)) distribution.

    Args:
        x: Fraction satisfying x >= 0.
        rng: Random Number Generator for sampling.
    """
    if not x >= 0:
        raise ValueError("x must be non-negative.")
    # Sample floor(x) independent Bernoulli(exp(-1))
    # If all are 1, return Bernoulli(exp(-(x-floor(x))))
    while x > 1:
        if _sample_bernoulli_exp1(Fraction(1, 1), rng) == 1:
            x = x - 1
        else:
            return 0
    return _sample_bernoulli_exp1(x, rng)


def _sample_geometric_exp_slow(x: Fraction, rng: SupportsRandRange) -> int:
    """Returns a sample from a geometric(1-exp(-x)) distribution.

    Args:
        x: Fraction satisfying x >= 0.
        rng: Random Number Generator for sampling.
    """
    if not x >= 0:
        raise ValueError("x must be non-negative.")
    k = 0
    while True:
        if _sample_bernoulli_exp(x, rng) == 1:
            k = k + 1
        else:
            return k


def _sample_geometric_exp_fast(x: Fraction, rng: SupportsRandRange) -> int:
    """Returns a sample from a geometric(1-exp(-x)) distribution.

    Args:
        x: Fraction satisfying x > 0.
        rng: Random Number Generator for sampling.
    """
    if x == 0:
        return 0  # degenerate case
    if not x > 0:
        raise ValueError("x must be positive.")

    t = x.denominator
    while True:
        u = _sample_uniform(t, rng)
        b = _sample_bernoulli_exp(Fraction(u, t), rng)
        if b == 1:
            break
    v = _sample_geometric_exp_slow(Fraction(1, 1), rng)
    value = v * t + u
    return value // x.numerator


def _sample_dlaplace(
    scale: Union[float, Fraction], rng: Optional[SupportsRandRange] = None
) -> int:
    r"""Returns a sample from a discrete Laplace distribution.

    In particular, this returns an integer :math:`x` with
    .. math::
        Pr(x) = exp(-\frac{|x|}{scale}) \cdot \frac{exp(\frac{1}{scale}) - 1}{exp(\frac{1}{xcale}) +1}  # pylint: disable=line-too-long

    Args:
        scale: Desired noise scale (>=0).
        rng: Random Number Generator for sampling.
    """
    if rng is None:
        rng = random.SystemRandom()
    scale_fraction = Fraction(scale)
    if scale_fraction < 0:
        raise ValueError("scale must be nonnegative.")
    while True:
        sign = _sample_bernoulli(Fraction(1, 2), rng)
        magnitude = _sample_geometric_exp_fast(1 / scale_fraction, rng)
        if sign == 1 and magnitude == 0:
            continue
        return magnitude * (1 - 2 * sign)


def _floorsqrt(x: Union[float, int, Fraction]) -> int:
    """Returns floor of square root of the input."""
    if x < 0:
        raise ValueError("x must be positive.")
    a = 0  # maintain a^2<=x
    b = 1  # maintain b^2>x
    while b * b <= x:
        b = 2 * b  # double to get upper bound
    # now do binary search
    while a + 1 < b:
        c = (a + b) // 2  # c=floor((a+b)/2)
        if c * c <= x:
            a = c
        else:
            b = c
    return a


@typechecked
def sample_dgauss(
    sigma_squared: Union[float, Fraction, int], rng: SupportsRandRange = None
) -> int:
    r"""Returns a sample from a discrete Gaussian distribution.

    In particular, this returns a sample from discrete Gaussian
        :math:`\mathcal{N}_{\mathbb{Z}}(sigma\_squared)`

    Args:
        sigma_squared: Variance of discrete Gaussian distribution to sample from.
        rng: Random Number Generator for sampling.
    """
    if rng is None:
        rng = random.SystemRandom()
    if math.isnan(sigma_squared) or math.isinf(sigma_squared):
        raise ValueError(f"sigma_squared must be positive, not {sigma_squared}.")
    sigma_squared_fraction = Fraction(sigma_squared)
    if sigma_squared_fraction == 0:
        return 0  # degenerate case
    if sigma_squared_fraction <= 0:
        raise ValueError("sigma_squared must be positive.")
    t = _floorsqrt(sigma_squared_fraction) + 1
    while True:
        candidate = _sample_dlaplace(t, rng=rng)
        bias = ((abs(candidate) - sigma_squared_fraction / t) ** 2) / (
            2 * sigma_squared_fraction
        )
        if _sample_bernoulli_exp(bias, rng) == 1:
            return candidate


#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS
