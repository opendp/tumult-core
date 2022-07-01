"""Wrappers for changing a measurements's output measure."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Tuple

import sympy as sp
from typeguard import typechecked

from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class PureDPToRhoZCDP(Measurement):
    """Measurement for converting pure DP to zCDP."""

    @typechecked
    def __init__(self, pure_dp_measurement: Measurement):
        """Constructor.

        Args:
            pure_dp_measurement: The pure DP measurement to convert.
        """
        if pure_dp_measurement.output_measure != PureDP():
            raise ValueError("Input measure must be pure dp.")
        if pure_dp_measurement.is_interactive:
            raise ValueError("Can only convert non-interactive measurements.")

        self._pure_dp_measurement = pure_dp_measurement

        super().__init__(
            input_domain=pure_dp_measurement.input_domain,
            input_metric=pure_dp_measurement.input_metric,
            output_measure=RhoZCDP(),
            is_interactive=False,
        )

    @property
    def pure_dp_measurement(self) -> Measurement:
        """Return the wrapped pure DP measurement."""
        return self._pure_dp_measurement

    @typechecked
    def privacy_function(self, d_in: Any) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        The returned d_out (:math:`\rho`) is :math:`\rho=\frac{\epsilon^2}{2}`

        where :math:`\epsilon` is the d_out returned by :attr:`~.pure_dp_measurement`'s
        :meth:`~.Measurement.privacy_function` on d_in.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If self.pure_dp_measurement.privacy_function(d_in)
                raises :class:`NotImplementedError`.
        """
        epsilon = self.pure_dp_measurement.privacy_function(d_in)
        assert isinstance(epsilon, ExactNumber)
        rho = (epsilon ** 2) / 2
        self.output_measure.validate(rho)
        return rho

    @typechecked
    def privacy_relation(self, d_in: Any, d_out: ExactNumberInput) -> bool:
        r"""Return True if close inputs produce close outputs.

        Let :math:`\epsilon = \sqrt{2 \cdot \rho}`

        Returns self.pure_dp_measurement.privacy_relation(d_in, :math:`\epsilon`)

        where :math:`\rho` is the input argument "d_out".

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_measure.
        """
        self.input_metric.validate(d_in)
        self.output_measure.validate(d_out)
        pure_dp_epsilon = sp.sqrt(2 * ExactNumber(d_out).expr)
        return self.pure_dp_measurement.privacy_relation(d_in, pure_dp_epsilon)

    def __call__(self, data: Any):
        """Apply measurement."""
        return self.pure_dp_measurement(data)


class PureDPToApproxDP(Measurement):
    """Measurement for converting pure DP to approximate DP."""

    @typechecked
    def __init__(self, pure_dp_measurement: Measurement):
        """Constructor.

        Args:
            pure_dp_measurement: The pure DP measurement to convert.
        """
        if pure_dp_measurement.output_measure != PureDP():
            raise ValueError("Input measure must be pure DP.")
        if pure_dp_measurement.is_interactive:
            raise ValueError("Can only convert non-interactive measurements.")

        self._pure_dp_measurement = pure_dp_measurement

        super().__init__(
            input_domain=pure_dp_measurement.input_domain,
            input_metric=pure_dp_measurement.input_metric,
            output_measure=ApproxDP(),
            is_interactive=False,
        )

    @property
    def pure_dp_measurement(self) -> Measurement:
        """Return the wrapped pure DP measurement."""
        return self._pure_dp_measurement

    @typechecked
    def privacy_function(self, d_in: Any) -> Tuple[ExactNumber, ExactNumber]:
        r"""Returns the smallest d_out satisfied by the measurement.

        Returns (self.pure_dp_measurement.privacy_function(d_in), 0).

        Every (:math:`\epsilon`)-DP measurement is also (:math:`\epsilon`, 0)-DP.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If :attr:`~.pure_dp_measurement`'s
                :meth:`~.Measurement.privacy_function` raises
                :class:`NotImplementedError`.
        """
        return self.pure_dp_measurement.privacy_function(d_in), ExactNumber(0)

    @typechecked
    def privacy_relation(
        self, d_in: Any, d_out: Tuple[ExactNumberInput, ExactNumberInput]
    ) -> bool:
        r"""Returns the smallest d_out satisfied by the measurement.

        Returns self.pure_dp_measurement.privacy_relation(d_in, d_out[0])

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_measure.
        """
        self.input_metric.validate(d_in)
        self.output_measure.validate(d_out)
        return self.pure_dp_measurement.privacy_relation(d_in, d_out[0])

    def __call__(self, data: Any):
        """Apply measurement."""
        return self.pure_dp_measurement(data)


class RhoZCDPToApproxDP(Measurement):
    """Measurement for converting zCDP to approximate DP."""

    @typechecked
    def __init__(self, zcdp_measurement: Measurement):
        """Constructor.

        Args:
            zcdp_measurement: The zCDP measurement to convert.
        """
        if zcdp_measurement.output_measure != RhoZCDP():
            raise ValueError("Input measure must be rho zCDP.")
        if zcdp_measurement.is_interactive:
            raise ValueError("Can only convert non-interactive measurements.")

        self._zcdp_measurement = zcdp_measurement

        super().__init__(
            input_domain=zcdp_measurement.input_domain,
            input_metric=zcdp_measurement.input_metric,
            output_measure=ApproxDP(),
            is_interactive=False,
        )

    @property
    def zcdp_measurement(self) -> Measurement:
        """Return the wrapped zCDP measurement."""
        return self._zcdp_measurement

    @typechecked
    def privacy_relation(
        self, d_in: Any, d_out: Tuple[ExactNumberInput, ExactNumberInput]
    ) -> bool:
        r"""Return True if close inputs produce close outputs.

        Special cases:

        * Every measurement is (:math:`\infty, \delta`)-DP
          for any :math:`\delta \ge 0`
        * Every measurement is (:math:`epsilon, 1`)-DP
          for any :math:`\epsilon \ge 0`
        * :class:`~.RhoZCDP` with :math:`\rho \gt 0` cannot be converted to
          (:math:`\epsilon, 0`)-DP for any finite :math:`\epsilon`

        General case:

        Let :math:`\rho` be the unique solution to
        :math:`\rho - 2 * \sqrt{\rho * log(\frac{1}{\delta})} - \epsilon = 0`

        where:

        * :math:`\delta` is the first element of the input argument "d_out"
        * :math:`\epsilon` is the second element of the input argument "d_out"

        Returns self.zcdp_measurement.privacy_relation(d_in, :math:`\rho`)

        See Proposition 1.3 in :cite:`BunS16` for more information.

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_measure.
        """
        self.input_metric.validate(d_in)
        self.output_measure.validate(d_out)
        epsilon = ExactNumber(d_out[0]).expr
        delta = ExactNumber(d_out[1]).expr
        if epsilon == sp.oo or delta == 1:
            return True
        if delta == 0:
            return self.zcdp_measurement.privacy_relation(d_in, 0)
        # In the general case, we solve the conversion formula to find rho. It
        # is increasing between rho=0 and rho=infinity, negative in rho=0, and
        # unbounded when rho=infinity, so there is a single solution.
        rho = sp.Symbol("rho")
        solutions = sp.solve(
            rho + sp.Integer(2) * sp.sqrt(rho * sp.log(sp.Integer(1) / delta)) - epsilon
        )
        return self.zcdp_measurement.privacy_relation(d_in, solutions[0])

    def __call__(self, data: Any):
        """Apply measurement."""
        return self.zcdp_measurement(data)
