"""Measurements on whole pandas tables.

This is the pandas counterpart of
:mod:`tmlt.core.measurements.spark_measurements`, which currently covers adding
noise to one aggregated column. See `the architecture guide
<https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_ for more
information.

Warning:
    A measurement in this module draws its noise from
    :data:`tmlt.core.random.rng.prng`, a process-global generator. A forked
    child process inherits its parent's generator *state*, so two children
    forked from one parent draw the *same* noise. Nothing here may therefore be
    run under :mod:`multiprocessing` (or any other library that forks worker
    processes) to noise shards of one table in parallel: the shards would be
    given identical noise, which is not the mechanism whose privacy loss the
    privacy function accounts for. Adding noise is cheap next to the
    aggregation that produced the column anyway.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.pandas_domains import PandasTableDomain
from tmlt.core.exceptions import DomainMismatchError, UnsupportedMetricError
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.series import AddNoiseToSeries
from tmlt.core.metrics import OnColumn, RootSumOfSquared, SumOf
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import get_fullname

_NOISE_OUTPUT_DTYPES: dict[type, np.dtype] = {
    AddGeometricNoise: np.dtype("int64"),
    AddDiscreteGaussianNoise: np.dtype("int64"),
    AddLaplaceNoise: np.dtype("float64"),
    AddGaussianNoise: np.dtype("float64"),
}
"""The dtype the column a noise mechanism noised must end up with.

The discrete mechanisms return a Python :class:`int` and the continuous ones a
:class:`float`, which is the same split as the Spark type each mechanism
declares as its ``output_type`` -- ``LongType`` and ``DoubleType`` respectively
-- and which the Spark implementation gives the noised column through the type
of its pandas UDF. Pandas infers a column's dtype from its values instead, and
infers nothing at all from no values, so the dtype is stated here rather than
left to inference; see :meth:`AddNoiseToColumn.__call__`.

Lookups go through :func:`_noise_output_dtype`, which matches a mechanism by
``isinstance`` rather than by exact type, so a subclass of one of these gets
its base's dtype -- the same answer the Spark twin gets by delegating to the
mechanism's ``output_type``.
"""


def _noise_output_dtype(noise_measurement: Measurement) -> np.dtype:
    """Returns the dtype a column ``noise_measurement`` noised must end up with.

    Args:
        noise_measurement: The noise mechanism whose output dtype to resolve.

    Raises:
        ValueError: If the mechanism is not one of the ones in
            :data:`_NOISE_OUTPUT_DTYPES`, nor a subclass of one.
    """
    for mechanism, dtype in _NOISE_OUTPUT_DTYPES.items():
        if isinstance(noise_measurement, mechanism):
            return dtype
    raise ValueError(
        "AddNoiseToColumn has no output dtype for noise mechanism"
        f" {get_fullname(noise_measurement)}. The pandas backend must state the"
        " dtype of the column a mechanism noised, since pandas infers nothing"
        " from an empty column, and it knows only"
        f" {', '.join(sorted(get_fullname(m) for m in _NOISE_OUTPUT_DTYPES))}."
    )


class AddNoiseToColumn(Measurement):
    """Adds noise to a single aggregated column of a pandas DataFrame.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.measurements.noise_mechanisms import (
            ...     AddLaplaceNoise,
            ... )
            >>> from tmlt.core.measurements.pandas_measurements.series import (
            ...     AddNoiseToSeries,
            ... )
            >>> from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasIntegerColumnDescriptor,
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a1", "a2", "a2"],
            ...         "B": ["b1", "b2", "b1", "b2"],
            ...         "count": [3, 2, 1, 0],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B  count
        0  a1  b1      3
        1  a1  b2      2
        2  a2  b1      1
        3  a2  b2      0
        >>> # Create a measurement that can add noise to a pd.Series
        >>> add_laplace_noise = AddLaplaceNoise(
        ...     scale="0.5",
        ...     input_domain=NumpyIntegerDomain(),
        ... )
        >>> # Create a measurement that can add noise to a pandas DataFrame
        >>> add_laplace_noise_to_column = AddNoiseToColumn(
        ...     input_domain=PandasTableDomain(
        ...         schema={
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...             "count": PandasIntegerColumnDescriptor(),
        ...         },
        ...     ),
        ...     measurement=AddNoiseToSeries(add_laplace_noise),
        ...     measure_column="count",
        ... )
        >>> # Apply measurement to data
        >>> noisy_dataframe = add_laplace_noise_to_column(dataframe)
        >>> print_pandas(noisy_dataframe)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            A   B   count
        0  a1  b1 ...
        1  a1  b2 ...
        2  a2  b1 ...
        3  a2  b2 ...

    Measurement Contract:
        * Input domain - :class:`~.PandasTableDomain`
        * Output type - pandas DataFrame
        * Input metric - :class:`~.OnColumn` with metric
          ``SumOf(SymmetricDifference())`` (for :class:`~.PureDP`) or
          ``RootSumOfSquared(SymmetricDifference())`` (for :class:`~.RhoZCDP`) on each
          column.
        * Output measure - :class:`~.PureDP` or :class:`~.RhoZCDP`

        >>> add_laplace_noise_to_column.input_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False), 'count': PandasIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> add_laplace_noise_to_column.input_metric
        OnColumn(column='count', metric=SumOf(inner_metric=AbsoluteDifference()))
        >>> add_laplace_noise_to_column.output_measure
        PureDP()

        Privacy Guarantee:
            :class:`~.AddNoiseToColumn`'s :meth:`~.privacy_function` returns the output
            of privacy function on the :class:`~.AddNoiseToSeries` measurement.

            >>> add_laplace_noise_to_column.privacy_function(1)
            2

    Note:
        This subclasses :class:`~.Measurement` directly, where the Spark
        counterpart subclasses
        :class:`~tmlt.core.measurements.spark_measurements.SparkMeasurement`.
        That base class exists because a Spark DataFrame is a *plan*: the noise
        a measurement adds is recomputed, differently, every time the frame is
        collected, so the plan has to be materialized and re-read from a
        checkpoint before it can be handed back (see
        :ref:`pseudo-side-channel-mitigations`, and the
        ``tmlt.core.measurements.spark_measurements._get_sanitized_df`` that
        :class:`~.SparkMeasurement`'s ``__call__`` puts every output through).
        Pandas is eager: by the time
        :meth:`__call__` returns, the noise has been drawn once and lives in a
        concrete numpy array that no later operation can redraw. There is
        nothing to re-materialize, so there is no pandas counterpart of that
        base class and none is needed.
    """  # noqa: E501

    # output_dtype is a restatement of the wrapped mechanism, which is rendered
    # as this measurement's child anyway; hiding it keeps the head line the same
    # as the Spark twin's.
    FORMAT_EXCLUDED_ATTRS = Measurement.FORMAT_EXCLUDED_ATTRS | {"output_dtype"}
    """Fields hidden from output when formatting this measurement. @nodoc"""

    @typechecked
    def __init__(
        self,
        input_domain: PandasTableDomain,
        measurement: AddNoiseToSeries,
        measure_column: str,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input pandas DataFrames.
            measurement: :class:`~.AddNoiseToSeries` measurement for adding noise to
                ``measure_column``.
            measure_column: Name of column to add noise to.

        Note:
            The input metric of this measurement is derived from the ``measure_column``
            and the input metric of the ``measurement`` to be applied. In particular,
            the input metric of this measurement is ``measurement.input_metric`` on the
            specified ``measure_column``.

        Raises:
            DomainMismatchError: If ``measure_column``'s domain is not the domain
                the ``measurement`` adds noise to.
            UnsupportedMetricError: If the ``measurement``'s input metric is
                neither :class:`~.SumOf` nor :class:`~.RootSumOfSquared`.
            ValueError: If the dtype the ``measurement``'s noise mechanism
                leaves its column with is not known; see
                :data:`_NOISE_OUTPUT_DTYPES`.
        """
        measure_column_domain = input_domain[measure_column].to_numpy_domain()
        if measure_column_domain != measurement.input_domain.element_domain:
            raise DomainMismatchError(
                (measure_column_domain, measurement.input_domain.element_domain),
                (
                    f"{measure_column} has domain {measure_column_domain}, which is"
                    " incompatible with measurement's input domain"
                    f" {measurement.input_domain.element_domain}"
                ),
            )
        # The Spark counterpart asserts this instead. An AddNoiseToSeries always
        # has one of these two metrics, so neither can fire, but an assert is
        # stripped under -O and says nothing useful when it does fire.
        if not isinstance(measurement.input_metric, (SumOf, RootSumOfSquared)):
            raise UnsupportedMetricError(
                measurement.input_metric,
                (
                    "The measurement's input metric must be SumOf or"
                    f" RootSumOfSquared, not {measurement.input_metric}."
                ),
            )
        # Resolved here rather than when the output dtype is asked for, which
        # is after __call__ has drawn its noise: a measurement that cannot say
        # what dtype its output column has cannot be built at all, and no
        # privacy budget is spent finding that out.
        output_dtype = _noise_output_dtype(measurement.noise_measurement)
        super().__init__(
            input_domain=input_domain,
            input_metric=OnColumn(measure_column, measurement.input_metric),
            output_measure=measurement.output_measure,
            is_interactive=False,
        )
        self._measure_column = measure_column
        self._measurement = measurement
        self._output_dtype = output_dtype

    @property
    def input_domain(self) -> PandasTableDomain:
        """Return input domain for the measurement."""
        return cast(PandasTableDomain, super().input_domain)

    @property
    def measure_column(self) -> str:
        """Returns the name of the column to add noise to."""
        return self._measure_column

    @property
    def measurement(self) -> AddNoiseToSeries:
        """The :class:`~.AddNoiseToSeries` measurement to apply to measure column."""
        return self._measurement

    @property
    def output_dtype(self) -> np.dtype:
        """Returns the dtype the measure column has after noise is added.

        The discrete mechanisms leave it an ``int64`` column and the continuous
        ones make it a ``float64`` one, matching the Spark type the wrapped
        mechanism declares as its ``output_type``. It is resolved when the
        measurement is built, so this cannot fail.
        """
        return self._output_dtype

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        See `the architecture guide <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If the :meth:`~.Measurement.privacy_function` of the
                :class:`~.AddNoiseToSeries` measurement
                raises :class:`NotImplementedError`.
        """
        self.input_metric.validate(d_in)
        return self.measurement.privacy_function(d_in)

    def __call__(self, val: pd.DataFrame) -> pd.DataFrame:
        """Applies measurement to measure column.

        The returned frame is a new object, indexed from zero; the frame passed
        in is never written to. As everywhere in the pandas backend, the
        returned frame is to be treated as immutable -- when no noise is added
        it may share its buffers with the input -- see
        :class:`~.PandasTableDomain`'s note on mutability.

        Args:
            val: The frame whose measure column to noise.
        """
        if self.measurement.noise_measurement.adds_no_noise:
            # Mirrors the Spark implementation's short-circuit, which returns
            # the frame it was given untouched. Reindexing is not part of that:
            # it is this implementation's own guarantee, and it copies nothing
            # the caller holds.
            return val.reset_index(drop=True)
        # Copied before anything is written to it, so that neither the caller's
        # frame nor anything sharing its buffers changes under it. The reindex
        # is both the output guarantee and what makes the assignment below --
        # which aligns on the index -- line up with the column it noised.
        df = val.reset_index(drop=True).copy()
        noised = self.measurement(df[self.measure_column])
        # The cast is explicit because pandas would otherwise infer the dtype
        # from the values: a zero-row frame gives `apply` nothing to infer from
        # and leaves an object column behind, and a continuous mechanism that
        # happened to draw only integral values would give an integer one.
        df[self.measure_column] = noised.astype(self.output_dtype)
        return df
