"""Example illustrating application of the exponential mechanism."""
from fractions import Fraction
# <placeholder: boilerplate>
from typing import Dict

import numpy as np
import pandas as pd

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasSeriesDomain
from tmlt.core.measurements.composition import (
    MeasurementQuery,
    create_adaptive_composition,
)
from tmlt.core.measurements.exponential_mechanism import ExponentialMechanism
from tmlt.core.measures import PureDP
from tmlt.core.metrics import AbsoluteDifference, DictMetric, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class CumulativeHistogram(Transformation):
    """Compute a discrete CDF of a numeric dataset."""

    def __init__(self, bins):
        # For each x in bin, count # records <= x and return a dictionary
        self._bins = bins

        super().__init__(
            input_domain=PandasSeriesDomain(NumpyFloatDomain()),
            input_metric=SymmetricDifference(),
            output_domain=DictDomain({str(key): NumpyIntegerDomain() for key in bins}),
            output_metric=DictMetric({str(key): AbsoluteDifference() for key in bins}),
        )

    def stability_function(self, d_in: ExactNumberInput) -> Dict[str, ExactNumber]:
        """Returns True only if close inputs produce close outputs.

        Args:
            d_in: Distance between inputs under HammingDistance().
        """
        self.input_metric.validate(d_in)
        return {str(key): ExactNumber(d_in) for key in self._bins}

    def __call__(self, data: pd.Series):
        """Perform the discrete CDF transformation."""
        bins = np.r_[-np.inf, self._bins, np.inf]
        intervals = pd.IntervalIndex.from_breaks(bins, closed="right")
        cdf = pd.cut(data, intervals).value_counts()[intervals].cumsum()
        return {str(key): val for key, val in zip(self._bins, cdf.values)}


class ReverseCumulativeHistogram(Transformation):
    """Compute a discrete survival function of a numeric dataset."""

    def __init__(self, bins):
        # For each x in bin, count # records >= x and return a dictionary
        self._bins = bins

        super().__init__(
            input_domain=PandasSeriesDomain(NumpyFloatDomain()),
            input_metric=SymmetricDifference(),
            output_domain=DictDomain({str(key): NumpyIntegerDomain() for key in bins}),
            output_metric=DictMetric({str(key): AbsoluteDifference() for key in bins}),
        )

    def stability_function(self, d_in: ExactNumberInput) -> Dict[str, ExactNumber]:
        """Returns True only if close inputs produce close outputs.
        Args:
            d_in: Distance between inputs under HammingDistance().
        """
        self.input_metric.validate(d_in)
        return {str(key): ExactNumber(d_in) for key in self._bins}

    def __call__(self, data: pd.Series):
        """Perform the discrete survival function transformation."""
        bins = np.r_[-np.inf, self._bins, np.inf]
        intervals = pd.IntervalIndex.from_breaks(bins, closed="left")
        cdf = pd.cut(data, intervals).value_counts()[intervals][::-1].cumsum()[::-1]
        return {str(key): val for key, val in zip(self._bins, cdf.values[1:])}


class ComputeRevenueFromCDF(Transformation):
    """Transformation to compute revenue from output of ReverseCumulativeHistogram."""

    def __init__(self, bins):
        self._bins = bins

        super().__init__(
            input_domain=DictDomain({str(key): NumpyIntegerDomain() for key in bins}),
            input_metric=DictMetric({str(key): AbsoluteDifference() for key in bins}),
            output_domain=DictDomain({str(key): NumpyFloatDomain() for key in bins}),
            output_metric=DictMetric({str(key): AbsoluteDifference() for key in bins}),
        )

    def stability_function(
        self, d_in: Dict[str, ExactNumberInput]
    ) -> Dict[str, ExactNumber]:
        """Returns True only if close inputs produce close outputs.

        Returns True only if the following holds:
            For every bin B, and for every pair on inputs x, x' such that
             | x[B] - x'[B] | <= d_in[B], we have
             | Transformation(x)[B] - Transformation(x')[B] | <= d_out[B]

        Args:
            d_in: Dictionary of distances between inputs under AbsoluteDifference().
            d_out: Dictionary of distances between outputs under AbsoluteDifference().
        """
        self.input_metric.validate(d_in)
        return {key: ExactNumber(d_in_i) for key, d_in_i in d_in.items()}

    def __call__(self, data: Dict[str, int]):
        return {str(key): key * data[str(key)] for key in self._bins}


def main():
    """Main function.  Example inspired by slide 10,
    https://www.cis.upenn.edu/~aaroth/courses/slides/Lecture3.pdf

    Given a dataset where each individual's data corresponds to the maximum price
     they are willing to pay for an apple, our goal is to select a price that will
     maximize revenue.  For a given price p, the revenue is equal to:

    Revenue = p * #(people whose max price is >= p)

    We will only consider prices from a finite set {$0.01, $0.02, ..., $9.99, $10.00}.
    We will use Revenue as the quality function, which has maximum sensitivity of 10
    (since the maximum price is 10, and the count has sensitivity 1).
    """
    df = pd.DataFrame(
        100 * [["Alice", 1.0], ["Bob", 1.0], ["Carol", 1.0], ["Dan", 4.01]],
        columns=["Name", "Max Price"],
    )
    # The best price is $4.01.  The next best price is $4.00 or $1.00.
    # $4.02 gives $0 revenue and $1.01 gives $1.01 revenue.

    prices = list(np.linspace(0.01, 10, 1000, endpoint=True))
    candidates = [str(p) for p in prices]
    exp_mech = ExponentialMechanism(
        output_measure=PureDP(),
        candidates=candidates,
        # privacy budget of 1, sensitivity is max(prices) = 10
        epsilon=Fraction(1, 10),
    )

    cdf_transformation = ReverseCumulativeHistogram(prices)
    score_transformation = ComputeRevenueFromCDF(prices)

    compute_scores_from_data = cdf_transformation | score_transformation

    get_best_price = compute_scores_from_data | exp_mech

    measurement = MeasurementQuery(measurement=get_best_price, d_out=1)

    interactive_measurement = create_adaptive_composition(
        input_domain=PandasSeriesDomain(NumpyFloatDomain()),
        input_metric=SymmetricDifference(),
        d_in=1,
        privacy_budget=1,
        output_measure=PureDP(),
    )
    queryable = interactive_measurement(df["Max Price"])

    # The mechanism should return prices close to and below $1.00, or prices close
    # to and below $4.01. It should not return prices like $1.05, $2.23, etc.
    # It might return prices around $3.00, but these will not be as likely.ß
    best_price = queryable(measurement)

    print("Best Price is %.2f" % float(best_price))


if __name__ == "__main__":
    main()
