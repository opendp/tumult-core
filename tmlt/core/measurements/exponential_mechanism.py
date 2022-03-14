"""Measurements implementing variants on the exponential mechanism."""

# <placeholder: boilerplate>
import sys
from typing import Any, Dict, List, Union

import numpy as np
from scipy.special import softmax
from typeguard import typechecked

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference, DictMetric
from tmlt.core.random.rng import prng
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class ExponentialMechanism(Measurement):
    """A general-purpose implementation of the Exponential Mechanism."""

    @typechecked
    def __init__(
        self,
        output_measure: Union[PureDP, RhoZCDP],
        candidates: List[Any],
        epsilon: ExactNumberInput,
    ):
        """Constructor.

        Args:
            output_measure: Output measure.
            candidates: A list of candidates.
            epsilon: The privacy parameter to use for this mechanism.
        """
        if len(candidates) == 0:
            raise ValueError("candidates list must be non-empty.")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidates list must not contain duplicates.")
        epsilon = ExactNumber(epsilon)
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        self._candidates = list(candidates)
        self._epsilon = epsilon

        super().__init__(
            input_domain=DictDomain({key: NumpyFloatDomain() for key in candidates}),
            input_metric=DictMetric({key: AbsoluteDifference() for key in candidates}),
            output_measure=output_measure,
            is_interactive=False,
        )

    @property
    def candidates(self) -> List[Any]:
        """Returns the list of candidates to select from."""
        return self._candidates.copy()

    @property
    def epsilon(self) -> ExactNumber:
        """Returns the privacy parameter used for calling this mechanism."""
        return self._epsilon

    @typechecked
    def privacy_function(self, d_in: Dict[Any, ExactNumberInput]) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        Let :math:`\Delta = \max_i(d_{in}[i])`.

        The returned d_out is:

        * :math:`\epsilon \cdot \Delta` if the output measure is
          :class:`~.PureDP`
        * :math:`\frac{1}{8}(\epsilon \cdot \Delta)^2` if the output measure is
          :class:`~.RhoZCDP`

        where:

        * :math:`d_{in}` is the input argument "d_in"
        * :math:`\epsilon` is :attr:`~.epsilon`

        See :cite:`Cesar021` for the :class:`~.RhoZCDP` privacy analysis.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        delta = max(ExactNumber(d_in_i) for d_in_i in d_in.values())
        if self.output_measure == PureDP():
            return self.epsilon * delta
        assert self.output_measure == RhoZCDP()
        return (self.epsilon * delta) ** 2 / 8

    def __call__(self, quality_scores: Dict[Any, np.float64]) -> Any:
        r"""Select a candidate based on the input quality scores.

        Let :math:`Z = \sum_i \exp(\epsilon \cdot q[i] / 2)`

        Candidate is sampled according to the following probability distribution:

        :math:`\Pr[M(q) = i] = \frac{1}{Z} \exp(\epsilon \cdot q[i] / 2)`,

        where:

        * :math:`\epsilon` is :attr:`~.epsilon`
        * :math:`q` are the quality_scores

        See :cite:`McSherryT07` for more details about this mechanism.

        Args:
            quality_scores: The input quality scores used as the selection criteria
                (higher scoring candidates are more likely to be chosen).
        """
        n = len(self._candidates)
        eps = min(self.epsilon.to_float(round_up=False), sys.float_info.max)
        scores = np.array([quality_scores[key] for key in self._candidates])
        with np.errstate(over="ignore"):
            logits = 0.5 * eps * (scores - scores.max())
        probas = softmax(logits)
        idx = prng().choice(range(n), p=probas)
        return self._candidates[idx]


class PermuteAndFlip(Measurement):
    """A general-purpose implementation of the Permute-and-Flip mechanism."""

    @typechecked
    def __init__(self, candidates: List[Any], epsilon: ExactNumberInput):
        """Constructor.

        Args:
            candidates: A list of candidates.
            epsilon: The privacy parameter to use for this mechanism.
        """
        if len(candidates) == 0:
            raise ValueError("candidates list must be non-empty.")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidates list must not contain duplicates.")
        epsilon = ExactNumber(epsilon)
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        self._candidates = list(candidates)
        self._epsilon = epsilon

        super().__init__(
            input_domain=DictDomain({key: NumpyFloatDomain() for key in candidates}),
            input_metric=DictMetric({key: AbsoluteDifference() for key in candidates}),
            output_measure=PureDP(),
            is_interactive=False,
        )

    @property
    def candidates(self) -> List[Any]:
        """Returns the list of candidates to select from."""
        return self._candidates.copy()

    @property
    def epsilon(self) -> ExactNumber:
        """Returns the privacy parameter used for calling this mechanism."""
        return self._epsilon

    @typechecked
    def privacy_function(self, d_in: Dict[Any, ExactNumberInput]) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        Let :math:`\Delta = \max_i(d_{in}[i])`.

        The returned d_out is :math:`\epsilon \cdot \Delta`.

        where:

        * :math:`d_{in}` is the input argument "d_in"
        * :math:`\epsilon` is :attr:`~.epsilon`

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        delta = max(ExactNumber(d_in_i) for d_in_i in d_in.values())
        return self.epsilon * delta

    def __call__(self, quality_scores: Dict[Any, np.float64]) -> Any:
        r"""Select a candidate based on the input quality scores.

        See :cite:`McKennaS20` for more details about this mechanism.

        Note that this implementation does not calibrate probabilities based on
        the sensitivity term :math:`\Delta`, as described in :cite:`McKennaS20`.
        Instead, :math:`\Delta` appears in the privacy_function.

        Specifically, the coin flip probability for a candidate i is:

        :math:`p_i = \exp{(0.5 \cdot \epsilon \cdot (q[i] - max_j q[j]))}`

        where:

        * :math:`\epsilon` is :attr:`~.epsilon`
        * :math:`q` are the quality_scores

        Note that if the :math:`\Delta` is 1, then this implementation is
        equivalent to the one described in :cite:`McKennaS20`.

        Args:
            quality_scores: The input quality scores used as the selection criteria
                (higher scoring candidates are more likely to be chosen).
        """
        eps = min(self.epsilon.to_float(round_up=False), sys.float_info.max)
        scores = np.array([quality_scores[key] for key in self._candidates])
        with np.errstate(over="ignore"):
            probas = np.exp(0.5 * eps * (scores - scores.max()))
        for i in prng().permutation(scores.size):
            if prng().random() <= probas[i]:
                return self._candidates[i]
        return self._candidates[np.argmax(scores)]
