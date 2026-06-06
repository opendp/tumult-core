"""Tests for :mod:`tmlt.core.utils.format`."""

# SPDX-License-Identifier: Apache-2.0

import textwrap
from typing import Any, Callable, List

import pytest

from tmlt.core.domains.base import Domain
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.chaining import ChainTM
from tmlt.core.measurements.composition import Composition
from tmlt.core.measures import Measure, PureDP
from tmlt.core.metrics import AbsoluteDifference, Metric
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.identity import Identity
from tmlt.core.utils.format import format_labeled_siblings


class _TaggedTransformation(Transformation):
    """An Identity-like transformation with an extra ``tag`` attribute."""

    def __init__(self, domain: Domain, metric: Metric, tag: str):
        super().__init__(
            input_domain=domain,
            input_metric=metric,
            output_domain=domain,
            output_metric=metric,
        )
        self._tag = tag

    @property
    def tag(self) -> str:
        """The tag value."""
        return self._tag

    def stability_function(self, d_in: Any) -> Any:
        return d_in

    def __call__(self, data: Any) -> Any:
        return data


class _NoOpMeasurement(Measurement):
    """A leaf measurement that just returns its input."""

    def __init__(
        self,
        domain: Domain,
        metric: Metric,
        measure: Measure,
        label: str,
    ):
        super().__init__(
            input_domain=domain,
            input_metric=metric,
            output_measure=measure,
            is_interactive=False,
        )
        self._label = label

    @property
    def label(self) -> str:
        """A human-readable label for this measurement."""
        return self._label

    def privacy_function(self, d_in: Any) -> Any:
        return d_in

    def __call__(self, data: Any) -> Any:
        return data


class _MeasurementWrapper(Measurement):
    """A measurement that wraps another measurement plus a callback."""

    def __init__(self, measurement: Measurement, f: Callable[[Any], Any]):
        super().__init__(
            input_domain=measurement.input_domain,
            input_metric=measurement.input_metric,
            output_measure=measurement.output_measure,
            is_interactive=measurement.is_interactive,
        )
        self._measurement = measurement
        self._f = f

    @property
    def measurement(self) -> Measurement:
        """The wrapped measurement."""
        return self._measurement

    @property
    def f(self) -> Callable[[Any], Any]:
        """The post-processing function."""
        return self._f

    def privacy_function(self, d_in: Any) -> Any:
        return self.measurement.privacy_function(d_in)

    def __call__(self, data: Any) -> Any:
        return self.f(self.measurement(data))


# --- Fixtures -----------------------------------------------------------------

_DOMAIN = NumpyIntegerDomain()
_METRIC = AbsoluteDifference()
_MEASURE = PureDP()


def _t(tag: str) -> _TaggedTransformation:
    return _TaggedTransformation(_DOMAIN, _METRIC, tag)


def _m(label: str) -> _NoOpMeasurement:
    return _NoOpMeasurement(_DOMAIN, _METRIC, _MEASURE, label)


# --- Tests --------------------------------------------------------------------


def test_leaf_transformation():
    """A leaf renders as just its class name."""
    assert Identity(_METRIC, _DOMAIN).format() == "Identity"


def test_leaf_with_attrs():
    """A leaf includes its public, non-excluded attrs inline."""
    assert _t("a").format() == "_TaggedTransformation tag='a'"


def test_two_element_chain():
    """A simple chain opens with ``┌`` and closes with ``└``."""
    chain = ChainTT(_t("a"), _t("b"))
    assert chain.format() == textwrap.dedent(
        """\
        ┌ _TaggedTransformation tag='a'
        └ _TaggedTransformation tag='b'"""
    )


def test_left_nested_chain_tt_flattens():
    """ChainTT(ChainTT(a, b), c) flattens correctly."""
    chain = (_t("a") | _t("b")) | _t("c")
    assert chain.format() == textwrap.dedent(
        """\
        ┌ _TaggedTransformation tag='a'
        ├ _TaggedTransformation tag='b'
        └ _TaggedTransformation tag='c'"""
    )


def test_right_nested_chain_tt_flattens():
    """ChainTT(a, ChainTT(b, c)) flattens correctly."""
    chain = _t("a") | (_t("b") | _t("c"))
    assert chain.format() == textwrap.dedent(
        """\
        ┌ _TaggedTransformation tag='a'
        ├ _TaggedTransformation tag='b'
        └ _TaggedTransformation tag='c'"""
    )


def test_left_nested_chain_tm_flattens():
    """ChainTM(ChainTT(a, b), c) flattens correctly."""
    chain = (_t("a") | _t("b")) | _m("c")
    assert chain.format() == textwrap.dedent(
        """\
        ┌ _TaggedTransformation tag='a'
        ├ _TaggedTransformation tag='b'
        └ _NoOpMeasurement label='c'"""
    )


def test_right_nested_chain_tm_flattens():
    """ChainTM(a, ChainTM(b, c)) flattens correctly."""
    chain = _t("a") | (_t("b") | _m("c"))
    assert chain.format() == textwrap.dedent(
        """\
        ┌ _TaggedTransformation tag='a'
        ├ _TaggedTransformation tag='b'
        └ _NoOpMeasurement label='c'"""
    )


def test_non_chain_container_uses_plain_indent():
    """A container that isn't a Chain* uses plain indented children."""
    wrapped = _MeasurementWrapper(_m("inner"), lambda x: x)

    f_qualname = "test_non_chain_container_uses_plain_indent.<locals>.<lambda>"
    assert wrapped.format() == textwrap.dedent(
        f"""\
        _MeasurementWrapper f=<function {f_qualname}>
          _NoOpMeasurement label='inner'"""
    )


def test_chain_inside_non_chain_container():
    """Chains nested in non-chain containers renders markers at correct level."""
    inner = _t("a") | _t("b") | _m("count")
    wrapped = _MeasurementWrapper(inner, lambda x: x)

    f_qualname = "test_chain_inside_non_chain_container.<locals>.<lambda>"
    assert wrapped.format() == textwrap.dedent(
        f"""\
        _MeasurementWrapper f=<function {f_qualname}>
          ┌ _TaggedTransformation tag='a'
          ├ _TaggedTransformation tag='b'
          └ _NoOpMeasurement label='count'"""
    )


def test_non_chain_child_of_chain_member_uses_plain_indent():
    """When a chain member has a non-chain child, that child is indented +2."""
    wrapped = _MeasurementWrapper(_m("base"), lambda x: x)
    full = ChainTM(_t("a"), wrapped)

    f_qualname = (
        "test_non_chain_child_of_chain_member_uses_plain_indent.<locals>.<lambda>"
    )
    # The wrapped measurement is a plain child of the chain member, so it
    # sits at indent 4 (two past the chain member's label column of 2).
    assert full.format() == textwrap.dedent(
        f"""\
        ┌ _TaggedTransformation tag='a'
        └ _MeasurementWrapper f=<function {f_qualname}>
            _NoOpMeasurement label='base'"""
    )


def test_format_callable_uses_qualname_not_address():
    """Callable attrs print with qualname, not a memory address."""

    def named(x):
        return x

    out = _MeasurementWrapper(_m("x"), named).format()
    assert (
        "f=<function test_format_callable_uses_qualname_not_address.<locals>.named>"
        in out
    )
    assert " at 0x" not in out


def test_multi_child_container_rejected():
    """Multi-child containers cannot be formatted with the default formatter."""

    class _MultiMeasurement(Measurement):
        """A measurement that holds a list of child measurements."""

        def __init__(self, measurements: List[Measurement]):
            first = measurements[0]
            super().__init__(
                input_domain=first.input_domain,
                input_metric=first.input_metric,
                output_measure=first.output_measure,
                is_interactive=False,
            )
            self._measurements = list(measurements)

        @property
        def measurements(self) -> List[Measurement]:
            """The child measurements."""
            return list(self._measurements)

        def __call__(self, data: Any) -> Any:
            return [m(data) for m in self._measurements]

    multi = _MultiMeasurement([_m("a"), _m("b"), _m("c")])
    with pytest.raises(
        NotImplementedError, match="multiple child components cannot be formatted"
    ):
        multi.format()


def test_siblings():
    """Sibling children render with ``* `` markers under the head line."""
    multi = Composition([_m("a"), _m("b"), _m("c")])
    assert multi.format() == textwrap.dedent(
        """\
        Composition
        * _NoOpMeasurement label='a'
        * _NoOpMeasurement label='b'
        * _NoOpMeasurement label='c'"""
    )


def test_siblings_with_chain_child():
    """A sibling whose own format spans multiple lines is indented past the marker."""
    chain_child = _t("a") | _t("b") | _m("c")
    multi = Composition([_m("first"), chain_child])
    assert multi.format() == textwrap.dedent(
        """\
        Composition
        * _NoOpMeasurement label='first'
        * ┌ _TaggedTransformation tag='a'
          ├ _TaggedTransformation tag='b'
          └ _NoOpMeasurement label='c'"""
    )


def test_labeled_siblings_compact():
    """Single-line members are padded so their renderings align in a column."""
    out = format_labeled_siblings([("a", _m("x")), ("longer", _m("y"))])
    assert out == textwrap.dedent(
        """\
        * a:      _NoOpMeasurement label='x'
        * longer: _NoOpMeasurement label='y'"""
    )


def test_labeled_siblings_multiline():
    """A multi-line member places its block below the label, past the marker."""
    wrapped = _MeasurementWrapper(_m("inner"), lambda x: x)
    f_qualname = "test_labeled_siblings_multiline.<locals>.<lambda>"
    out = format_labeled_siblings([("a", wrapped), ("b", _m("leaf"))])
    assert out == textwrap.dedent(
        f"""\
        * a:
          _MeasurementWrapper f=<function {f_qualname}>
            _NoOpMeasurement label='inner'
        * b:
          _NoOpMeasurement label='leaf'"""
    )
