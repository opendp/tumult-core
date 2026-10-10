"""Tests for named, derived measurements."""

# SPDX-License-Identifier: Apache-2.0
# Copyright the Tumult Core Contributors

from test.conftest import create_mock_measurement

import pytest
from typeguard import TypeCheckError

from tmlt.core.exceptions import InvalidComponentError
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.derived import DerivedMeasurement


class NamedMeasurement(DerivedMeasurement):
    """A wrapper with mutable metadata and potentially colliding private fields."""

    def __init__(self, wrapped: Measurement, labels: list[str]) -> None:
        """Initialize the fixed implementation and metadata."""
        super().__init__(wrapped)
        self._labels = labels
        self.__wrapped_measurement = None
        self.__input_domain = None  # type: ignore[assignment]

    @property
    def labels(self) -> list[str]:
        """Return descriptive metadata."""
        return self._labels


def test_properties_and_execution():
    """Properties and execution come from the wrapped measurement."""
    wrapped = create_mock_measurement(is_interactive=True)
    derived = NamedMeasurement(wrapped, ["original"])

    assert derived.wrapped_measurement is wrapped
    assert derived.input_domain is wrapped.input_domain
    assert derived.input_metric is wrapped.input_metric
    assert derived.output_measure is wrapped.output_measure
    assert derived.is_interactive is True

    data = object()
    assert derived(data) is wrapped.return_value
    assert wrapped.call_count == 1
    assert wrapped.call_args.args == (data,)
    assert wrapped.call_args.kwargs == {}

    with pytest.raises(AttributeError):
        derived.wrapped_measurement = wrapped  # type: ignore[misc]


def test_privacy_delegation():
    """Functions and relations are completely delegated, including exceptions."""
    wrapped = create_mock_measurement(privacy_function_implemented=True)
    derived = NamedMeasurement(wrapped, ["original"])

    assert derived.privacy_function(2) is wrapped.privacy_function.return_value
    wrapped.privacy_function.assert_called_once_with(2)

    wrapped.privacy_function.side_effect = NotImplementedError("relation only")
    with pytest.raises(NotImplementedError, match="relation only"):
        derived.privacy_function(2)

    for result in (True, False):
        wrapped.privacy_relation.return_value = result
        assert derived.privacy_relation(2, 3) is result
        wrapped.privacy_relation.assert_called_with(2, 3)


def test_metadata_and_formatting():
    """Changing metadata does not replace the implementation or expand it."""
    wrapped = create_mock_measurement(privacy_function_implemented=True)
    derived = NamedMeasurement(wrapped, ["original"])

    derived.labels.append("updated")
    assert derived.wrapped_measurement is wrapped
    assert derived.privacy_function(1) is wrapped.privacy_function.return_value
    assert derived.format() == "NamedMeasurement labels=['original', 'updated']"
    wrapped.format.assert_not_called()


def test_abstract_base():
    """The base class requires subclasses to supply a constructor."""
    with pytest.raises(TypeError, match="abstract"):
        DerivedMeasurement(None)  # type: ignore[abstract, arg-type]


def test_invalid_wrapped_type():
    """Reject objects that are not measurements."""
    with pytest.raises(TypeCheckError):
        NamedMeasurement(object(), [])  # type: ignore[arg-type]


def test_missing_initialization():
    """A broken subclass constructor identifies the invalid component."""

    class Uninitialized(DerivedMeasurement):
        def __init__(self) -> None:
            pass

    with pytest.raises(InvalidComponentError) as exc_info:
        Uninitialized().wrapped_measurement
    assert exc_info.value.component is Uninitialized
    assert str(exc_info.value) == (
        "Uninitialized.__init__() must call super().__init__(wrapped_measurement)"
    )
