"""Tests for named, derived transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright the Tumult Core Contributors

from test.conftest import create_mock_transformation

import pytest
from typeguard import TypeCheckError

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.exceptions import InvalidComponentError
from tmlt.core.metrics import AbsoluteDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.derived import DerivedTransformation
from tmlt.core.transformations.identity import Identity


class NamedTransformation(DerivedTransformation):
    """A wrapper with mutable metadata and potentially colliding private fields."""

    def __init__(self, wrapped: Transformation, labels: list[str]) -> None:
        """Initialize the fixed implementation and metadata."""
        super().__init__(wrapped)
        self._labels = labels
        self.__wrapped_transformation = None
        self.__input_domain = None  # type: ignore[assignment]

    @property
    def labels(self) -> list[str]:
        """Return descriptive metadata."""
        return self._labels


def test_properties_and_execution():
    """Properties and execution come from the wrapped transformation."""
    wrapped = create_mock_transformation()
    derived = NamedTransformation(wrapped, ["original"])

    assert derived.wrapped_transformation is wrapped
    assert derived.input_domain is wrapped.input_domain
    assert derived.input_metric is wrapped.input_metric
    assert derived.output_domain is wrapped.output_domain
    assert derived.output_metric is wrapped.output_metric

    data = object()
    assert derived(data) is wrapped.return_value
    assert wrapped.call_count == 1
    assert wrapped.call_args.args == (data,)
    assert wrapped.call_args.kwargs == {}

    with pytest.raises(AttributeError):
        derived.wrapped_transformation = wrapped  # type: ignore[misc]


def test_stability_delegation():
    """Functions and relations are completely delegated, including exceptions."""
    wrapped = create_mock_transformation(stability_function_implemented=True)
    derived = NamedTransformation(wrapped, ["original"])

    assert derived.stability_function(2) is wrapped.stability_function.return_value
    wrapped.stability_function.assert_called_once_with(2)

    wrapped.stability_function.side_effect = NotImplementedError("relation only")
    with pytest.raises(NotImplementedError, match="relation only"):
        derived.stability_function(2)

    for result in (True, False):
        wrapped.stability_relation.return_value = result
        assert derived.stability_relation(2, 3) is result
        wrapped.stability_relation.assert_called_with(2, 3)


def test_metadata_and_formatting():
    """Changing metadata does not replace the implementation or expand it."""
    wrapped = create_mock_transformation(stability_function_implemented=True)
    derived = NamedTransformation(wrapped, ["original"])

    derived.labels.append("updated")
    assert derived.wrapped_transformation is wrapped
    assert derived.stability_function(1) is wrapped.stability_function.return_value
    assert derived.format() == "NamedTransformation labels=['original', 'updated']"
    wrapped.format.assert_not_called()


def test_abstract_base():
    """The base class requires subclasses to supply a constructor."""
    with pytest.raises(TypeError, match="abstract"):
        DerivedTransformation(None)  # type: ignore[abstract, arg-type]


def test_invalid_wrapped_type():
    """Reject objects that are not transformations."""
    with pytest.raises(TypeCheckError):
        NamedTransformation(object(), [])  # type: ignore[arg-type]


def test_missing_initialization():
    """A broken subclass constructor identifies the invalid component."""

    class Uninitialized(DerivedTransformation):
        def __init__(self) -> None:
            pass

    with pytest.raises(InvalidComponentError) as exc_info:
        Uninitialized().wrapped_transformation
    assert exc_info.value.component is Uninitialized
    assert str(exc_info.value) == (
        "Uninitialized.__init__() must call super().__init__(wrapped_transformation)"
    )


def test_chaining():
    """A derived component participates in a normal transformation chain."""
    identity = Identity(AbsoluteDifference(), NumpyIntegerDomain())
    derived = NamedTransformation(identity | identity, ["identity"])
    chain = derived | identity

    data = object()
    assert chain(data) is data
    assert chain.stability_function(2) == 2
    assert chain.format() == "┌ NamedTransformation labels=['identity']\n└ Identity"
