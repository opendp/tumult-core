"""Helpers shared by component formatter functions.

Each transformation/measurement implements its own ``format`` (and a few
related hooks) on the class itself; the helpers here cover shared logic that
doesn't naturally belong on any single component.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    FrozenSet,
    Iterable,
    Iterator,
    Sequence,
)

if TYPE_CHECKING:
    from tmlt.core.measurements.base import Measurement
    from tmlt.core.transformations.base import Transformation


def indent_block(block: str, n: int) -> str:
    """Indent every line of ``block`` by ``n`` spaces."""
    pad = " " * n
    return "\n".join(pad + line for line in block.split("\n"))


def _is_component(value: Any) -> bool:
    """Whether ``value`` is a transformation/measurement."""
    from tmlt.core.measurements.base import Measurement  # noqa: PLC0415
    from tmlt.core.transformations.base import Transformation  # noqa: PLC0415

    return isinstance(value, (Transformation, Measurement))


def _is_component_collection(value: Any) -> bool:
    """Whether ``value`` is a non-empty list/tuple of components."""
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(_is_component(v) for v in value)
    )


def _walk_public_properties(
    component: Any, excluded: Collection[str]
) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, value)`` for each public ``@property`` of the component.

    Walks the MRO base-to-derived so properties on base classes appear before
    subclass-specific ones; within a class, declaration order is
    preserved.

    This function only extracts *properties*, not e.g. dataclass attributes. As
    a result, it only produces useful results on classes where the fields
    appearing in the formatted output are all properties. This is true of
    Transformations and Measurements, but not of e.g. Domains.
    """
    seen: set[str] = set()
    for klass in reversed(inspect.getmro(type(component))):
        for name, descriptor in vars(klass).items():
            if name in seen or name.startswith("_") or name in excluded:
                continue
            if not isinstance(descriptor, property):
                continue
            seen.add(name)
            yield name, getattr(component, name)


def format_value(value: Any) -> str:
    """Format a scalar attribute value for inline display.

    Callables are rendered as ``<function qualname>`` to avoid leaking
    non-deterministic memory addresses into the output.
    """
    if callable(value) and not inspect.isclass(value):
        qualname = getattr(value, "__qualname__", None) or getattr(
            value, "__name__", None
        )
        if qualname:
            return f"<function {qualname}>"
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, list):
        return f"[{', '.join(format_value(v) for v in value)}]"
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({value[0]},)"
        return f"({', '.join(format_value(v) for v in value)})"
    if isinstance(value, (set, frozenset)):
        return f"{{{', '.join(format_value(v) for v in value)}}}"
    return str(value)


def default_format_attrs(
    component: Any, excluded: FrozenSet[str]
) -> list[tuple[str, str]]:
    """Default ``_format_attrs`` implementation, walking public properties.

    Returns ``(name, value_str)`` pairs for every public ``@property`` on the
    component's class whose value is *not* a transformation or measurement
    (those are considered children, not inline attributes). Properties in
    ``excluded`` are skipped.
    """
    out: list[tuple[str, str]] = []
    for name, value in _walk_public_properties(component, excluded):
        if _is_component(value) or _is_component_collection(value):
            continue
        out.append((name, format_value(value)))
    return out


def get_child(
    component: Transformation | Measurement,
) -> Transformation | Measurement | None:
    """Get any Transformation-/Measurement-valued property of a component.

    Returns a nested transformation/measurement found on a public ``@property``
    of the given component, if any. If more than one is found, raises
    ValueError.
    """
    children: list[Any] = []
    for _, value in _walk_public_properties(component, excluded=frozenset()):
        if _is_component(value):
            children.append(value)
        elif _is_component_collection(value):
            children.extend(value)

    if len(children) > 1:
        raise ValueError("Component has more than one child")
    if len(children) == 1:
        return children[0]
    return None


def default_format_children(component: Transformation | Measurement) -> str:
    """Format a component's children.

    Chain children are returned flush (no indent) so their leading markers
    align with the parent's head column; non-chain children are indented by
    two spaces so they sit visibly below the parent.

    Components with multiple children cannot be formatted with this default
    formatter; passing one in will raise :exc:`NotImplementedError`.
    """
    try:
        child = get_child(component)
    except ValueError as e:
        # Child ordering is important, but this function doesn't have any way of
        # knowing what the right ordering is. So, reject components that have
        # multiple child components and make them define custom formatters.
        raise NotImplementedError(
            "Components with multiple child components cannot be formatted by "
            "the default formatter"
        ) from e

    if not child:
        return ""

    return indent_block(child.format(), 2)


def get_chain_children(
    component: Transformation | Measurement,
) -> list[Transformation | Measurement]:
    """Produce a flat list of the chained children of a component."""
    from tmlt.core.measurements.chaining import ChainTM  # noqa: PLC0415
    from tmlt.core.transformations.chaining import ChainTT  # noqa: PLC0415

    if isinstance(component, ChainTT):
        return get_chain_children(component.transformation1) + get_chain_children(
            component.transformation2
        )
    if isinstance(component, ChainTM):
        return get_chain_children(component.transformation) + get_chain_children(
            component.measurement
        )
    return [component]


def _marked_block(block: str, marker: str, body_marker: str | None = None) -> str:
    """Prefix ``block``'s first line with ``marker``, the rest with ``body_marker``.

    When ``body_marker`` is None, body lines are indented by spaces matching
    the width of ``marker`` so they align past it.
    """
    if body_marker is None:
        body_marker = " " * len(marker)
    head, *rest = block.split("\n")
    return "\n".join([marker + head] + [body_marker + line for line in rest])


def format_chain(components: Sequence[Transformation | Measurement]) -> str:
    """Render a flattened chain as a multi-line block with the first line at col 0.

    Each member's own ``format`` output is included in full. The first head
    line opens the box with ``"┌ "``, intermediate head lines are ticked with
    ``"├ "``, and the last head line closes the box with ``"└ "``. Body lines
    of each non-last member are prefixed with ``"│ "`` so a continuous vertical
    rule runs from the opening corner to the closing one; the last member's
    body lines sit below the closing corner with plain two-space indent.
    """
    n = len(components)
    blocks = []
    for i, component in enumerate(components):
        if i == 0:
            marker, body_marker = "┌ ", "│ "
        elif i == n - 1:
            marker, body_marker = "└ ", "  "
        else:
            marker, body_marker = "├ ", "│ "
        blocks.append(_marked_block(component.format(), marker, body_marker))
    return "\n".join(blocks)


def format_siblings(components: Sequence[Transformation | Measurement]) -> str:
    """Render a list of sibling children.

    Used by components like ``Composition`` and ``ParallelComposition`` whose
    children are run independently rather than composed into a chain. Each
    member's first line is prefixed with ``"* "``; subsequent lines are
    indented by two spaces to align past the marker.
    """
    return "\n".join(
        _marked_block(component.format(), "* ") for component in components
    )


def format_labeled_siblings(
    items: Iterable[tuple[str, Transformation | Measurement]],
) -> str:
    """Render labeled sibling children, e.g. for column-keyed aggregations.

    Like :func:`format_siblings`, but each component carries a ``label`` (such
    as a column name) rendered as ``"label:"`` after the ``"* "`` marker. When
    every component formats to a single line, the labels are padded so that the
    member renderings line up in a column. Otherwise, each member's block is
    placed on the lines below its label, indented two spaces past the marker.
    """
    blocks = [(label, component.format()) for label, component in items]
    if all("\n" not in block for _, block in blocks):
        # +2 leaves at a space between the longest label and its value.
        width = max(len(label) for label, _ in blocks) + 2
        return "\n".join(
            f"* {(label + ':').ljust(width)}{block}" for label, block in blocks
        )
    return "\n".join(
        _marked_block(f"{label}:\n{block}", "* ") for label, block in blocks
    )
