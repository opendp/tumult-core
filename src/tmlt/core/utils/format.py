"""Helpers shared by formatter functions on domains, transformations, and measurements.

Domains, transformations, and measurements all inherit from :class:`Formattable`,
which provides a default multi-line ``format()`` rendering. The helpers here
cover the shared logic that the mixin and its overrides build on.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any, Collection, Iterable, Iterator, Sequence

if TYPE_CHECKING:
    from tmlt.core.measurements.base import Measurement
    from tmlt.core.transformations.base import Transformation


class Formattable:
    """Mixin providing a default multi-line ``format()`` rendering.

    Subclasses can override :attr:`FORMAT_EXCLUDED_ATTRS` to hide attributes, or
    override :meth:`_format_head` / :meth:`_format_children` to customize how
    these sections are displayed. Alternatively, they can override
    :meth:`format` for complete control over rendering.
    """

    FORMAT_EXCLUDED_ATTRS: frozenset[str] = frozenset()
    """Attributes hidden from the formatted output."""

    def format(self) -> str:
        """Return a human-readable multi-line description of this object."""
        head = self._format_head()
        children = self._format_children()
        if not children:
            return head
        return f"{head}\n{children}"

    def _format_head(self) -> str:
        """Render this object's head line: class name followed by its attrs."""
        parts = [type(self).__name__]
        parts.extend(
            f"{name}={value}"
            for name, value in default_format_attrs(self, self.FORMAT_EXCLUDED_ATTRS)
        )
        return " ".join(parts)

    def _format_children(self) -> str:
        """Render the block for nested formattables, or "" if there are none.

        Values with multiple non-excluded children cannot be rendered with
        this default; subclasses with multiple children must override.
        """
        try:
            child = get_child(self, self.FORMAT_EXCLUDED_ATTRS)
        except ValueError as e:
            raise NotImplementedError(
                f"{type(self).__name__} has multiple child components and must "
                "override _format_children() to render them."
            ) from e

        if not child:
            return ""

        return indent_block(child.format(), 2)


def indent_block(block: str, n: int) -> str:
    """Indent every line of ``block`` by ``n`` spaces."""
    pad = " " * n
    return "\n".join(pad + line for line in block.split("\n"))


def _marked_block(block: str, marker: str, body_marker: str | None = None) -> str:
    """Prefix ``block``'s first line with ``marker``, the rest with ``body_marker``.

    When ``body_marker`` is None, body lines are indented by spaces matching
    the width of ``marker`` so they align past it.
    """
    if body_marker is None:
        body_marker = " " * len(marker)
    head, *rest = block.split("\n")
    return "\n".join([marker + head] + [body_marker + line for line in rest])


def _is_formattable_child(value: Any) -> bool:
    """Whether ``value`` should be rendered as a child block rather than inline."""
    return isinstance(value, Formattable)


def _is_formattable_child_collection(value: Any) -> bool:
    """Whether ``value`` is a non-empty list/tuple of child-renderable objects."""
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(_is_formattable_child(v) for v in value)
    )


def _walk_public_attrs(
    value: Any, excluded: Collection[str]
) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, value)`` for the public attributes of ``value``.

    Yields dataclass fields first in declaration order (when ``value`` is a
    dataclass instance), then ``@property`` descriptors walking the MRO
    base-to-derived so base-class properties come before subclass-specific
    ones; within a class, declaration order is preserved. Names already
    yielded are skipped, as are names in ``excluded`` and names starting
    with ``_``.

    This unifies attribute discovery across transformations and measurements
    (which expose state via ``@property``) and domains (some of which are
    dataclasses with state in plain fields).
    """
    # Passing class objects adds edge cases to the below logic, and there is not
    # a current use-case for allowing it.
    if isinstance(value, type):
        raise ValueError("value must not be an instance of type 'type'")

    seen: set[str] = set()
    cls = type(value)
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            name = field.name
            if name in seen or name.startswith("_") or name in excluded:
                continue
            seen.add(name)
            yield name, getattr(value, name)
    for klass in reversed(inspect.getmro(cls)):
        for name, descriptor in vars(klass).items():
            if name in seen or name.startswith("_") or name in excluded:
                continue
            if not isinstance(descriptor, property):
                continue
            seen.add(name)
            yield name, getattr(value, name)


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
        # Sort to guarantee consistent ordering
        return f"{{{', '.join(sorted(format_value(v) for v in value))}}}"
    return str(value)


def default_format_attrs(
    value: Any, excluded: Collection[str]
) -> list[tuple[str, str]]:
    """Default ``(name, value_str)`` pairs for inline attribute rendering.

    Walks public attributes of ``value`` -- dataclass fields and
    ``@property`` descriptors -- and returns each as a formatted pair.
    Attributes whose values are themselves child-renderable (any
    :class:`Formattable`, or non-empty list/tuples thereof) are skipped,
    since those are rendered as nested blocks by
    :meth:`Formattable._format_children`. Names in ``excluded`` are also
    skipped.
    """
    out: list[tuple[str, str]] = []
    for n, v in _walk_public_attrs(value, excluded):
        if _is_formattable_child(v) or _is_formattable_child_collection(v):
            continue
        out.append((n, format_value(v)))
    return out


def get_child(value: Any, excluded: Collection[str]) -> Formattable | None:
    """Return the single child-renderable attribute of ``value``, or None.

    A "child" is any nested :class:`Formattable`. Attributes named in
    ``excluded`` are skipped. If more than one child is found, raises
    :exc:`ValueError`.
    """
    children: list[Any] = []
    for _, v in _walk_public_attrs(value, excluded=excluded):
        if _is_formattable_child(v):
            children.append(v)
        elif _is_formattable_child_collection(v):
            children.extend(v)

    if len(children) > 1:
        raise ValueError("Value has more than one child")
    if len(children) == 1:
        return children[0]
    return None


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


def format_chain(components: Sequence[Formattable]) -> str:
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


def format_siblings(values: Sequence[Formattable]) -> str:
    """Render a list of sibling children.

    Used by components like ``Composition`` and ``ParallelComposition`` whose
    children are run independently rather than composed into a chain. Each
    member's first line is prefixed with ``"* "``; subsequent lines are
    indented by two spaces to align past the marker.
    """
    return "\n".join(_marked_block(v.format(), "* ") for v in values)


def format_labeled_siblings(
    items: Iterable[tuple[str, Formattable]],
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
