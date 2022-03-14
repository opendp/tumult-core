"""Wraps a transformation such that the result is cached."""

# <placeholder: boilerplate>
from abc import ABC, abstractmethod
from typing import Any

from pyspark.sql import DataFrame
from typeguard import typechecked

from tmlt.core.transformations.base import Transformation


class Cache(Transformation):
    """Wraps a transformation such that the result is cached.

    .. warning::

        For the cache to recognize the input, it must have the same python id (be the
        same object). Many transformations do not always output the same object when
        given the same input object. If such a transformation is chained before this
        transformation, this transformation will not work properly. To deal with this,
        consider caching the transformation that comes before too.
    """

    @typechecked
    def __init__(self, transformation: Transformation):
        """Constructor.

        Args:
            transformation: Transformation object to be wrapped.
        """
        super().__init__(
            input_domain=transformation.input_domain,
            input_metric=transformation.input_metric,
            output_domain=transformation.output_domain,
            output_metric=transformation.output_metric,
        )
        self._transformation = transformation

    @property
    def transformation(self) -> Transformation:
        """Returns transformation object being cached."""
        return self._transformation

    @typechecked
    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        Returns self.transformation.stability_function(d_in)

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If self.transformation.stability_function(d_in) raises
                :class:`NotImplementedError`.
        """
        self.input_metric.validate(d_in)
        return self.transformation.stability_function(d_in)

    @typechecked
    def stability_relation(self, d_in: Any, d_out: Any) -> bool:
        """Returns True if close inputs produce close outputs.

        Returns self.transformation.stability_relation(d_in, d_out)

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under input_metric.
        """
        return self.transformation.stability_relation(d_in, d_out)

    def __call__(self, data: Any) -> Any:
        """Return result for transformation.

        If result is in cache (for supplied input), result is returned.
        Otherwise, transformation is performed and result is cached.
        """
        key = id(self)
        try:
            return cache().get(key, data)
        except KeyError:
            output_value = self.transformation(data)
            cache().set(key, data, output_value)
            return output_value

    def __del__(self):
        """Clears any cache result for this transformation."""
        try:
            cache().delete(id(self))
        except KeyError:
            pass


class TransformationCache(ABC):
    """Base class for a cache for transformations."""

    @abstractmethod
    def get(self, key: int, input_value: Any) -> Any:
        """Retrieve result from cache.

        Args:
            key: Unique id for the transformation to identify cache entry.
            input_value: Input to transformation.

        Raises:
            KeyError: If nothing is found.
        """
        ...

    @abstractmethod
    def set(self, key: int, input_value: Any, output_value: Any):
        """Write to cache.

        Args:
            key: Unique id for the transformation to identify cache entry.
            input_value: Input to transformation.
            output_value: Output of transformation.
        """
        ...

    @abstractmethod
    def delete(self, key: int):
        """Delete particular cache entry.

        Args:
            key: Unique id for the transformation to identify cache entry.

        Raises:
            KeyError: If nothing is found.
        """
        ...


class SingleItemCache(TransformationCache):
    """Cache for single calls to transformations.

    For each transformation, this cache only stores the output of the transformation
    for one particular input.
    """

    def __init__(self):
        """Constructor."""
        self._cache = dict()

    def get(self, key: int, input_value: Any) -> Any:
        """Retrieve result from cache.

        Args:
            key: Unique id for the transformation to identify cache entry.
            input_value: Input to transformation.

        Raises:
            KeyError: If nothing is found.
        """
        try:
            cached_input, cached_output = self._cache[key]
            # Requires they are literally the same object, this may be too strict...
            if cached_input is input_value:
                return cached_output
        except KeyError:
            pass
        raise KeyError((key, input_value))

    def set(self, key: int, input_value: Any, output_value: Any):
        """Write to cache.

        Args:
            key: Unique id for the transformation to identify cache entry.
            input_value: Input to transformation.
            output_value: Output of transformation.
        """
        if isinstance(output_value, DataFrame):
            output_value.persist()
        self._cache[key] = (input_value, output_value)

    def delete(self, key: int):
        """Delete particular cache entry.

        Args:
            key: Unique id for the transformation to identify cache entry.

        Raises:
            KeyError: If nothing is found.
        """
        _, output_value = self._cache[key]
        if isinstance(output_value, DataFrame):
            output_value.unpersist()
        del self._cache[key]


_cache = SingleItemCache()


def cache() -> TransformationCache:
    """Getter for cache."""
    return _cache
