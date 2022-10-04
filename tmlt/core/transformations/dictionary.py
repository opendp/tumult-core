"""Transformations and utilities for manipulating dictionaries.

Note that while most transformations in this module (:class:`~.CreateDictFromValue`,
:class:`~.Subset`, and :class:`~.GetValue`) support the metric :class:`~.AddRemoveKeys`,
:class:`~.AugmentDictTransformation` does not. Because of this, none of the included
derived transformations (such as :func:`create_copy_and_transform_value`) support
:class:`~.AddRemoveKeys`. Instead, use transformations in :mod:`~.add_remove_keys`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Callable, Dict, List, Mapping, Tuple, Union, cast

from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.metrics import (
    AddRemoveKeys,
    DictMetric,
    IfGroupedBy,
    Metric,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.identity import Identity
from tmlt.core.utils.misc import get_nonconflicting_string


class CreateDictFromValue(Transformation):
    """Create a dictionary from an object."""

    @typechecked
    def __init__(
        self,
        input_domain: Domain,
        input_metric: Metric,
        key: Any,
        use_add_remove_keys: bool = False,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input objects.
            input_metric: Distance metric on input objects.
            key: Key for constructing dictionary with given object.
            use_add_remove_keys: Whether to use :class:`~.AddRemoveKeys` as the output
                metric instead of :class:`~.DictMetric`.
        """
        output_metric: Union[DictMetric, AddRemoveKeys]
        if use_add_remove_keys:
            if not (
                isinstance(input_metric, IfGroupedBy)
                and isinstance(input_metric.inner_metric, SymmetricDifference)
            ):
                raise ValueError(
                    "Input metric must be IfGroupedBy with an inner metric of "
                    "SymmetricDifference to use AddRemoveKeys as the output metric"
                )
            output_metric = AddRemoveKeys({key: input_metric.column})
        else:
            output_metric = DictMetric({key: input_metric})
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=DictDomain({key: input_domain}),
            output_metric=output_metric,
        )
        self._key = key

    @property
    def key(self) -> Any:
        """Returns the key for the created dictionary."""
        return self._key

    @typechecked
    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        The returned d_out is {self.key: d_in}.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if isinstance(self.output_metric, DictMetric):
            return {self.key: d_in}
        else:
            return d_in

    def __call__(self, val: Any) -> Dict[Any, Any]:
        """Returns dictionary with value associated with specified key."""
        return {self.key: val}


class AugmentDictTransformation(Transformation):
    """Applies transformation to a dictionary and appends the output to the input."""

    @typechecked
    def __init__(self, transformation: Transformation):
        """Constructor.

        Args:
            transformation: Transformation to be applied to input dictionary.
        """
        if not isinstance(transformation.input_domain, DictDomain):
            raise ValueError(
                "Invalid transformation input domain: Must be a DictDomain."
            )
        if not isinstance(transformation.output_domain, DictDomain):
            raise ValueError(
                "Invalid transformation output domain: Must be a DictDomain"
            )

        assert isinstance(transformation.input_metric, DictMetric)
        assert isinstance(transformation.output_metric, DictMetric)

        overlapping_keys = set(transformation.output_domain.key_to_domain) & set(
            transformation.input_domain.key_to_domain
        )
        if overlapping_keys:
            raise ValueError(
                "Invalid transformation output domain. Contains overlapping keys:"
                f" {overlapping_keys}"
            )

        d: Dict[Union[str, Tuple], Domain] = {
            **transformation.input_domain.key_to_domain,
            **transformation.output_domain.key_to_domain,
        }
        output_domain = DictDomain(d)
        output_metric = DictMetric(
            {
                **transformation.input_metric.key_to_metric,
                **transformation.output_metric.key_to_metric,
            }
        )
        super().__init__(
            input_domain=transformation.input_domain,
            input_metric=transformation.input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )
        self._inner_transformation = transformation

    @property
    def inner_transformation(self) -> Transformation:
        """Returns the inner transformation."""
        return self._inner_transformation

    @typechecked
    def stability_function(self, d_in: Dict[Any, Any]) -> Dict[Any, Any]:
        r"""Returns the smallest d_out satisfied.

        Returns {\*\*d_in, \*\*self.transformation.stability_function(d_in)}.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If self.inner_transformation.stability_function(d_in)
                raises :class:`NotImplementedError`.
        """
        self.input_metric.validate(d_in)
        d_out = self.inner_transformation.stability_function(d_in)
        return {**d_in, **d_out}

    @typechecked
    def stability_relation(self, d_in: Any, d_out: Any) -> bool:
        """Returns True if close inputs produce close outputs.

        Returns True if both of the following are true:

        * d_in[key] <= d_out[key] for all augmented keys.
        * self.inner_transformation.stability_relation(d_in, original_d_out)

        where original_d_out is the subset of d_out excluding the augmented keys.

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_metric.
        """
        try:
            return super().stability_relation(d_in, d_out)
        except NotImplementedError:
            pass
        original_d_out = {
            k: d_out[k]
            for k in cast(
                DictMetric, self.inner_transformation.output_metric
            ).key_to_metric
        }
        return all(
            cast(DictMetric, self.input_metric)[k].compare(d_in_k, d_out[k])
            for k, d_in_k in d_in.items()
        ) and self.inner_transformation.stability_relation(d_in, original_d_out)

    def __call__(self, input_dict: Dict[Any, Any]) -> Dict[Any, Any]:
        """Applies transformation on given key to produce augmented dictionary."""
        return {**input_dict, **self.inner_transformation(input_dict)}


class Subset(Transformation):
    """Retrieve a subset of a dictionary by keys."""

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: Union[DictMetric, AddRemoveKeys],
        keys: List[Any],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input dictionaries.
            input_metric: Distance metric over input dictionaries.
            keys: Keys to be used for extracting subset.
        """
        if not keys:
            raise ValueError("No keys provided.")
        if not set(keys) <= set(input_domain.key_to_domain):
            raise ValueError(
                "Can not retrieve subset from dictionary. Invalid keys: "
                f"{set(keys) - set(input_domain.key_to_domain)}"
            )
        output_metric: Union[DictMetric, AddRemoveKeys]
        if isinstance(input_metric, DictMetric):
            if set(input_domain.key_to_domain) != set(input_metric.key_to_metric):
                raise ValueError(
                    "Input metric invalid for input domain: Expected keys: "
                    f"{set(input_domain.key_to_domain)}, not: "
                    f"{set(input_metric.key_to_metric)}."
                )
            output_metric = DictMetric({k: input_metric[k] for k in keys})
        else:
            output_metric = AddRemoveKeys(
                {k: input_metric.df_to_key_column[k] for k in keys}
            )
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=DictDomain({k: input_domain[k] for k in keys}),
            output_metric=output_metric,
        )
        self._keys = keys.copy()

    @property
    def keys(self) -> List[Any]:
        """Returns the keys to keep."""
        return self._keys.copy()

    @typechecked
    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        The returned d_out is {key: d_in[key] for key in self.keys}.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if isinstance(self.input_metric, DictMetric):
            return {key: d_in[key] for key in self.keys}
        return d_in

    def __call__(self, input_dict: Any) -> Any:
        """Returns subset of dictionary specified by keys."""
        return {k: input_dict[k] for k in self.keys}


class GetValue(Transformation):
    """Retrieve an object from a dictionary."""

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: Union[DictMetric, AddRemoveKeys],
        key: Any,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input dictionaries.
            input_metric: Distance metric for input dictionaries.
            key: Key for retrieval.
        """
        if key not in input_domain.key_to_domain:
            raise KeyError(f"{repr(key)} is not one of the input domain's keys")
        # Below is the check in base class, but needs to happen before so
        # output_metric = input_metric[key] won't get a KeyError
        if not input_metric.supports_domain(input_domain):
            raise ValueError(
                f"Input metric {input_metric} and input domain {input_domain} are not"
                " compatible."
            )
        output_metric: Metric
        if isinstance(input_metric, DictMetric):
            output_metric = input_metric[key]
        else:
            output_metric = IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            )

        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=input_domain[key],
            output_metric=output_metric,
        )
        self._key = key

    @property
    def key(self) -> List[Any]:
        """Returns the key to keep."""
        return self._key

    @typechecked
    def stability_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the transformation.

        The returned d_out is d_in[self.key].

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if isinstance(self.input_metric, DictMetric):
            return d_in[self.key]
        else:
            return d_in

    def __call__(self, input_dict: Any) -> Any:
        """Returns value for specified key."""
        return input_dict[self.key]


def create_copy_and_transform_value(
    input_domain: DictDomain,
    input_metric: DictMetric,
    key: Any,
    new_key: Any,
    transformation: Any,
    hint: Callable[[Any, Any], Any],
) -> Transformation:
    """Returns a transformation that transforms and re-adds a value in the input dict.

    The returned transformation has roughly the same behavior as

    .. code-block:: python

        def copy_and_transform_value(data):
            data[new_key] = transformation(data[key])
            return data

    The input is a dictionary, a single value is transformed and added to
    the dictionary at a new key. Note that the original value is left unchanged in the
    dictionary.

    Args:
        input_domain: The domain for the input data.
        input_metric: The metric for the input data.
        key: The key containing the data to transform.
        new_key: The key to store the transformed data.
        transformation: The transformation to apply.
        hint: A hint for the transformation.
    """
    # High level algorithm:
    # 1. Grab the element to transform from the dictionary
    # 2. Create a one element dictionary from the element
    # 3. Augment around (1 + 2)
    if key not in input_domain.key_to_domain:
        raise ValueError("key is not in the domain")
    if new_key in input_domain.key_to_domain:
        raise ValueError("new_key is already in the domain")
    copy_and_transform_value = AugmentDictTransformation(
        ChainTT(
            transformation1=ChainTT(
                transformation1=GetValue(
                    input_domain=input_domain, input_metric=input_metric, key=key
                ),
                transformation2=transformation,
                hint=lambda d_in, _: d_in[key],
            ),
            transformation2=CreateDictFromValue(
                input_domain=transformation.output_domain,
                input_metric=transformation.output_metric,
                key=new_key,
            ),
            hint=lambda d_in, d_out: hint(d_in[key], d_out),
        )
    )
    assert copy_and_transform_value.input_domain == input_domain
    assert copy_and_transform_value.input_metric == input_metric
    assert copy_and_transform_value.output_domain == DictDomain(
        {**input_domain.key_to_domain, **{new_key: transformation.output_domain}}
    )
    assert copy_and_transform_value.output_metric == DictMetric(
        {**input_metric.key_to_metric, **{new_key: transformation.output_metric}}
    )
    return copy_and_transform_value


def create_rename(
    input_domain: DictDomain, input_metric: DictMetric, key: Any, new_key: Any
) -> Transformation:
    """Returns a transformation that renames a single key.

    The returned transformation has roughly the same behavior as

    .. code-block:: python

        def rename(data):
            data[new_key] = data.pop(key)
            return data

    Args:
        input_domain: The domain for the input data.
        input_metric: The metric for the input data.
        key: The original key.
        new_key: The new key.
    """
    # High level algorithm:
    # 1. Copy the element to the new key
    # 2. Use subset to drop the original key
    if key not in input_domain.key_to_domain:
        raise ValueError("key is not in the domain")
    if new_key in input_domain.key_to_domain:
        raise ValueError("new_key is already in the domain")
    copy_and_transform_value = create_copy_and_transform_value(
        input_domain=input_domain,
        input_metric=input_metric,
        key=key,
        new_key=new_key,
        transformation=Identity(domain=input_domain[key], metric=input_metric[key]),
        hint=lambda d_in, _: d_in,
    )
    subset_keys = list(input_domain.key_to_domain) + [new_key]
    subset_keys.remove(key)
    rename = ChainTT(
        transformation1=copy_and_transform_value,
        transformation2=Subset(
            input_domain=cast(DictDomain, copy_and_transform_value.output_domain),
            input_metric=cast(DictMetric, copy_and_transform_value.output_metric),
            keys=subset_keys,
        ),
        hint=lambda d_in, d_out: {**d_in, **{new_key: d_in[key]}},
    )
    assert rename.input_domain == input_domain
    assert rename.input_metric == input_metric
    assert rename.output_domain == DictDomain(
        {
            other_key if other_key != key else new_key: input_domain[other_key]
            for other_key in input_domain.key_to_domain
        }
    )
    assert rename.output_metric == DictMetric(
        {
            other_key if other_key != key else new_key: input_metric[other_key]
            for other_key in input_metric.key_to_metric
        }
    )
    return rename


def create_apply_dict_of_transformations(
    transformation_dict: Mapping[Any, Transformation],
    hint_dict: Mapping[Any, Callable[[Any, Any], Any]],
) -> Transformation:
    """Returns a transformation that applies all given transformations to the input.

    The returned transformation has roughly the same behavior as

    .. code-block:: python

        def apply_dict_of_transformations(data):
            return {
                key: transformation(data)
                for key, transformation in transformation_dict.items()
            }

    The input is a single element, and the output is a dictionary where
    each value is the input transformed by the corresponding transformation.

    Args:
        transformation_dict: A dictionary of transformations with matching input domains
            and input metrics.
        hint_dict: A dictionary of hints for the corresponding transformations in
            transformation_dict.
    """
    # High level algorithm:
    # 1. Create a dictionary with one element and a temporary key
    # 2. For each key, transformation,
    #       add that key + output of transformation to the dictionary
    # 3. Remove the temporary key from the dictionary
    if not transformation_dict:
        raise ValueError("transformation_dict cannot be empty")
    if not set(transformation_dict) == set(hint_dict):
        raise ValueError("transformation_dict and hint_dict must have the same keys")
    some_transformation = next(iter(transformation_dict.values()))
    input_domain = some_transformation.input_domain
    input_metric = some_transformation.input_metric
    if not all(
        input_domain == transformation.input_domain
        for transformation in transformation_dict.values()
    ):
        raise ValueError("Transformations do not have matching input domains")
    if not all(
        input_metric == transformation.input_metric
        for transformation in transformation_dict.values()
    ):
        raise ValueError("Transformations do not have matching input metrics")
    if list(transformation_dict) != list(hint_dict):
        raise ValueError("transformation_dict and hint_dict do not have matching keys")

    base_key = get_nonconflicting_string(
        [key for key in transformation_dict if isinstance(key, str)]
    )

    def create_intermediate_transformation_hint(
        intermediate_transformation: Transformation,
    ) -> Callable[[Any, Any], Dict[Any, Any]]:
        """Return a hint for the intermediate_transformation with the given keys."""
        return lambda d_in, d_out: {
            key: (d_in if key == base_key else hint_dict[key](d_in, d_out))
            for key in cast(
                DictDomain, intermediate_transformation.output_domain
            ).key_to_domain
        }

    intermediate_transformation: Transformation = CreateDictFromValue(
        input_domain=input_domain, input_metric=input_metric, key=base_key
    )
    for key, transformation in transformation_dict.items():
        intermediate_transformation = ChainTT(
            intermediate_transformation,
            create_copy_and_transform_value(
                input_domain=cast(
                    DictDomain, intermediate_transformation.output_domain
                ),
                input_metric=cast(
                    DictMetric, intermediate_transformation.output_metric
                ),
                key=base_key,
                new_key=key,
                transformation=transformation,
                hint=hint_dict[key],
            ),
            hint=create_intermediate_transformation_hint(intermediate_transformation),
        )
    cleanup = Subset(
        input_domain=cast(DictDomain, intermediate_transformation.output_domain),
        input_metric=cast(DictMetric, intermediate_transformation.output_metric),
        keys=list(transformation_dict.keys()),
    )
    apply_dict_of_transformations = ChainTT(
        intermediate_transformation,
        cleanup,
        hint=create_intermediate_transformation_hint(intermediate_transformation),
    )
    assert apply_dict_of_transformations.input_domain == input_domain
    assert apply_dict_of_transformations.input_metric == input_metric
    assert apply_dict_of_transformations.output_domain == DictDomain(
        {
            key: transformation.output_domain
            for key, transformation in transformation_dict.items()
        }
    )
    assert apply_dict_of_transformations.output_metric == DictMetric(
        {
            key: transformation.output_metric
            for key, transformation in transformation_dict.items()
        }
    )
    return apply_dict_of_transformations


def create_transform_value(
    input_domain: DictDomain,
    input_metric: DictMetric,
    key: Any,
    transformation: Any,
    hint: Callable[[Any, Any], Any],
) -> Transformation:
    """Returns a transformation that transforms a single value in the input dict.

    The returned transformation has roughly the same behavior as

    .. code-block:: python

        def transform_value(data):
            data[key] = transformation(data[key])
            return data

    Notice that the input is a dictionary, a single value is transformed, while the
    other values are left unchanged.

    Args:
        input_domain: The domain for the input data.
        input_metric: The metric for the input data.
        key: The key to transform.
        transformation: The transformation to apply.
        hint: A hint for the transformation.
    """
    # High level algorithm:
    # 1. Transform the element and store in at temporary key
    # 2. Remove the original key
    # 3. Rename from the temporary key to the original key
    if key not in input_domain.key_to_domain:
        raise ValueError("key is not in the domain")
    temporary_key = get_nonconflicting_string(
        [
            other_key
            for other_key in input_domain.key_to_domain
            if isinstance(other_key, str)
        ]
    )
    copy_and_transform_value = create_copy_and_transform_value(
        input_domain=input_domain,
        input_metric=input_metric,
        key=key,
        new_key=temporary_key,
        transformation=transformation,
        hint=hint,
    )
    subset_keys = list(
        cast(DictDomain, copy_and_transform_value.output_domain).key_to_domain
    )
    subset_keys.remove(key)
    drop_original_key = Subset(
        input_domain=cast(DictDomain, copy_and_transform_value.output_domain),
        input_metric=cast(DictMetric, copy_and_transform_value.output_metric),
        keys=subset_keys,
    )
    transform_value = ChainTT(
        transformation1=ChainTT(
            transformation1=copy_and_transform_value,
            transformation2=drop_original_key,
            hint=lambda d_in, d_out: {
                **d_in,
                **{temporary_key: hint(d_in[key], d_out[temporary_key])},
            },
        ),
        transformation2=create_rename(
            input_domain=cast(DictDomain, drop_original_key.output_domain),
            input_metric=cast(DictMetric, drop_original_key.output_metric),
            key=temporary_key,
            new_key=key,
        ),
        hint=lambda d_in, d_out: {
            **{
                other_key: d_in[other_key]
                for other_key in input_domain.key_to_domain
                if other_key != key
            },
            **{temporary_key: hint(d_in[key], d_out[key])},
        },
    )
    assert transform_value.input_domain == input_domain
    assert transform_value.input_metric == input_metric
    assert transform_value.output_domain == DictDomain(
        {
            other_key: input_domain[other_key]
            if other_key != key
            else transformation.output_domain
            for other_key in input_domain.key_to_domain
        }
    )
    assert transform_value.output_metric == DictMetric(
        {
            other_key: input_metric[other_key]
            if other_key != key
            else transformation.output_metric
            for other_key in input_metric.key_to_metric
        }
    )
    return transform_value


def create_transform_all_values(
    transformation_dict: Mapping[Any, Transformation],
    hint_dict: Mapping[Any, Callable[[Any, Any], Any]],
) -> Transformation:
    """Returns a transformation that transforms every value in the input dict.

    The returned transformation has roughly the same behavior as

    .. code-block:: python

        def transform_all_values(data):
            return {
                key: transformation(data[key])
                for key, transformation in transformation_dict.items()
            }

    Notice that the input is a dictionary, and each value in the dictionary is
    transformed by the corresponding transformation.

    Args:
        transformation_dict: A dictionary of transformations with matching input domains
            and input metrics.
        hint_dict: A dictionary of hints for the corresponding transformations in
            transformation_dict.
    """
    # High level algorithm:
    # 1. For each key, transformation
    #       transform that key in place
    if not transformation_dict:
        raise ValueError("transformation_dict cannot be empty")
    if not set(transformation_dict) == set(hint_dict):
        raise ValueError("transformation_dict and hint_dict must have the same keys")
    transform_all_values: Transformation = Identity(
        domain=DictDomain(
            {
                key: transformation.input_domain
                for key, transformation in transformation_dict.items()
            }
        ),
        metric=DictMetric(
            {
                key: transformation.input_metric
                for key, transformation in transformation_dict.items()
            }
        ),
    )
    transformed_keys: List[Any] = []
    for key, transformation in transformation_dict.items():
        transform_all_values = transform_all_values | create_transform_value(
            cast(DictDomain, transform_all_values.output_domain),
            cast(DictMetric, transform_all_values.output_metric),
            key=key,
            transformation=transformation,
            hint=hint_dict[key],
        )
        transformed_keys.append(key)
    assert transformed_keys == list(transformation_dict)
    assert transform_all_values.input_domain == DictDomain(
        {
            key: transformation.input_domain
            for key, transformation in transformation_dict.items()
        }
    )
    assert transform_all_values.input_metric == DictMetric(
        {
            key: transformation.input_metric
            for key, transformation in transformation_dict.items()
        }
    )
    assert transform_all_values.output_domain == DictDomain(
        {
            key: transformation.output_domain
            for key, transformation in transformation_dict.items()
        }
    )
    assert transform_all_values.output_metric == DictMetric(
        {
            key: transformation.output_metric
            for key, transformation in transformation_dict.items()
        }
    )
    return transform_all_values
