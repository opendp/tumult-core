.. _architecture:

Tumult Core Architecture
========================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2022

.. testsetup::

   from tmlt.core.transformations.spark_transformations.filter import Filter
   from tmlt.core.transformations.spark_transformations.agg import Count
   from tmlt.core.domains.spark_domains import convert_spark_schema, SparkDataFrameDomain
   from tmlt.core.metrics import SymmetricDifference
   from tmlt.core.measurements.noise_mechanisms import AddGeometricNoise
   from tmlt.core.transformations.chaining import ChainTT
   from tmlt.core.measurements.chaining import ChainTM
   from tmlt.core.utils.misc import print_sdf
   from pyspark.sql.types import *
   from pyspark.sql import SparkSession
   spark = SparkSession.builder.getOrCreate()

Tumult Core is a collection of composable components for implementing
algorithms to perform differentially private computations. The design of Tumult Core
is based on the design proposed in the `OpenDP White Paper
<https://projects.iq.harvard.edu/files/opendp/files/opendp_programming_framework_11may2020_1_01.pdf>`_.
On this page, we give an overview of this design. Readers who want more
information should refer to the linked white paper.

Transformation and Measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main building blocks of Tumult Core are *transformations* and
*measurements*. In this section we begin by defining a transformation,
then later define a measurement, which shares many of the same concepts.

A transformation is a function that takes an input from some
defined *input domain* and produces an output in some given *output domain*.
Each transformation additionally has an *input metric* -- a function defining
distances [#]_ between elements in the input domain -- and an *output metric* -- a
function defining distances between elements in the output domain.

.. note::
   In Tumult core, subclasses of :class:`~.Transformation` and
   :class:`~.Measurement` are constructors for various types of transformations;
   instances of these classes are the transformations and measurements
   themselves. For example, the :class:`~.FlatMap` class does not define a
   single transformation. It is a constructor that can create many distinct
   transformations, all with different domains, metrics, and stability
   relations.

For example, the following code uses the :class:`~.Filter` transformation
constructor to create a transformation that filters out the records with ages of
at least 18 years. This transformation constructor takes only a single domain
and metric, because the input and output metric/domain must match. Even if some
metric and domain information is inferred by the constructor, we can always
inspect the constructed transformation to see its input domain, input metric,
output domain, and output metric.

.. testcode::

   spark_schema = StructType(
       [StructField("Name", StringType()), StructField("Age", IntegerType())]
   )
   df = spark.createDataFrame([("Alice", 30), ("Bob", 15), ("Carlos", 50)], schema=spark_schema)
   tumult_schema = SparkDataFrameDomain(convert_spark_schema(spark_schema))
   filter_transformation = Filter(filter_expr="Age >= 18", domain=tumult_schema, metric=SymmetricDifference())

   print(filter_transformation.input_domain)
   print(filter_transformation.output_domain)
   print(filter_transformation.input_metric)
   print(filter_transformation.output_metric)

.. testoutput::

   SparkDataFrameDomain(schema={'Name': SparkStringColumnDescriptor(allow_null=True), 'Age': SparkIntegerColumnDescriptor(allow_null=True, size=32)})
   SparkDataFrameDomain(schema={'Name': SparkStringColumnDescriptor(allow_null=True), 'Age': SparkIntegerColumnDescriptor(allow_null=True, size=32)})
   SymmetricDifference()
   SymmetricDifference()

Finally, each transformation has a *stability function* (this statement is not
completely true, see :ref:`function-vs-relation`). This is a function that, given an input
distance ``d_in``, returns an output distance ``d_out`` satisfying the following:

    For any pair of inputs in the input domain whose distance is at most ``d_in``
    in the input metric, the corresponding outputs (the result of the
    transformation applied to each of the two inputs) will have distance at most
    ``d_out`` in the output metric.

Ideally, we want the stability function to give the smallest such ``d_out``
satisfying the statement above, but this is not required. Although this
guarantee on its own is not a privacy guarantee, we will see in the following
section that it allows us to construct complex measurements with provable
privacy guarantees.

.. testcode::

   print(filter_transformation.stability_function(1))

.. testoutput::

   1

Measurements have many things in common with transformations, for example they
both have input domains and input metrics. One key difference is that
measurements produced random outputs, while transformations are deterministic.
Since the output is random, a measurement has an *output measure* rather than an
output metric.

The measure defines a distance between distributions of outputs over the output
domain of the measurement. For example, if :math:`P` and :math:`Q` denote two
distributions over some output domain, the distance between :math:`P` and
:math:`Q` in the :class:`~.PureDP` output measure is :math:`D_{\infty}(X \| Y)`,
where :math:`D_{\infty}(P \| Q) = \sup _{x \in \operatorname{supp} (Q)} \log
\frac{P(x)}{Q(x)}` is the Rényi divergence of infinite order.

.. testcode::

   add_noise = AddGeometricNoise(2)

   print(add_noise.input_domain)
   print(add_noise.input_metric)
   print(add_noise.output_measure)

.. testoutput::

   NumpyIntegerDomain(size=64)
   AbsoluteDifference()
   PureDP()

Like transformations, measurements have a guarantee that relates a distance in
the input metric to a distance in the output measure. We call this guarantee the
*privacy function*, and it works similarly to a stability function. The privacy
function takes an input distance ``d_in`` and returns an output distance
``d_out`` satisfying the following:

    For any pair of inputs in the input domain whose distance is at most
    ``d_in`` in the input metric, the distribution of the corresponding random
    outputs (the result of the measurements applied to each of the two inputs),
    will have distance at most ``d_out`` in the output measure.

Like stability functions, we want the privacy function to give the smallest such
``d_out`` satisfying the statement above in order to get the best possible
privacy guarantee, but this is not required. The privacy function of a
measurement can give us a standard :math:`\epsilon`-differential privacy
guarantee, but is more general and can give other provable guarantees (see
:ref:`privacy-guarantee`). In addition, the privacy function can be used to
build more complex measurements, as we will see in the next section.

.. testcode::

   print(add_noise.privacy_function(1))

.. testoutput::

   1/2

.. _function-vs-relation:

A note on privacy/stability functions and relations
"""""""""""""""""""""""""""""""""""""""""""""""""""

In some cases, privacy and stability functions are not sufficient to capture
the guarantees we want to make about the transformations or measurements. For
example, the *approximate differential privacy* measure has two parameters:
:math:`\epsilon` and :math:`\delta`. Some measurements that use this privacy
measure actually satisfy a continuum of :math:`(\epsilon, \delta)`-differential
privacy guarantees that are not comparable, and it is therefore not possible for
the privacy function to give the best guarantee.

Because of cases like these, we also consider more general versions of the
stability and privacy functions, called *stability relations* and *privacy
relations* respectively. A stability relation is a function that, given *both*
an input distance ``d_in`` and an output distance ``d_out``, returns either
``True`` or ``False``. If the relation returns ``True``, it means that the
transformation satisfies the following:

    For any pair of inputs in the input domain whose distance is at most ``d_in``
    in the input metric, the corresponding outputs (the result of the
    transformation applied to each of the two inputs) will have distance at most
    ``d_out`` in the output metric.

Similarly, a privacy relation takes an input distance ``d_in`` and an output
distance ``d_out``, and returns ``True`` or ``False``. If it returns ``True``,
the measurement satisfies the following:

    For any pair of inputs in the input domain whose distance is at most
    ``d_in`` in the input metric, the distribution of the corresponding random
    outputs (the result of the measurements applied to each of the two inputs),
    will have distance at most ``d_out`` in the output measure.

Every transformation has a stability relation and every measurement has a
privacy relation. If the stability and privacy functions are defined, these
relations are defined using the corresponding functions. For transformations and
measurements for which the stability and privacy functions are insufficient,
only the stability and privacy relations are defined. Transformations and
measurements with only stability and privacy relations have the drawback that
they take extra work to chain (see :ref:`combinators` for an overview on
chaining).

.. testcode::

   print(filter_transformation.stability_relation(1,2))
   print(filter_transformation.stability_function(1))

.. testoutput::

   True
   1

.. testcode::

   print(add_noise.privacy_relation(1,1))
   print(add_noise.privacy_function(1))

.. testoutput::

   True
   1/2


.. _combinators:

Combinators
^^^^^^^^^^^

The power of Tumult Core lies in the ways that we can combine components to
produce larger and more complex components. The first way that we can do this is
by *chaining* components. The :class:`~.ChainTT` component combines two
transformations into a single transformation and :class:`~.ChainTM` combines a
transformation and measurement into a new measurement. These components behave
like function composition, e.g. :class:`~.ChainTT` applies the first
transformation to the input, then passes the output to the second transformation,
then returns the output of the second transformation. Most importantly though,
:class:`~.ChainTT` and :class:`~.ChainTM` have their own stability and privacy
relations (and functions, if the subcomponents have stability/privacy functions)
that are derived automatically [#]_ from their constituent pieces. This is what
allows us to build complex measurements with privacy relations that are
automatically determined.

The following example uses :class:`~.ChainTT` [#]_ to combine our previous ``filter``
transformation with a new ``count`` transformation.

.. testcode::

   count = Count(input_domain=tumult_schema, input_metric=SymmetricDifference())
   filter_and_count = ChainTT(filter_transformation, count)

   print(filter_and_count.input_domain)
   print(filter_and_count.output_domain)
   print(filter_and_count.input_metric)
   print(filter_and_count.output_metric)
   print(filter_and_count.stability_function(1))

.. testoutput::

   SparkDataFrameDomain(schema={'Name': SparkStringColumnDescriptor(allow_null=True), 'Age': SparkIntegerColumnDescriptor(allow_null=True, size=32)})
   NumpyIntegerDomain(size=64)
   SymmetricDifference()
   AbsoluteDifference()
   1

This new transformation can be chained with our previously created measurement
using :class:`~.ChainTM` to create a new measurement.

.. testcode::

   measurement = ChainTM(filter_and_count, add_noise)

   print(measurement.input_domain)
   print(measurement.input_metric)
   print(measurement.output_measure)
   print(measurement.privacy_function(1))

.. testoutput::

   SparkDataFrameDomain(schema={'Name': SparkStringColumnDescriptor(allow_null=True), 'Age': SparkIntegerColumnDescriptor(allow_null=True, size=32)})
   SymmetricDifference()
   PureDP()
   1/2

Additionally, Tumult Core provides components for composing measurements. These
components are specific to particular privacy measures, and leverage the
composition properties of that measure. For example, the *sequential
composition* property of :math:`\epsilon`-differential privacy is leveraged in
the :class:`~.Composition` class.  Another example of a measurement combinator
is :class:`~.ParallelComposition`, which composes measurements that are applied
to a series of datasets with a bound on the contribution of a single user across
all the datasets.  Note that this class is actually a different type of
measurement: one that supports interactivity (discussed in the next section).
Tumult Core could additionally define a measurement for composing measurements
in a non-interactive way, but the interactive versions provide the same
functionality and more.

Interactivity
^^^^^^^^^^^^^

Tumult Core also supports interactivity. Instances of the class
:class:`~.Queryable` (that we refer to as *queryables*) are objects that can
queried interactively. Queryables have some state, including, e.g. the private
data and the remaining privacy budget afforded to the queryable.  Queryables are
not instantiated directly, but rather by evaluating an *interactive measurement*
which gives the privacy guarantee of the queryable. An interactive measurement
(a :class:`~.Measurement` with ``is_interactive`` set to ``True``) is a type of
measurement that produces a queryable, rather than directly producing some
private output.  Unlike non-interactive measurements, the output measure of an
interactive measurement applies to the transcript resulting from an interaction
between a user and the produced queryable (all queries made to the queryable,
along with the responses from the queryable).

Tumult Core supports various types of interactive components, such as *privacy
filters* :cite:`RogersVRU16` (:class:`~.SequentialComposition`).  Like the
non-interactive components of Tumult Core, interactive components are composable
and extensible. For example, queryables can evaluate interactive measurements
that spawn new queryables without knowing anything about the behavior of the
spawned queryable. This allows for rich interactions between the user and the
private data whose privacy properties are derived from the constituent
interactive measurements.

One complexity surrounding interactivity is that the privacy guarantee of the
*concurrent composition* of queryables is not known for some privacy
definitions.  Concurrent composition occurs when queries to the composed
queryables are interleaved (i.e. ask a query of queryable 1, ask a query of
qeuryable 2, then again ask a query of queryable 1). Although differential
privacy supports concurrent composition, other privacy notions such as zCDP
:cite:`BunS16` have not been shown to support concurrent composition (see
:cite:`VadhanT21`). Tumult Core currently maintains a consistent approach to
concurrent composition: it is not permitted regardless of the privacy measure.

The modularity of the Tumult Core design of interactivity, combined with the
restriction on concurrent composition makes interactivity somewhat complicated
to work with.  For this reason, Tumult Core provides the
:class:`~.PrivacyAccountant` interface for working with interactivity that hides
some of this complexity and manages details for the user.

Our model for interactive is based on the *interactive mechanisms* defined by
:cite:`VadhanT21`. Compared to this work, we use slightly different terminology
and roughly split :cite:`VadhanT21`'s interactive mechanism into an
instantiation phase (Tumult Core interactive measurement) and the interaction
phase (Tumult Core queryable).

.. rubric:: Footnotes

.. [#] Although it is useful to think of this function as defining distances, in
       reality these distances need not satisfy the triangle inequality, and do
       not even need to be numbers (e.g. see :class:`~.DictMetric`) -- "distances"
       are any set with a partial ordering.
.. [#] When both the constituent components do not have stability/privacy
       functions defined, it's necessary to provide a *chaining hint* to the
       chaining component in order for it to construct it's stability/privacy
       guarantee. Providing a good hint can be challenging, but since most
       components have stability or privacy functions defined, chaining hints
       are beyond the scope of this article.
.. [#] In these examples we explicitly use :class:`~.ChainTT` and
       :class:`~.ChainTM`. More commonly, we use the operator ``|`` for
       chaining, which automatically selects between the two chaining combinators.
