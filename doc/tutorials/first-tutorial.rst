Simple Data Analysis with Tumult Core
=====================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2022

In this tutorial you will learn how to:

-  compute a basic query on a dataset, and
-  observe the privacy properties of this computation.

In this tutorial, we show how to count the number of records in a dataset whose
age is greater than 18. Tumult Core can handle multiple types of data, but at present
it primarily uses Spark DataFrames. Before we do anything, we need to create a spark
session and read in the data:

.. _Java 11 configuration example:

.. testcode::

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

This creates an Core-ready Spark Session. For more details on using Spark sessions with Core, or to troubleshoot, see the :ref:`Spark Topic Guide<Spark>`.

.. testcode::

   from pyspark.sql.types import *

   spark_schema = StructType(
       [StructField("Name", StringType()), StructField("Age", IntegerType())]
   )
   df = spark.createDataFrame([("Alice", 30), ("Bob", 15), ("Carlos", 50)], schema=spark_schema)

We build queries out of basic *transformations* and *measurements*.
Transformations are functions that transform the data, but are not
private on their own. Measurements are randomized mechanisms with
privacy properties. We can build complex measurements by combining
transformations with simple measurements. Additionally we can combine
transformations to produce more complex transformations.

To compute the number of records with age over 18, we combine the
following 3 components:

#. (Transformation) Filter out records with age < 18.
#. (Transformation) Count the total number of records.
#. (Measurement) Add noise to the count to produce a noisy count.

Note that while the noise added by the measurement is necessary to
guarantee privacy, the transformations also have properties that are
tracked and contribute to the privacy guarantee. For this reason,
constructing the entire analysis task from transformations and
measurements is needed for the privacy guarantee to hold (rather than,
e.g., performing steps (1) and (2) in pure Spark).

We begin by constructing the full measurement (steps 1-3 above), running
the measurement on our data, and printing the privacy guarantee of the
measurement. Next, we will walk through and explain each step in this
process.

.. testcode::

   from tmlt.core.transformations.spark_transformations.filter import Filter
   from tmlt.core.transformations.spark_transformations.agg import Count
   from tmlt.core.domains.spark_domains import convert_spark_schema, SparkDataFrameDomain
   from tmlt.core.metrics import SymmetricDifference
   from tmlt.core.measurements.noise_mechanisms import AddGeometricNoise
   from tmlt.core.utils.misc import print_sdf

   tumult_schema = SparkDataFrameDomain(convert_spark_schema(spark_schema))
   over_18_measurement = (
       Filter(filter_expr="Age >= 18", domain=tumult_schema, metric=SymmetricDifference())
       | Count(input_domain=tumult_schema, input_metric=SymmetricDifference())
       | AddGeometricNoise(2)
   )
   print("Noisy count of records with age >= 18:")
   print(over_18_measurement(df))
   print("Privacy loss (epsilon):")
   print(over_18_measurement.privacy_function(1))

.. testoutput::

   Noisy count of records with age >= 18:
   5
   Privacy loss (epsilon):
   1/2

.. testoutput::
   :hide:

   Noisy count of records with age >= 18:
   ...
   Privacy loss (epsilon):
   1/2

The first step is to construct the :class:`filter<tmlt.core.transformations.spark_transformations.filter.Filter>` component.

.. testcode::

   tumult_schema = SparkDataFrameDomain(convert_spark_schema(spark_schema))
   filter = Filter(filter_expr="Age >= 18", domain=tumult_schema, metric=SymmetricDifference())

This component also requires a schema, but the format is slightly
different from the Spark schema, so we used a conversion function.

The ``filter`` transformation
created above is a function that can be
run on our Spark DataFrame. The component filters out records with age
less than 18, as well as tracking other properties necessary to ensure
the privacy guarantee holds when we eventually create a measurement.

.. testcode::

   print_sdf(filter(df))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

        Name  Age
   0   Alice   30
   1  Carlos   50

Next, we construct the
:class:`count<tmlt.core.transformations.spark_transformations.agg.Count>` component.

.. testcode::

   count = Count(input_domain=tumult_schema, input_metric=SymmetricDifference())

Like the ``filter`` transformation we constructed above, ``count`` can
be run on the data, and will produce the exact count of records in the
dataset.

.. testcode::

   print(count(df))

.. testoutput::

   3

However, we want to count the number of records in the filtered dataset,
not the original dataset. To do this, we create a new transformation
that performs both the filter and the count. We can combine
transformations into new transformations using the chain operator,
``|``.

.. testcode::

   filter_and_count = filter | count

``filter_and_count`` is a new transformation that chains together the
filter and count transformations, as we can verify below:

.. testcode::

   print(filter_and_count(df))

.. testoutput::

   2

Finally, we create a measurement to :class:`add noise<tmlt.core.measurements.noise_mechanisms.AddGeometricNoise>` in a privacy-preserving
way. The following measurement produces a noisy number by adding
geometric noise with scale ``alpha``.

.. testcode::

   add_noise = AddGeometricNoise(2)

To create a measurement that filters and counts before adding noise, we
chain our previous ``filter_and_count`` transformation with the
``add_noise`` measurement we just created.

.. testcode::

   over_18_measurement = filter_and_count | add_noise

If we apply our ``over_18_measurement`` to our dataset, we see a noisy
count of the number of records with age over 18.

.. testcode::

   print(over_18_measurement(df))

.. testoutput::

   2

.. testoutput::
   :hide:

   ...

This measurement has a privacy guarantee, which is automatically
calculated from properties of its constituent parts. You can see the
privacy guarantee of the measurement using the ``privacy_function``
member.

.. testcode::

   print(over_18_measurement.privacy_function(1))

.. testoutput::

   1/2

The privacy guarantee says, informally, that if you call this function on similar dataframes, you will get statistically similar noisy counts. The ``privacy_function`` quantifies this guarantee precisely. By calling this function with an input of 1, we learn how statistically similar the outputs will be for two dataframes that differ by 1 row. The function return value tells us that the noisy counts satisfy :math:`\epsilon`-differential privacy with :math:`\epsilon = 1/2`.

If we call this function with an input of 2 (dataframes differing by 2 rows), we learn how statistically similar the outputs will be for two dataframes that differ by 2 rows. That is, we learn that the *group privacy* guarantee: the mechanism satisfies :math:`\epsilon`-differential privacy for groups of size 2, with :math:`\epsilon = 1`.

.. testcode::

   print(over_18_measurement.privacy_function(2))

.. testoutput::

   1
