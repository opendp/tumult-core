.. _special-values:

NaNs, nulls, and infs
=====================

This page describes how Tumult Core handles NaNs, nulls, and infs.

Preliminaries
-------------

:class:`~.SparkDataFrameDomain`\ s are constructed by specifying column constraints using :class:`~.SparkColumnDescriptor`\ s that describe the data type as well as some metadata about what special values are
permitted on each column. In particular, all :class:`~.SparkColumnDescriptor`\ s allow
specifying a flag :code:`allow_null` to indicate if null (:code:`None`) values  are permitted
in a column; additionally, :class:`~.SparkFloatColumnDescriptor` allows specifying
:code:`allow_inf` and :code:`allow_nan` to indicate if a column with floating 
point values can contain (+/-)\ :code:`inf` or :code:`NaN` respectively.

Tumult Core also supports transformations and measurements on Pandas DataFrames and Series using :mod:`~.pandas_domains` and :mod:`~.numpy_domains`. :class:`~.PandasDataFrameDomain`\ s are constructed by specifying a :class:`~.PandasSeriesDomain` for each column, which in turn is specified by a :class:`~.NumpyDomain`. Unlike :class:`~.SparkIntegerColumnDescriptor` and :class:`~.SparkFloatColumnDescriptor`, :class:`~.NumpyIntegerDomain` and :class:`~.NumpyFloatDomain` do not permit :code:`null` values. Pandas domains are only used for Quantiles (discussed `below <#quantile>`_) currently.

Comparison Operators
--------------------

This section summarizes the behavior of comparison operators in Spark when one or both of the operands are special (null, NaN or inf). We will reference these operators to explain how our components handle these values.

Nulls
^^^^^

Comparisons (using :code:`<, >, =, <=, >=`) between a null value and any other value evaluates to null.
Spark's null-safe equality operator :code:`<=>` allows safely comparing potentially null values such that
:code:`X <=> Y` evaluates to True if :code:`X` and :code:`Y` are both non-null values and :code:`X = Y`, or :code:`X` and :code:`Y` are both nulls.

.. note::

    Python's :code:`==` operator is equivalent to :code:`=` in Spark. For example, :code:`df1` and :code:`df2` are equivalent below:

    .. code-block::

        df1 = dataframe.filter("D = E")
        df2 = dataframe.filter(col(D) == col(E))

    The null-safe equality operator :code:`<=>` corresponds to :code:`eqNullSafe` `method <https://spark.apache.org/docs/3.0.1/api/python/pyspark.sql.html#pyspark.sql.Column.eqNullSafe>`_ . Concretely, :code:`df3` and :code:`df4` are equivalent below:

    .. code-block::

        df3 = dataframe.filter("D <=> E")
        df4 = dataframe.filter(col(D).eqNullSafe(col(E)))


NaNs and infs
^^^^^^^^^^^^^

1. :code:`inf = inf` evaluates to True. Consequently, :code:`inf <= inf` and :code:`inf >= inf` also evaluate to True.
2. :code:`NaN = NaN` evaluates to True (unlike standard floating point implementations including python's). For any non-null numeric value (including :code:`inf`) :code:`X`  (incl. :code:`inf`), :code:`NaN > X` evaluates to True and :code:`X > NaN` evaluates to False. :code:`NaN = X` evaluates to False for all non-null values except :code:`NaN`.
3. :code:`inf > X` evaluates to True for all non-null numeric values (except :code:`inf` and :code:`nan`). :code:`-inf < X` evaluates to True for all non-null numeric values (except :code:`-inf`).


Filter
------

A :class:`~.Filter` transformation can be constructed with a SQL filter expression that may refer to one or more columns in the input domain and contain comparison operators (:code:`<, <=, >, >=, =, <=>`) and logical operators (:code:`AND, OR or NOT`).

The following table describes how the logical operators behave when one or both values are :code:`null` (note that :code:`AND` and :code:`OR` are commutative):

.. list-table:: Logical Operators and NULLs
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - X
     - Y
     - X AND Y
     - X OR Y
     - NOT X
   * - NULL
     - True
     - NULL
     - True
     - NULL
   * - NULL
     - False
     - False
     - NULL
     - NULL
   * - NULL
     - NULL
     - NULL
     - NULL
     - NULL

Comparison between two columns work according to the `comparison semantics <#comparison-operators>`_ described above. The following expressions demonstrate how a column can be compared against a special literal value:

* :code:`"X = 'INF'"` evaluates to True only if X is :code:`inf`
* :code:`"X = '-INF'"` evaluates to True only if X is :code:`-inf`
* :code:`"X = 'NaN'"` evaluates to True only if X is :code:`NaN` 
* :code:`"X <=> NULL"` evaluates to True only if X is :code:`null`

.. note::

    Since :code:`X = NULL` evaluates to :code:`NULL` for any value of :code:`X`, using the  filter expression :code:`"NOT X = NULL"` results in an empty DataFrame. In order to filter out rows where :code:`X` is null, filtering with :code:`"NOT X <=> NULL"` would work; however, :class:`~.DropNulls` is better suited for this since it also modifies the domain to indicate that nulls are absent from column :code:`X` in the output.

PartitionByKeys
---------------

For a :class:`~.PartitionByKeys` transformation, the :attr:`~.PartitionByKeys.list_values` corresponding to partition keys can contain :code:`Inf`, :code:`NaN` or :code:`null`. A partition corresponding to a particular key is obtained by comparing row values in key columns with the key values.

Quantile
--------

A :class:`~.NoisyQuantile` measurement requires a :class:`~.PandasSeriesDomain` (over :class:`~.NumpyIntegerDomain` or :class:`~.NumpyFloatDomain`) as its input domain. Additionally, if the input domain is a :class:`~.PandasSeriesDomain` over :class:`~.NumpyFloatDomain`, it should also disallow :code:`NaN`\ s.

When constructing quantile measurements that work on :class:`~.SparkDataFrameDomain`\ s (with :func:`~.create_quantile_measurement` for example), the input domain must disallow :code:`nulls` and :code:`NaNs` on the measure column. More generally, :class:`~.ApplyInPandas` does not support :attr:`~.ApplyInPandas.aggregation_function`\ s that operate on numeric nullable columns.

GroupBy
-------

For a :class:`~.GroupBy` transformation, a group key can contain a null
only if the input domain permits nulls in the corresponding :class:`~.SparkColumnDescriptor`. 
A group key containing a null (or one that is a null -- when grouping by a single column) is treated
like any other value - i.e. all rows with this key are grouped together.
Since :class:`~.GroupBy` does not permit grouping on :class:`~.SparkFloatColumnDescriptor`
columns, group keys cannot contain NaNs or infs.

Joins
-----

Both :class:`~.PrivateJoin` and :class:`~.PublicJoin` use the :code:`=` semantics described above by default.
Consequently, all null values on the join columns are dropped. In order to join on nulls, construct the transformation with :code:`join_on_nulls=True` to use the :code:`<=>` semantics.

Removing NaNs, nulls, and infs
------------------------------

Tumult Core provides transformations to drop or replace NaNs, nulls, and infs. In particular, 
:class:`~.ReplaceNulls`, :class:`~.ReplaceNaNs`, and :class:`~.ReplaceInfs` allow replacing these values on one or more columns; :class:`~.DropNulls`, :class:`~.DropNaNs`, and :class:`~.DropInfs` allow dropping rows containing these values in one or more columns.


Sum and SumGrouped
------------------

:class:`~.Sum` and :class:`~.SumGrouped` aggregations require NaNs and nulls to be disallowed
from the measure column. Consequently, derived measurements (requiring sums) like :func:`~.create_average_measurement`, :func:`~.create_standard_deviation_measurement` and :func:`~.create_variance_measurement` also require that the measure column disallow NaNs and nulls.

+/- :code:`inf` values are correctly clipped to the upper and lower clipping bounds specified
on the aggregations.


CountDistinct
-------------

:class:`~.CountDistinct` uses the :code:`<=>` semantics `described above <#comparison-operators>`_ . For example, the following rows are considered identical by this transformation:

* :code:`(NULL, NaN, Inf)`
* :code:`(NULL, NaN, Inf)`
