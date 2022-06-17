.. _privacy-guarantee:

Tumult Core Privacy Guarantee
=============================

The privacy guarantee of a Core :class:`~.Measurement`, :math:`\mathcal{M}` is
the following. Let :math:`r` denote the privacy relation of :math:`\mathcal{M}`,
:math:`I` denote the input domain, and :math:`d` denote the input metric, and
let :math:`D` denote the output measure. Then, for any pair of elements
:math:`x, y \in I` and for all distances :math:`\alpha` such that :math:`d(x,y)
\le \alpha`, :math:`r(\alpha, \beta) = True` implies
:math:`D(\mathcal{M}(x), \mathcal{M}(y)) \le \beta`.

This privacy guarantee generalizes :math:`\epsilon`-differential privacy, and we
can get the standard :math:`\epsilon`-differential privacy guarantee for
specific settings of the parameters. Suppose the input metric is
:class:`~.SymmetricDifference` and the output measure is defined as
:math:`D(X, Y) = \epsilon` if :math:`D_{\infty}(X \| Y) = \epsilon` (in Core,
this is when output measure of the :class:`~.Measurement` is :class:`~.PureDP`).
Then, :math:`r(1, \epsilon) = True` implies that the mechanism satisfies
:math:`\epsilon`-differential privacy.

The rest of this section discusses qualifications to the privacy guarantee with
respect to side channel information. In addition, see
:ref:`known-vulnerabilities` for vulnerabilities in Tumult Core that may affect
the privacy guarantee.

.. _pseudo-side-channel:

Pseudo-side channel information
-------------------------------

The privacy guarantee in the previous section applies to the abstract outputs of
the mechanism. That is, the output of :math:`\mathcal{M}` may be a list of
numbers, or a multiset of records. However, the implementations in Tumult Core
are python objects that may contain additional information that could leak
private data. An example of pseudo-side channel information is the ordering of
records in Spark DataFrame, which is meant to represent an unordered multiset of
records.

We call pseudo-side channel information *distinguishing* if it can be used to
learn about the private input.  In Tumult Core, protecting against leakage of
distinguishing pseudo-side channel information is not part of the privacy guarantee.
However, we make a best-effort attempt to make sure that all pseudo-side channel
information released by measurements is not distinguishing.

.. _pseudo-side-channel-mitigations:

Specific protections against pseudo-side channel leakage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For Spark DataFrame outputs, we perform the following mitigations against leaking
distinguishing pseudo-side channel information.

-  **Materialize the DataFrame**: Spark DataFrames are computed lazily, and
   Spark DataFrame with random noise will compute a new sample of the noise
   each time an action is performed (e.g. printing the DataFrame). To prevent
   this, we eagerly compute the DataFrame and save it.
-  **Repartition the DataFrame**: Spark DataFrames are partitioned, and the number
   and content of these partitions are potentially distinguishing pseudo-side channel
   information. To prevent this, we randomly repartition the output DataFrame.
-  **Sorting each partition**: The records in a Spark DataFrame have an order, but our
   privacy guarantee is on the unordered multiset of records that this DataFrame
   represents. To prevent the ordering from leaking private data, after repartitioning
   the DataFrame as described above, we sort each partition.


.. _postprocessing-udf-assumptions:

Postprocessing udfs and pseudo-side channel information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some parts of the Tumult Core code accept user code when constructing
pre/postprocessing functions (:class:`~.PostProcess`,
:class:`~.NonInteractivePostProcess`, :class:`~.DecorateQueryable`). For these
measurements, our privacy guarantee relies on the assumption that these
functions do not use distinguishing pseudo-side channel information. Tumult Core
makes a best effort attempt to make sure all pseudo-side channel information is
not distinguishing, see :ref:`pseudo-side-channel-mitigations` for details.

More formally, our privacy guarantee is based on the assumption that these
functions are well defined on the abstract domains.  Let :math:`f` be the
function. Let :math:`A` be abstract input domain of :math:`f`.  That is,
:math:`A` contains the abstract elements represented by objects passed to the
implementation of :math:`f`. (in the case of a Spark DataFrame, these elements
are unordered multisets of records).  Likewise, let :math:`B` be the abstract
output domain of `f`. Then for any :math:`x,y \in A` such that :math:`x = y`, it
must be the case that :math:`f(x) = f(y)`.

Suppose, for example, that Tumult core did not protect against information
leakage via the ordering of the records in the Spark DataFrame, and the ordering
revealed something about the private data. Then for some record :math:`r`,
consider a postprocessing function on a Spark DataFrame that outputs 1 if the
first record of the DataFrame is :math:`r`, and 0 otherwise. Such a function
would break the Tumult Core privacy guarantee, because it uses distinguishing
pseudo-side channel information.  This function is also not well defined on the
abstract input domain.  There exist two DataFrames :math:`D,D'` that represent
the same multiset of records (and are therefore equal in the abstract domain),
but :math:`f(D) \ne f(D')` because :math:`r` is the first record of :math:`D` but
not :math:`D'`. This is example is hypothetical since Tumult Core *does* protect
against information leakage via the ordering.


.. _side-channel:

Side channel information
------------------------

Side channel information includes any information that can be learned from
running a measurement that is not explicitly part of the output of the
measurement. Examples include the amount of time it takes for a measurement to
run, or the amount of memory consumed when running the measurement. Note that
the amount of time the measurement takes to run could be measured indirectly by
the user: if user code adds timestamped entries to a logfile at different points
in the measurement, the resulting logfile could leak private data and this is not
protected by the Tumult Core guarantee.

The privacy guarantee of Core Measurements applies only to the explicit output,
it does not extend to any side channel information. Additionally, Tumult Core
makes no attempt to make side channel information non-distinguishing.
