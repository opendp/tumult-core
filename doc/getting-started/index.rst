Getting Started
===============

Introduction to Tumult Core
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tumult Core is a collection of composable components for implementing
algorithms to perform differentially private computations. The design of Tumult Core
is based on the design proposed in the `OpenDP White Paper
<https://projects.iq.harvard.edu/files/opendifferentialprivacy/files/opendp_white_paper_11may2020.pdf>`_,
and can automatically verify the privacy properties of algorithms constructed
from Tumult Core components. Tumult Core is scalable, includes a wide variety of components,
and supports multiple privacy definitions.

Intended Users
^^^^^^^^^^^^^^

This library is intended to be used by data scientists and programmers who are familiar with differential privacy.

Installation Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: In order to install Tumult Core, you'll need a token-username and deploy-token.
    If you do not currently have these, contact `support@tmlt.io <mailto:support@tmlt.io>`_

.. code-block:: bash

    pip install --extra-index-url https://<token-username>:<token>@gitlab.com/api/v4/projects/17405343/packages/pypi/simple tmlt.core

The package files can also be downloaded directly, either as a wheel:

.. code-block:: bash

    pip download --no-deps --extra-index-url https://<token-username>:<token>@gitlab.com/api/v4/projects/17405343/packages/pypi/simple tmlt.core

or as a source distribution:

.. code-block:: bash

    pip download --no-deps --no-binary=:all: --extra-index-url https://<token-username>:<token>@gitlab.com/api/v4/projects/17405343/packages/pypi/simple tmlt.core

.. note:: `PySpark <http://spark.apache.org/docs/latest/api/python/>`__ is a Tumult Core dependency  and will be installed automatically. However, PySpark may have `additional dependencies <http://spark.apache.org/docs/latest/api/python/getting_started/install.html#dependencies>`__, such as Java 8.


..
   TODO(#1845): Remove this section once Windows support is added.

.. attention:: If you are installing on a Windows machine, please install `python-flint <https://fredrikj.net/python-flint/>`__ (see `instructions <https://github.com/fredrik-johansson/python-flint/#installation>`__) before installing Tumult Core. 


How this documentation is organized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a high level overview of Tumult Core's documentation:

- :ref:`Tutorials` section aims to take the user by the hand through insightful guided exercises using Tumult Core.
- :ref:`Explanations` provide a high level overview of various concepts underlying the library.
- :ref:`Reference` contains detailed technical descriptions of all public classes and functions in the Tumult Core library.

