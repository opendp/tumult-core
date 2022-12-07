Tumult Core documentation
=========================

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation <installation>
   tutorials/index
   topic-guides/index
   API reference <reference/tmlt/core/index>
   additional-resources/index

Introduction to Tumult Core
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tumult Core is a collection of composable components for implementing
algorithms to perform differentially private computations. The design of Tumult Core
is based on the design proposed in the `OpenDP White Paper
<https://projects.iq.harvard.edu/files/opendifferentialprivacy/files/opendp_white_paper_11may2020.pdf>`__,
and can automatically verify the privacy properties of algorithms constructed
from Tumult Core components. Tumult Core is scalable, includes a wide variety of components,
and supports multiple privacy definitions.

A good starting point for new users is the :ref:`installation instructions <installation instructions>`.

Intended Users
^^^^^^^^^^^^^^

This library is intended for advanced users who need capabilities beyond those available in Tumult Analytics.
Most users should use `Tumult Analytics <https://docs.tmlt.dev/analytics/>`__ rather than using Tumult Core directly.

.. panels::
   :card: + intro-card text-center
   :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 p-2

   ---
   :img-top: /images/index_tutorials.svg

   **Tutorials**
   ^^^^^^^^^^^^^

   Tutorials are the place where new users can learn the basics of how to use
   the library.

   .. link-button:: tutorials/index
       :type: ref
       :text:
       :classes: stretched-link

   ---
   :img-top: images/index_topic_guides.svg

   **Topic guides**
   ^^^^^^^^^^^^^^^^

   Topic guides explain core concepts and advanced topics in more detail.

   .. link-button:: topic-guides/index
       :type: ref
       :text:
       :classes: stretched-link

   ---
   :img-top: images/index_api.svg

   **API reference**
   ^^^^^^^^^^^^^^^^^

   The API reference contains a detailed description of the packages, classes, and
   methods available in Tumult Core. It assumes that you have an understanding of the
   key concepts.

   .. link-button:: reference/tmlt/core/index
       :type: ref
       :text:
       :classes: stretched-link

   ---
   :img-top: images/index_more.svg


   **Additional resources**
   ^^^^^^^^^^^^^^^^^^^^^^^^

   Additional resources include the :ref:`changelog <core-changelog>`, which
   describes notable changes to the library,
   :ref:`supporting materials <Bibliography>`,
   as well as :ref:`license information <License>`.

Contact Information
^^^^^^^^^^^^^^^^^^^
The best place to ask questions, file feature requests, or give feedback about Tumult Core
is our `Slack server <https://tmltdev.slack.com/join/shared_invite/zt-1bky0mh9v-vOB8azKAVoxmzJDUdWd5Wg#>`__.
We also use it for announcements of new releases and feature additions.

Documentation License
^^^^^^^^^^^^^^^^^^^^^
This documentation is licensed under the
Creative Commons Attribution-ShareAlike 4.0 Unported License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
