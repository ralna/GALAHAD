.. index:: pair: table; lms_time_type
.. _doxid-structlms__time__type:

lms_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lms.h>
	
	struct lms_time_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structlms__time__type_1ad3803b3bb79c5c74d9300520fbe733f4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`setup<doxid-structlms__time__type_1aaa0ec60bfc99c0ffd31001a4f59036b4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`form<doxid-structlms__time__type_1a8ac63de5e103d8e01b0e0f88bb7d230d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`apply<doxid-structlms__time__type_1a9d8129bf5b1a9f21dfcc24dc5c706274>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structlms__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_setup<doxid-structlms__time__type_1ac2bdffea5b986f6acd6b53c4d2344910>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_form<doxid-structlms__time__type_1ab275f3b71b8e019aa35acf43c3fd7473>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_apply<doxid-structlms__time__type_1afbbb1dd5fc63c640620fbd32a0481493>`;
	};
.. _details-structlms__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structlms__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

total cpu time spent in the package

.. index:: pair: variable; setup
.. _doxid-structlms__time__type_1aaa0ec60bfc99c0ffd31001a4f59036b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` setup

cpu time spent setting up space for the secant approximation

.. index:: pair: variable; form
.. _doxid-structlms__time__type_1a8ac63de5e103d8e01b0e0f88bb7d230d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` form

cpu time spent updating the secant approximation

.. index:: pair: variable; apply
.. _doxid-structlms__time__type_1a9d8129bf5b1a9f21dfcc24dc5c706274:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` apply

cpu time spent applying the secant approximation

.. index:: pair: variable; clock_total
.. _doxid-structlms__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

total clock time spent in the package

.. index:: pair: variable; clock_setup
.. _doxid-structlms__time__type_1ac2bdffea5b986f6acd6b53c4d2344910:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_setup

clock time spent setting up space for the secant approximation

.. index:: pair: variable; clock_form
.. _doxid-structlms__time__type_1ab275f3b71b8e019aa35acf43c3fd7473:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_form

clock time spent updating the secant approximation

.. index:: pair: variable; clock_apply
.. _doxid-structlms__time__type_1afbbb1dd5fc63c640620fbd32a0481493:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_apply

clock time spent applying the secant approximation

