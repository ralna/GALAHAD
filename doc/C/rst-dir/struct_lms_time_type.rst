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
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structlms__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`setup<doxid-structlms__time__type_setup>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`form<doxid-structlms__time__type_form>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`apply<doxid-structlms__time__type_apply>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structlms__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_setup<doxid-structlms__time__type_clock_setup>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_form<doxid-structlms__time__type_clock_form>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_apply<doxid-structlms__time__type_clock_apply>`;
	};
.. _details-structlms__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structlms__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

total cpu time spent in the package

.. index:: pair: variable; setup
.. _doxid-structlms__time__type_setup:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` setup

cpu time spent setting up space for the secant approximation

.. index:: pair: variable; form
.. _doxid-structlms__time__type_form:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` form

cpu time spent updating the secant approximation

.. index:: pair: variable; apply
.. _doxid-structlms__time__type_apply:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` apply

cpu time spent applying the secant approximation

.. index:: pair: variable; clock_total
.. _doxid-structlms__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

total clock time spent in the package

.. index:: pair: variable; clock_setup
.. _doxid-structlms__time__type_clock_setup:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_setup

clock time spent setting up space for the secant approximation

.. index:: pair: variable; clock_form
.. _doxid-structlms__time__type_clock_form:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_form

clock time spent updating the secant approximation

.. index:: pair: variable; clock_apply
.. _doxid-structlms__time__type_clock_apply:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_apply

clock time spent applying the secant approximation

