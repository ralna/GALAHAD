.. index:: pair: struct; nodend_time_type
.. _doxid-structnodend__time__type:

nodend_time_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_nodend.h>
	
	struct nodend_time_type {
		// components
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structnodend__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`metis<doxid-structnodend__time__type_metis>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structnodend__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_metis<doxid-structnodend__time__type_clock_metis>`;
	};
.. _details-structnodend__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structnodend__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; metis
.. _doxid-structnodend__time__type_metis:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` metis

the CPU time spent in the METIS package

.. index:: pair: variable; clock_total
.. _doxid-structnodend__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_metis
.. _doxid-structnodend__time__type_clock_metis:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_metis

the clock time spent in the METIS package


