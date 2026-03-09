.. index:: pair: struct; blls_time_type
.. _doxid-structblls__time__type:

blls_time_type structure
------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_blls.h>
	
	struct blls_time_type {
		// components
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structblls__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structblls__time__type_clock_total>`;
	};
.. _details-structblls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structblls__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; clock_total
.. _doxid-structblls__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package


