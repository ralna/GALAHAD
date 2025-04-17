.. index:: pair: struct; ugo_time_type
.. _doxid-structugo__time__type:

ugo_time_type structure
-----------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ugo.h>
	
	struct ugo_time_type {
		// components
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structugo__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structugo__time__type_clock_total>`;
	};
.. _details-structugo__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structugo__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; clock_total
.. _doxid-structugo__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

