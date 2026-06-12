.. index:: pair: struct; snls_time_type
.. _doxid-structsnls__time__type:

snls_time_type structure
------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_snls.h>
	
	struct snls_time_type {
		// components
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structsnls__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`slls<doxid-structsnls__time__type_slls>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`sllsb<doxid-structsnls__time__type_sllsb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structsnls__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_slls<doxid-structsnls__time__type_clock_slls>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_sllsb<doxid-structsnls__time__type_clock_sllsb>`;
	};
.. _details-structsnls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structsnls__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; slls
.. _doxid-structsnls__time__type_slls:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` slls

the CPU time spent in the ``slls`` package

.. index:: pair: variable; sllsb
.. _doxid-structsnls__time__type_sllsb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` sllsb

the CPU time spent in the ``sllsb`` package

.. index:: pair: variable; clock_total
.. _doxid-structsnls__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_slls
.. _doxid-structsnls__time__type_clock_slls:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_slls

the clock time spent in the ``slls`` package

.. index:: pair: variable; clock_sllsb
.. _doxid-structsnls__time__type_clock_sllsb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_sllsb

the clock time spent in the ``sllsb`` package
