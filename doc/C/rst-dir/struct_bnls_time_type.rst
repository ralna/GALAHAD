.. index:: pair: struct; bnls_time_type
.. _doxid-structbnls__time__type:

bnls_time_type structure
------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bnls.h>
	
	struct bnls_time_type {
		// components
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structbnls__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`blls<doxid-structbnls__time__type_blls>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`bllsb<doxid-structbnls__time__type_bllsb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structbnls__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_blls<doxid-structbnls__time__type_clock_blls>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_bllsb<doxid-structbnls__time__type_clock_bllsb>`;
	};
.. _details-structbnls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbnls__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

the total CPU time spent in the package

.. index:: pair: variable; blls
.. _doxid-structbnls__time__type_blls:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` blls

the CPU time spent in the ``blls`` package

.. index:: pair: variable; bllsb
.. _doxid-structbnls__time__type_bllsb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` bllsb

the CPU time spent in the ``bllsb`` package

.. index:: pair: variable; clock_total
.. _doxid-structbnls__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

the total clock time spent in the package

.. index:: pair: variable; clock_blls
.. _doxid-structbnls__time__type_clock_blls:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_blls

the clock time spent in the ``blls`` package

.. index:: pair: variable; clock_bllsb
.. _doxid-structbnls__time__type_clock_bllsb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_bllsb

the clock time spent in the ``bllsb`` package
