.. index:: pair: table; scu_inform_type
.. _doxid-structscu__inform__type:

scu_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_scu.h>
	
	struct scu_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structscu__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structscu__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`inertia<doxid-structscu__inform__type_1a4387aa0c0c3c91d72fb78c8925f05993>`[3];
	};
.. _details-structscu__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structscu__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. A non-zero value indicates an error or a request for further information. See SCU_solve for details.

.. index:: pair: variable; alloc_status
.. _doxid-structscu__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the return status from the last attempted internal workspace array allocation or deallocation. A non-zero value indicates that the allocation or deallocation was unsuccessful, and corresponds to the fortran STAT= value on the user’s system.

.. index:: pair: variable; inertia
.. _doxid-structscu__inform__type_1a4387aa0c0c3c91d72fb78c8925f05993:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` inertia[3]

the inertia of $S$ when the extended matrix is symmetric. Specifically, inertia(i), i=0,1,2 give the number of positive, negative and zero eigenvalues of $S$ respectively.

