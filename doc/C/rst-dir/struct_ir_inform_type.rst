.. index:: pair: table; ir_inform_type
.. _doxid-structir__inform__type:

ir_inform_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ir.h>
	
	struct ir_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structir__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structir__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structir__inform__type_bad_alloc>`[81];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_initial_residual<doxid-structir__inform__type_norm_initial_residual>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_final_residual<doxid-structir__inform__type_norm_final_residual>`;
	};
.. _details-structir__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structir__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

the return status. Possible values are:

* **0**

  the solution has been found.

* **-1**

  An allocation error occurred. A message indicating the offending array
  is written on unit control.error, and the returned allocation status
  and a string containing the name of the offending array are held in
  inform.alloc_status and inform.bad_alloc respectively.

* **-2**

  A deallocation error occurred. A message indicating the offending
  array is written on unit control.error and the returned allocation
  status and a string containing the name of the offending array are
  held in inform.alloc_status and inform.bad_alloc respectively.

* **-11**

  Iterative refinement has not reduced the relative residual by more
  than control.required_relative_residual.

.. index:: pair: variable; alloc_status
.. _doxid-structir__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; bad_alloc
.. _doxid-structir__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; norm_initial_residual
.. _doxid-structir__inform__type_norm_initial_residual:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_initial_residual

the infinity norm of the initial residual

.. index:: pair: variable; norm_final_residual
.. _doxid-structir__inform__type_norm_final_residual:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_final_residual

the infinity norm of the final residual

