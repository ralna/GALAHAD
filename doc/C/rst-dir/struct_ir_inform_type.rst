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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structir__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structir__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structir__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_initial_residual<doxid-structir__inform__type_1a5d35136316d3841bb7f2d87495b619a9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_final_residual<doxid-structir__inform__type_1a95ba287dc64f4d10546b9ca9ea407fc2>`;
	};
.. _details-structir__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structir__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

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
.. _doxid-structir__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; bad_alloc
.. _doxid-structir__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; norm_initial_residual
.. _doxid-structir__inform__type_1a5d35136316d3841bb7f2d87495b619a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_initial_residual

the infinity norm of the initial residual

.. index:: pair: variable; norm_final_residual
.. _doxid-structir__inform__type_1a95ba287dc64f4d10546b9ca9ea407fc2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_final_residual

the infinity norm of the final residual

