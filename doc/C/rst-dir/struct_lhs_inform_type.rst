.. index:: pair: table; lhs_inform_type
.. _doxid-structlhs__inform__type:

lhs_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lhs.h>
	
	struct lhs_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structlhs__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structlhs__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structlhs__inform__type_bad_alloc>`[81];
	};
.. _details-structlhs__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlhs__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. Possible values are:

* **0**

  the call was successful.

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

* **-3**

  The random number seed has not been set.

.. index:: pair: variable; alloc_status
.. _doxid-structlhs__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; bad_alloc
.. _doxid-structlhs__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

