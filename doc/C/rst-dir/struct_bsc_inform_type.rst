.. index:: pair: table; bsc_inform_type
.. _doxid-structbsc__inform__type:

bsc_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bsc.h>
	
	struct bsc_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structbsc__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structbsc__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structbsc__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col_a<doxid-structbsc__inform__type_max_col_a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`exceeds_max_col<doxid-structbsc__inform__type_exceeds_max_col>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`time<doxid-structbsc__inform__type_time>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time<doxid-structbsc__inform__type_clock_time>`;
	};
.. _details-structbsc__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structbsc__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

the return status from the package. Possible values are:

* **0**

  The call was succcesful

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

  The restrictions n > 0 or m > 0 or requirement that a type contains
  its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been
  violated.

.. index:: pair: variable; alloc_status
.. _doxid-structbsc__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structbsc__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; max_col_a
.. _doxid-structbsc__inform__type_max_col_a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col_a

the maximum number of entries in a column of $A$

.. index:: pair: variable; exceeds_max_col
.. _doxid-structbsc__inform__type_exceeds_max_col:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` exceeds_max_col

the number of columns of $A$ that have more than control.max_col entries

.. index:: pair: variable; time
.. _doxid-structbsc__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` time

the total CPU time spent in the package

.. index:: pair: variable; clock_time
.. _doxid-structbsc__inform__type_clock_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time

the total clock time spent in the package

