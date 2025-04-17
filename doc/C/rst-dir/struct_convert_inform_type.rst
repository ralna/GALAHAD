.. index:: pair: table; convert_inform_type
.. _doxid-structconvert__inform__type:

convert_inform_type structure
-----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_convert.h>
	
	struct convert_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structconvert__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structconvert__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`duplicates<doxid-structconvert__inform__type_duplicates>`;
		char :ref:`bad_alloc<doxid-structconvert__inform__type_bad_alloc>`[81];
		struct :ref:`convert_time_type<doxid-structconvert__time__type>` :ref:`time<doxid-structconvert__inform__type_time>`;
	};
.. _details-structconvert__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structconvert__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

the return status. Possible values are:

* **0**

  a successful conversion.

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

  The restriction n > 0 or m > 0 or requirement that a type contains its
  relevant string 'coordinate', 'sparse_by_rows', 'sparse_by_columns',
  'dense_by_rows' or 'dense_by_columns' has been violated.

* **-32**

  provided integer workspace is not large enough.

* **-33**

  provided real workspace is not large enough.

* **-73**

  an input matrix entry has been repeated.

* **-79**

  there are missing optional arguments.

* **-90**

  a requested output format is not recognised.

.. index:: pair: variable; alloc_status
.. _doxid-structconvert__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; duplicates
.. _doxid-structconvert__inform__type_duplicates:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` duplicates

the number of duplicates found (-ve = not checked).

.. index:: pair: variable; bad_alloc
.. _doxid-structconvert__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; time
.. _doxid-structconvert__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`convert_time_type<doxid-structconvert__time__type>` time

timings (see above).

