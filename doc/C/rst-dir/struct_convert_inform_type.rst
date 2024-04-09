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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structconvert__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structconvert__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`duplicates<doxid-structconvert__inform__type_1a4266bf48aafe2914b08e60d6ef9cf446>`;
		char :ref:`bad_alloc<doxid-structconvert__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		struct :ref:`convert_time_type<doxid-structconvert__time__type>` :ref:`time<doxid-structconvert__inform__type_1a9d7e0c775ea50ee659169c07a40bb27d>`;
	};
.. _details-structconvert__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structconvert__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

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
.. _doxid-structconvert__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; duplicates
.. _doxid-structconvert__inform__type_1a4266bf48aafe2914b08e60d6ef9cf446:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` duplicates

the number of duplicates found (-ve = not checked).

.. index:: pair: variable; bad_alloc
.. _doxid-structconvert__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; time
.. _doxid-structconvert__inform__type_1a9d7e0c775ea50ee659169c07a40bb27d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`convert_time_type<doxid-structconvert__time__type>` time

timings (see above).

