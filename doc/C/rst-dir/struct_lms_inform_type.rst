.. index:: pair: table; lms_inform_type
.. _doxid-structlms__inform__type:

lms_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lms.h>
	
	struct lms_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structlms__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structlms__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`length<doxid-structlms__inform__type_length>`;
		bool :ref:`updates_skipped<doxid-structlms__inform__type_updates_skipped>`;
		char :ref:`bad_alloc<doxid-structlms__inform__type_bad_alloc>`[81];
		struct :ref:`lms_time_type<doxid-structlms__time__type>` :ref:`time<doxid-structlms__inform__type_time>`;
	};
.. _details-structlms__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlms__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

the return status. Possible values are:

* **0**

  the update was successful.

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

  One of the restrictions n > 0, delta > 0, lambda > 0 or s^T y > 0 has
  been violated and the update has been skipped.

* **-10**

  The matrix cannot be built from the current vectors {s_k} and {y_k}
  and values delta_k and lambda_k and the update has been skipped.

* **-31**

  A call to the function lhs_apply has been made without a prior call to
  lhs_form_shift or lhs_form with lambda specified when control.method =
  4, or lhs_form_shift has been called when control.method = 3, or
  lhs_change_method has been called after control.any_method = false was
  specified when calling lhs_setup.

.. index:: pair: variable; alloc_status
.. _doxid-structlms__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; length
.. _doxid-structlms__inform__type_length:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` length

the number of pairs (s,y) currently used to represent the limited-memory matrix.

.. index:: pair: variable; updates_skipped
.. _doxid-structlms__inform__type_updates_skipped:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool updates_skipped

have (s,y) pairs been skipped when forming the limited-memory matrix?

.. index:: pair: variable; bad_alloc
.. _doxid-structlms__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; time
.. _doxid-structlms__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_time_type<doxid-structlms__time__type>` time

timings (see above)

