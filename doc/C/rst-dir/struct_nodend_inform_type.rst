.. index:: pair: table; nodend_inform_type
.. _doxid-structnodend__inform__type:

nodend_inform_type structure
----------------------------

.. toctree::
     :hidden:

.. ref-code-block:: cpp
     :class: doxyrest-overview-code-block

     #include <galahad_nodend.h>
	
     struct nodend_inform_type {
       // fields
	
       :ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structnodend__inform__type_status>`;
       :ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structnodend__inform__type_alloc_status>`;
       char :ref:`bad_alloc<doxid-structnodend__inform__type_bad_alloc>`[81];
       char :ref:`version<doxid-structnodend__inform__type_version>`[4];
       struct :ref:`nodend_time_type<doxid-structnodend__time__type>` :ref:`time<doxid-structnodend__inform__type_time>`;
       };
.. _details-structnodend__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structnodend__inform__type_status:

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

  One of the restrictions
  n $> 0$, A.n $> 0$ or A.ne $< 0$, for co-ordinate entry,
  or requirements that A.type contain its relevant string
  'COORDINATE', 'SPARSE_BY_ROWS' or 'DENSE', and
  control.version in one of '4.0', '5.1' or '5.2'
  has been violated.

* **-26**

  The requested version of METIS is not available.

* **-57**

  METIS has insufficient memory to continue.

* **-71**

  An internal METIS error occurred.

.. index:: pair: variable; alloc_status
.. _doxid-structnodend__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structnodend__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; version
.. _doxid-structnodend__inform__type_version:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char version[4]

the version of METIS that was actually used

.. index:: pair: variable; time
.. _doxid-structnodend__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`nodend_time_type<doxid-structnodend__time__type>` time

timings (see above)
