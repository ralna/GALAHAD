.. index:: pair: table; rpd_inform_type
.. _doxid-structrpd__inform__type:

rpd_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_rpd.h>
	
	struct rpd_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structrpd__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structrpd__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structrpd__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`io_status<doxid-structrpd__inform__type_1a0ae587ad93ebdbad173f9e8475f936b9>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`line<doxid-structrpd__inform__type_1a41ebd28ef1d7c6ade45642cb6acc1039>`;
		char :ref:`p_type<doxid-structrpd__inform__type_1a1ed26df99ee0d5be3db580aff3ab5397>`[4];
	};
.. _details-structrpd__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structrpd__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. Possible values are:

* **0**

  The call was successful.

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

* **-22**

  An input/outpur error occurred.

* **-25**

  The end of the input file was reached prematurely.

* **-29**

  The problem type was not recognised.

.. index:: pair: variable; alloc_status
.. _doxid-structrpd__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation or deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structrpd__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation or deallocation error occurred

.. index:: pair: variable; io_status
.. _doxid-structrpd__inform__type_1a0ae587ad93ebdbad173f9e8475f936b9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` io_status

status from last read attempt

.. index:: pair: variable; line
.. _doxid-structrpd__inform__type_1a41ebd28ef1d7c6ade45642cb6acc1039:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` line

number of last line read from i/o file

.. index:: pair: variable; p_type
.. _doxid-structrpd__inform__type_1a1ed26df99ee0d5be3db580aff3ab5397:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char p_type[4]

problem type

