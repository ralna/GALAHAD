.. index:: pair: table; lms_control_type
.. _doxid-structlms__control__type:

lms_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lms.h>
	
	struct lms_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structlms__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structlms__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structlms__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structlms__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`memory_length<doxid-structlms__control__type_memory_length>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`method<doxid-structlms__control__type_method>`;
		bool :ref:`any_method<doxid-structlms__control__type_any_method>`;
		bool :ref:`space_critical<doxid-structlms__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structlms__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structlms__control__type_prefix>`[31];
	};
.. _details-structlms__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlms__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlms__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structlms__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structlms__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; memory_length
.. _doxid-structlms__control__type_memory_length:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` memory_length

limited memory length

.. index:: pair: variable; method
.. _doxid-structlms__control__type_method:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` method

limited-memory formula required (others may be added in due course):

* 1 BFGS (default).

* 2 Symmetric Rank-One (SR1).

* 3 The inverse of the BFGS formula.

* 4 The inverse of the shifted BFGS formula. This should be used instead of .method = 3 whenever a shift is planned.

.. index:: pair: variable; any_method
.. _doxid-structlms__control__type_any_method:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool any_method

allow space to permit different methods if required (less efficient)

.. index:: pair: variable; space_critical
.. _doxid-structlms__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlms__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structlms__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

