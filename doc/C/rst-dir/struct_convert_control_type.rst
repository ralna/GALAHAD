.. index:: pair: table; convert_control_type
.. _doxid-structconvert__control__type:

convert_control_type structure
------------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_convert.h>
	
	struct convert_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structconvert__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structconvert__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structconvert__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structconvert__control__type_print_level>`;
		bool :ref:`transpose<doxid-structconvert__control__type_transpose>`;
		bool :ref:`sum_duplicates<doxid-structconvert__control__type_sum_duplicates>`;
		bool :ref:`order<doxid-structconvert__control__type_order>`;
		bool :ref:`space_critical<doxid-structconvert__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structconvert__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structconvert__control__type_prefix>`[31];
	};
.. _details-structconvert__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structconvert__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structconvert__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structconvert__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structconvert__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; transpose
.. _doxid-structconvert__control__type_transpose:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool transpose

obtain the transpose of the input matrix?

.. index:: pair: variable; sum_duplicates
.. _doxid-structconvert__control__type_sum_duplicates:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool sum_duplicates

add the values of entries in duplicate positions?

.. index:: pair: variable; order
.. _doxid-structconvert__control__type_order:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool order

order row or column data by increasing index?

.. index:: pair: variable; space_critical
.. _doxid-structconvert__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structconvert__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structconvert__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

