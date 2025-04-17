.. index:: pair: table; roots_control_type
.. _doxid-structroots__control__type:

roots_control_type structure
----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_roots.h>
	
	struct roots_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structroots__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structroots__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structroots__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structroots__control__type_print_level>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`tol<doxid-structroots__control__type_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_coef<doxid-structroots__control__type_zero_coef>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_f<doxid-structroots__control__type_zero_f>`;
		bool :ref:`space_critical<doxid-structroots__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structroots__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structroots__control__type_prefix>`[31];
	};
.. _details-structroots__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structroots__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structroots__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structroots__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structroots__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; tol
.. _doxid-structroots__control__type_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` tol

the required accuracy of the roots

.. index:: pair: variable; zero_coef
.. _doxid-structroots__control__type_zero_coef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_coef

any coefficient smaller in absolute value than zero_coef will be regarde to be zero

.. index:: pair: variable; zero_f
.. _doxid-structroots__control__type_zero_f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_f

any value of the polynomial smaller in absolute value than zero_f will be regarded as giving a root

.. index:: pair: variable; space_critical
.. _doxid-structroots__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structroots__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structroots__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

