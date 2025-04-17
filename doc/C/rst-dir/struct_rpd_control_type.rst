.. index:: pair: table; rpd_control_type
.. _doxid-structrpd__control__type:

rpd_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_rpd.h>
	
	struct rpd_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structrpd__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`qplib<doxid-structrpd__control__type_qplib>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structrpd__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structrpd__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structrpd__control__type_print_level>`;
		bool :ref:`space_critical<doxid-structrpd__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structrpd__control__type_deallocate_error_fatal>`;
	};
.. _details-structrpd__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structrpd__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; qplib
.. _doxid-structrpd__control__type_qplib:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` qplib

QPLIB file input stream number.

.. index:: pair: variable; error
.. _doxid-structrpd__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structrpd__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structrpd__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

* $\leq$ 0 gives no output,

* $\geq$ 1 gives increasingly verbose (debugging) output

.. index:: pair: variable; space_critical
.. _doxid-structrpd__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structrpd__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

