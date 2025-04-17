.. index:: pair: table; lhs_control_type
.. _doxid-structlhs__control__type:

lhs_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lhs.h>
	
	struct lhs_control_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structlhs__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structlhs__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structlhs__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`duplication<doxid-structlhs__control__type_duplication>`;
		bool :ref:`space_critical<doxid-structlhs__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structlhs__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structlhs__control__type_prefix>`[31];
	};
.. _details-structlhs__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; error
.. _doxid-structlhs__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error.

.. index:: pair: variable; out
.. _doxid-structlhs__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out.

.. index:: pair: variable; print_level
.. _doxid-structlhs__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required. Possible values are:

* < 1 no output.

* > 0 debugging.

.. index:: pair: variable; duplication
.. _doxid-structlhs__control__type_duplication:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` duplication

the duplication factor. This must be at least 1, a value of 5 is reasonable.

.. index:: pair: variable; space_critical
.. _doxid-structlhs__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time.

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlhs__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue.

.. index:: pair: variable; prefix
.. _doxid-structlhs__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

