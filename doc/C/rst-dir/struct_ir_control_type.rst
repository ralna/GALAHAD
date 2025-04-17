.. index:: pair: table; ir_control_type
.. _doxid-structir__control__type:

ir_control_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ir.h>
	
	struct ir_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structir__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structir__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structir__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structir__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itref_max<doxid-structir__control__type_itref_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`acceptable_residual_relative<doxid-structir__control__type_acceptable_residual_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`acceptable_residual_absolute<doxid-structir__control__type_acceptable_residual_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`required_residual_relative<doxid-structir__control__type_required_residual_relative>`;
		bool :ref:`record_residuals<doxid-structir__control__type_record_residuals>`;
		bool :ref:`space_critical<doxid-structir__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structir__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structir__control__type_prefix>`[31];
	};
.. _details-structir__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structir__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structir__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structir__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structir__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; itref_max
.. _doxid-structir__control__type_itref_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itref_max

maximum number of iterative refinements allowed

.. index:: pair: variable; acceptable_residual_relative
.. _doxid-structir__control__type_acceptable_residual_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` acceptable_residual_relative

refinement will cease as soon as the residual $\|Ax-b\|$ falls below max( acceptable_residual_relative \* $\|b\|$, acceptable_residual_absolute )

.. index:: pair: variable; acceptable_residual_absolute
.. _doxid-structir__control__type_acceptable_residual_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` acceptable_residual_absolute

see acceptable_residual_relative

.. index:: pair: variable; required_residual_relative
.. _doxid-structir__control__type_required_residual_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` required_residual_relative

refinement will be judged to have failed if the residual $\|Ax-b\| \geq$ required_residual_relative \* $\|b\|$. No checking if required_residual_relative < 0

.. index:: pair: variable; record_residuals
.. _doxid-structir__control__type_record_residuals:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool record_residuals

record the initial and final residual

.. index:: pair: variable; space_critical
.. _doxid-structir__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structir__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structir__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

