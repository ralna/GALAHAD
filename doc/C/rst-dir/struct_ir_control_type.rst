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
	
		bool :ref:`f_indexing<doxid-structir__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structir__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structir__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structir__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itref_max<doxid-structir__control__type_1a903ba4ef0869186a65d4c32459a6a0ed>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`acceptable_residual_relative<doxid-structir__control__type_1a97a6571829dbdccad7598f7b5c3ddfbd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`acceptable_residual_absolute<doxid-structir__control__type_1a5ee0e70d90b1398019054b19b68057a0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`required_residual_relative<doxid-structir__control__type_1a9d3614debfe53f901305b68d9e460163>`;
		bool :ref:`record_residuals<doxid-structir__control__type_1aee80ae09cedfc446424c56719b30cfca>`;
		bool :ref:`space_critical<doxid-structir__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structir__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structir__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structir__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structir__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structir__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structir__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structir__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; itref_max
.. _doxid-structir__control__type_1a903ba4ef0869186a65d4c32459a6a0ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itref_max

maximum number of iterative refinements allowed

.. index:: pair: variable; acceptable_residual_relative
.. _doxid-structir__control__type_1a97a6571829dbdccad7598f7b5c3ddfbd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` acceptable_residual_relative

refinement will cease as soon as the residual $\|Ax-b\|$ falls below max( acceptable_residual_relative \* $\|b\|$, acceptable_residual_absolute )

.. index:: pair: variable; acceptable_residual_absolute
.. _doxid-structir__control__type_1a5ee0e70d90b1398019054b19b68057a0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` acceptable_residual_absolute

see acceptable_residual_relative

.. index:: pair: variable; required_residual_relative
.. _doxid-structir__control__type_1a9d3614debfe53f901305b68d9e460163:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` required_residual_relative

refinement will be judged to have failed if the residual $\|Ax-b\| \geq$ required_residual_relative \* $\|b\|$. No checking if required_residual_relative < 0

.. index:: pair: variable; record_residuals
.. _doxid-structir__control__type_1aee80ae09cedfc446424c56719b30cfca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool record_residuals

record the initial and final residual

.. index:: pair: variable; space_critical
.. _doxid-structir__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structir__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structir__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

