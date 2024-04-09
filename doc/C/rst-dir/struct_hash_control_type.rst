.. index:: pair: table; hash_control_type
.. _doxid-structhash__control__type:

hash_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_hash.h>
	
	struct hash_control_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structhash__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structhash__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structhash__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		bool :ref:`space_critical<doxid-structhash__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structhash__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structhash__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structhash__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; error
.. _doxid-structhash__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structhash__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structhash__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required. Possible values are:

* $\leq$ 0 no output,

* $\geq$ 1 debugging

.. index:: pair: variable; space_critical
.. _doxid-structhash__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structhash__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structhash__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

