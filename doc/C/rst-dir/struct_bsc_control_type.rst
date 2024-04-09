.. index:: pair: table; bsc_control_type
.. _doxid-structbsc__control__type:

bsc_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bsc.h>
	
	struct bsc_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structbsc__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structbsc__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structbsc__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structbsc__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col<doxid-structbsc__control__type_1abca2db33b9520095e98790d45a1be93f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_a<doxid-structbsc__control__type_1a7bea45d51fd9384037bbbf82f7750ce6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`extra_space_s<doxid-structbsc__control__type_1ad1d6fbcd01c19f28d44ca8ba150efce5>`;
		bool :ref:`s_also_by_column<doxid-structbsc__control__type_1abad1637e3128deb63e2a8eab714e5ffd>`;
		bool :ref:`space_critical<doxid-structbsc__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structbsc__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structbsc__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structbsc__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structbsc__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structbsc__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structbsc__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structbsc__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; max_col
.. _doxid-structbsc__control__type_1abca2db33b9520095e98790d45a1be93f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col

maximum permitted number of nonzeros in a column of $A$; -ve means unlimit

.. index:: pair: variable; new_a
.. _doxid-structbsc__control__type_1a7bea45d51fd9384037bbbf82f7750ce6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_a

how much has $A$ changed since it was last accessed:

* 0 = not changed,

* 1 = values changed,

* 2 = structure changed

* 3 = structure changed but values not required

.. index:: pair: variable; extra_space_s
.. _doxid-structbsc__control__type_1ad1d6fbcd01c19f28d44ca8ba150efce5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` extra_space_s

how much extra space is to be allocated in $S$ above that needed to hold the Schur complement

.. index:: pair: variable; s_also_by_column
.. _doxid-structbsc__control__type_1abad1637e3128deb63e2a8eab714e5ffd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool s_also_by_column

should s.ptr also be set to indicate the first entry in each column of $S$

.. index:: pair: variable; space_critical
.. _doxid-structbsc__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structbsc__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structbsc__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

