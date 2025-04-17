.. index:: pair: table; sha_control_type
.. _doxid-structsha__control__type:

sha_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sha.h>

	struct sha_control_type {
		// fields

		bool :ref:`f_indexing<doxid-structsha__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structsha__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structsha__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structsha__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`approximation_algorithm<doxid-structsha__control__type_approximation_algorithm>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dense_linear_solver<doxid-structsha__control__type_dense_linear_solver>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`extra_differences<doxid-structsha__control__type_extra_differences>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sparse_row<doxid-structsha__control__type_sparse_row>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`recursion_max<doxid-structsha__control__type_recursion_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`recursion_entries_required<doxid-structsha__control__type_recursion_entries_required>`;
		bool :ref:`average_off_diagonals<doxid-structsha__control__type_1a957fc1f4f26eeef3b0951791ff972e8e>`;
		bool :ref:`space_critical<doxid-structsha__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structsha__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structsha__control__type_prefix>`[31];
	};
.. _details-structsha__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsha__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsha__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structsha__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structsha__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required. <= 0 gives no output, = 1 gives a one-line summary for every iteration, = 2 gives a summary of the inner iteration for each iteration, >= 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; approximation_algorithm
.. _doxid-structsha__control__type_approximation_algorithm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` approximation_algorithm

which approximation algorithm should be used?

* 1 : unsymmetric, parallel (Alg 2.1 in paper)

* 2 : symmetric (Alg 2.2 in pape)

* 3 : composite, parallel (Alg 2.3 in paper)

* 4 : composite, block parallel (Alg 2.4 in paper)

.. index:: pair: variable; dense_linear_solver
.. _doxid-structsha__control__type_dense_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dense_linear_solver

which dense linear equation solver should be used?

* 1 : Gaussian elimination

* 2 : QR factorization

* 3 : singular-value decomposition

* 4 : singular-value decomposition with divide-and-conquer

.. index:: pair: variable; extra_differences
.. _doxid-structsha__control__type_extra_differences:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` extra_differences

if available use an addition extra_differences differences

.. index:: pair: variable; sparse_row
.. _doxid-structsha__control__type_sparse_row:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sparse_row

a row is considered sparse if it has no more than .sparse_row entries

.. index:: pair: variable; recursion_max
.. _doxid-structsha__control__type_recursion_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

        :ref:`ipc_<doxid-galahad__ipc_8h_>` recursion_max

limit on the maximum number of levels of recursion (Alg. 2.4)

.. index:: pair: variable; recursion_entries_required
.. _doxid-structsha__control__type_recursion_entries_required:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

        :ref:`ipc_<doxid-galahad__ipc_8h_>` recursion_entries_required

the minimum number of entries in a reduced row that are required
if a further level of recuresion is allowed (Alg. 2.4)

.. index:: pair: variable; average_off_diagonals
.. _doxid-structsha__control__type_1a957fc1f4f26eeef3b0951791ff972e8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool average_off_diagonals

select if pairs of off-diagonal Hessian estimates are to be averaged on return.
Otherwise pick the value from the upper triangle

.. index:: pair: variable; space_critical
.. _doxid-structsha__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structsha__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structsha__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'
