.. index:: pair: table; sha_control_type
.. _doxid-structsha__control__type:

sha_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sha_control_type{INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          approximation_algorithm::INT
          dense_linear_solver::INT
          extra_differences::INT
          sparse_row::INT
          recursion_max::INT
          recursion_entries_required::INT
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}

.. _details-structsha__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsha__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsha__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structsha__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structsha__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required. <= 0 gives no output, = 1 gives a one-line summary for every iteration, = 2 gives a summary of the inner iteration for each iteration, >= 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; approximation_algorithm
.. _doxid-structsha__control__type_approximation_algorithm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT approximation_algorithm

which approximation algorithm should be used?

* 1 : unsymmetric, parallel (Alg 2.1 in paper)

* 2 : symmetric (Alg 2.2 in pape)

* 3 : composite, parallel (Alg 2.3 in paper)

* 4 : composite, block parallel (Alg 2.4 in paper)

.. index:: pair: variable; dense_linear_solver
.. _doxid-structsha__control__type_dense_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT dense_linear_solver

which dense linear equation solver should be used?

* 1 : Gaussian elimination

* 2 : QR factorization

* 3 : singular-value decomposition

* 4 : singular-value decomposition with divide-and-conquer

.. index:: pair: variable; extra_differences
.. _doxid-structsha__control__type_extra_differences:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT extra_differences

if available use an addition extra_differences differences

.. index:: pair: variable; sparse_row
.. _doxid-structsha__control__type_sparse_row:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sparse_row

a row is considered sparse if it has no more than .sparse_row entries

.. index:: pair: variable; recursion_max
.. _doxid-structsha__control__type_recursion_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        :ref:`ipc_<doxid-galahad__ipc_8h_>` recursion_max

limit on the maximum number of levels of recursion (Alg. 2.4)

.. index:: pair: variable; recursion_entries_required
.. _doxid-structsha__control__type_recursion_entries_required:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        :ref:`ipc_<doxid-galahad__ipc_8h_>` recursion_entries_required

the minimum number of entries in a reduced row that are required
if a further level of recuresion is allowed (Alg. 2.4)

.. index:: pair: variable; space_critical
.. _doxid-structsha__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structsha__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structsha__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'
