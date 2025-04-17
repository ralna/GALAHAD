.. index:: pair: table; trs_control_type
.. _doxid-structtrs__control__type:

trs_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct trs_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          problem::INT
          print_level::INT
          dense_factorization::INT
          new_h::INT
          new_m::INT
          new_a::INT
          max_factorizations::INT
          inverse_itmax::INT
          taylor_max_degree::INT
          initial_multiplier::T
          lower::T
          upper::T
          stop_normal::T
          stop_absolute_normal::T
          stop_hard::T
          start_invit_tol::T
          start_invitmax_tol::T
          equality_problem::Bool
          use_initial_multiplier::Bool
          initialize_approx_eigenvector::Bool
          force_Newton::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          problem_file::NTuple{31,Cchar}
          symmetric_linear_solver::NTuple{31,Cchar}
          definite_linear_solver::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sls_control::sls_control_type{T,INT}
          ir_control::ir_control_type{T,INT}

.. _details-structtrs__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structtrs__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structtrs__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structtrs__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; problem
.. _doxid-structtrs__control__type_problem:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT problem

unit to write problem data into file problem_file

.. index:: pair: variable; print_level
.. _doxid-structtrs__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; dense_factorization
.. _doxid-structtrs__control__type_dense_factorization:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT dense_factorization

should the problem be solved by dense factorization? Possible values are

* 0 sparse factorization will be used

* 1 dense factorization will be used

* other the choice is made automatically depending on the dimension & sparsity

.. index:: pair: variable; new_h
.. _doxid-structtrs__control__type_new_h:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_h

how much of $H$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_m
.. _doxid-structtrs__control__type_new_m:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_m

how much of $M$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_a
.. _doxid-structtrs__control__type_new_a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_a

how much of $A$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; max_factorizations
.. _doxid-structtrs__control__type_max_factorizations:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_factorizations

the maximum number of factorizations (=iterations) allowed. -ve implies no limit

.. index:: pair: variable; inverse_itmax
.. _doxid-structtrs__control__type_inverse_itmax:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT inverse_itmax

the number of inverse iterations performed in the "maybe hard" case

.. index:: pair: variable; taylor_max_degree
.. _doxid-structtrs__control__type_taylor_max_degree:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT taylor_max_degree

maximum degree of Taylor approximant allowed

.. index:: pair: variable; initial_multiplier
.. _doxid-structtrs__control__type_initial_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T initial_multiplier

initial estimate of the Lagrange multipler

.. index:: pair: variable; lower
.. _doxid-structtrs__control__type_lower:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lower

lower and upper bounds on the multiplier, if known

.. index:: pair: variable; upper
.. _doxid-structtrs__control__type_upper:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T upper

see lower

.. index:: pair: variable; stop_normal
.. _doxid-structtrs__control__type_stop_normal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_normal

stop when $| ||x|| - radius | \leq$ max( stop_normal \* radius, stop_absolute_normal )

.. index:: pair: variable; stop_absolute_normal
.. _doxid-structtrs__control__type_stop_absolute_normal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_absolute_normal

see stop_normal

.. index:: pair: variable; stop_hard
.. _doxid-structtrs__control__type_stop_hard:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_hard

stop when bracket on optimal multiplier $\leq$ stop_hard \* max( bracket ends )

.. index:: pair: variable; start_invit_tol
.. _doxid-structtrs__control__type_start_invit_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T start_invit_tol

start inverse iteration when bracket on optimal multiplier $\leq$ stop_start_invit_tol \* max( bracket ends )

.. index:: pair: variable; start_invitmax_tol
.. _doxid-structtrs__control__type_start_invitmax_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T start_invitmax_tol

start full inverse iteration when bracket on multiplier $\leq$ stop_start_invitmax_tol \* max( bracket ends)

.. index:: pair: variable; equality_problem
.. _doxid-structtrs__control__type_equality_problem:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool equality_problem

is the solution is <b<required to lie on the boundary (i.e., is the constraint an equality)?

.. index:: pair: variable; use_initial_multiplier
.. _doxid-structtrs__control__type_use_initial_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool use_initial_multiplier

ignore initial_multiplier?

.. index:: pair: variable; initialize_approx_eigenvector
.. _doxid-structtrs__control__type_initialize_approx_eigenvector:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool initialize_approx_eigenvector

should a suitable initial eigenvector should be chosen or should a previous eigenvector may be used?

.. index:: pair: variable; force_Newton
.. _doxid-structtrs__control__type_force_Newton:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool force_Newton

ignore the trust-region if $H$ is positive definite

.. index:: pair: variable; space_critical
.. _doxid-structtrs__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structtrs__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; problem_file
.. _doxid-structtrs__control__type_problem_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char problem_file[31]

name of file into which to write problem data

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structtrs__control__type_symmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

symmetric (indefinite) linear equation solver

.. index:: pair: variable; definite_linear_solver
.. _doxid-structtrs__control__type_definite_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

definite linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structtrs__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structtrs__control__type_sls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for the Cholesky factorization and solution (see sls_c documentation)

.. index:: pair: variable; ir_control
.. _doxid-structtrs__control__type_ir_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_control_type<doxid-structir__control__type>` ir_control

control parameters for iterative refinement (see ir_c documentation)

