.. index:: pair: struct; qpa_control_type
.. _doxid-structqpa__control__type:

qpa_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct qpa_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          maxit::INT
          factor::INT
          max_col::INT
          max_sc::INT
          indmin::INT
          valmin::INT
          itref_max::INT
          infeas_check_interval::INT
          cg_maxit::INT
          precon::INT
          nsemib::INT
          full_max_fill::INT
          deletion_strategy::INT
          restore_problem::INT
          monitor_residuals::INT
          cold_start::INT
          sif_file_device::INT
          infinity::T
          feas_tol::T
          obj_unbounded::T
          increase_rho_g_factor::T
          infeas_g_improved_by_factor::T
          increase_rho_b_factor::T
          infeas_b_improved_by_factor::T
          pivot_tol::T
          pivot_tol_for_dependencies::T
          zero_pivot::T
          inner_stop_relative::T
          inner_stop_absolute::T
          multiplier_tol::T
          cpu_time_limit::T
          clock_time_limit::T
          treat_zero_bounds_as_general::Bool
          solve_qp::Bool
          solve_within_bounds::Bool
          randomize::Bool
          array_syntax_worse_than_do_loop::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          generate_sif_file::Bool
          symmetric_linear_solver::NTuple{31,Cchar}
          sif_file_name::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          each_interval::Bool
          sls_control::sls_control_type{T,INT}

.. _details-structqpa__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structqpa__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structqpa__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structqpa__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structqpa__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structqpa__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structqpa__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structqpa__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; factor
.. _doxid-structqpa__control__type_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factor

the factorization to be used. Possible values are 0 automatic 1 Schur-complement factorization 2 augmented-system factorization

.. index:: pair: variable; max_col
.. _doxid-structqpa__control__type_max_col:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_col

the maximum number of nonzeros in a column of A which is permitted with the Schur-complement factorization

.. index:: pair: variable; max_sc
.. _doxid-structqpa__control__type_max_sc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_sc

the maximum permitted size of the Schur complement before a refactorization is performed

.. index:: pair: variable; indmin
.. _doxid-structqpa__control__type_indmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT indmin

an initial guess as to the integer workspace required by SLS (OBSOLETE)

.. index:: pair: variable; valmin
.. _doxid-structqpa__control__type_valmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT valmin

an initial guess as to the real workspace required by SLS (OBSOLETE)

.. index:: pair: variable; itref_max
.. _doxid-structqpa__control__type_itref_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT itref_max

the maximum number of iterative refinements allowed (OBSOLETE)

.. index:: pair: variable; infeas_check_interval
.. _doxid-structqpa__control__type_infeas_check_interval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT infeas_check_interval

the infeasibility will be checked for improvement every infeas_check_interval iterations (see infeas_g_improved_by_factor and infeas_b_improved_by_factor below)

.. index:: pair: variable; cg_maxit
.. _doxid-structqpa__control__type_cg_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_maxit

the maximum number of CG iterations allowed. If cg_maxit < 0, this number will be reset to the dimension of the system + 1

.. index:: pair: variable; precon
.. _doxid-structqpa__control__type_precon:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT precon

the preconditioner to be used for the CG is defined by precon. Possible values are 0 automatic 1 no preconditioner, i.e, the identity within full factorization 2 full factorization 3 band within full factorization 4 diagonal using the barrier terms within full factorization

.. index:: pair: variable; nsemib
.. _doxid-structqpa__control__type_nsemib:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nsemib

the semi-bandwidth of a band preconditioner, if appropriate

.. index:: pair: variable; full_max_fill
.. _doxid-structqpa__control__type_full_max_fill:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT full_max_fill

if the ratio of the number of nonzeros in the factors of the reference matrix to the number of nonzeros in the matrix itself exceeds full_max_fill, and the preconditioner is being selected automatically (precon = 0), a banded approximation will be used instead

.. index:: pair: variable; deletion_strategy
.. _doxid-structqpa__control__type_deletion_strategy:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT deletion_strategy

the constraint deletion strategy to be used. Possible values are:

0 most violated of all 1 LIFO (last in, first out) k LIFO(k) most violated of the last k in LIFO

.. index:: pair: variable; restore_problem
.. _doxid-structqpa__control__type_restore_problem:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are 0 nothing restored 1 scalar and vector parameters 2 all parameters

.. index:: pair: variable; monitor_residuals
.. _doxid-structqpa__control__type_monitor_residuals:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT monitor_residuals

the frequency at which residuals will be monitored

.. index:: pair: variable; cold_start
.. _doxid-structqpa__control__type_cold_start:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cold_start

indicates whether a cold or warm start should be made. Possible values are

0 warm start - the values set in C_stat and B_stat indicate which constraints will be included in the initial working set. 1 cold start from the value set in X; constraints active at X will determine the initial working set. 2 cold start with no active constraints 3 cold start with only equality constraints active 4 cold start with as many active constraints as possible

.. index:: pair: variable; sif_file_device
.. _doxid-structqpa__control__type_sif_file_device:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structqpa__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; feas_tol
.. _doxid-structqpa__control__type_feas_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T feas_tol

any constraint violated by less than feas_tol will be considered to be satisfied

.. index:: pair: variable; obj_unbounded
.. _doxid-structqpa__control__type_obj_unbounded:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_unbounded

if the objective function value is smaller than obj_unbounded, it will be flagged as unbounded from below.

.. index:: pair: variable; increase_rho_g_factor
.. _doxid-structqpa__control__type_increase_rho_g_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T increase_rho_g_factor

if the problem is currently infeasible and solve_qp (see below) is .TRUE. the current penalty parameter for the general constraints will be increased by increase_rho_g_factor when needed

.. index:: pair: variable; infeas_g_improved_by_factor
.. _doxid-structqpa__control__type_infeas_g_improved_by_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infeas_g_improved_by_factor

if the infeasibility of the general constraints has not dropped by a fac of infeas_g_improved_by_factor over the previous infeas_check_interval iterations, the current corresponding penalty parameter will be increase

.. index:: pair: variable; increase_rho_b_factor
.. _doxid-structqpa__control__type_increase_rho_b_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T increase_rho_b_factor

if the problem is currently infeasible and solve_qp or solve_within_boun (see below) is .TRUE., the current penalty parameter for the simple boun constraints will be increased by increase_rho_b_factor when needed

.. index:: pair: variable; infeas_b_improved_by_factor
.. _doxid-structqpa__control__type_infeas_b_improved_by_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infeas_b_improved_by_factor

if the infeasibility of the simple bounds has not dropped by a factor of infeas_b_improved_by_factor over the previous infeas_check_interval iterations, the current corresponding penalty parameter will be increase

.. index:: pair: variable; pivot_tol
.. _doxid-structqpa__control__type_pivot_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol

the threshold pivot used by the matrix factorization. See the documentation for SLS for details (OBSOLE

.. index:: pair: variable; pivot_tol_for_dependencies
.. _doxid-structqpa__control__type_pivot_tol_for_dependencies:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol_for_dependencies

the threshold pivot used by the matrix factorization when attempting to detect linearly dependent constraints.

.. index:: pair: variable; zero_pivot
.. _doxid-structqpa__control__type_zero_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_pivot

any pivots smaller than zero_pivot in absolute value will be regarded to zero when attempting to detect linearly dependent constraints (OBSOLE

.. index:: pair: variable; inner_stop_relative
.. _doxid-structqpa__control__type_inner_stop_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_stop_relative

the search direction is considered as an acceptable approximation to the minimizer of the model if the gradient of the model in the preconditioning(inverse) norm is less than max( inner_stop_relative \* initial preconditioning(inverse) gradient norm, inner_stop_absolute )

.. index:: pair: variable; inner_stop_absolute
.. _doxid-structqpa__control__type_inner_stop_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_stop_absolute

see inner_stop_relative

.. index:: pair: variable; multiplier_tol
.. _doxid-structqpa__control__type_multiplier_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier_tol

any dual variable or Lagrange multiplier which is less than multiplier_t outside its optimal interval will be regarded as being acceptable when checking for optimality

.. index:: pair: variable; cpu_time_limit
.. _doxid-structqpa__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structqpa__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structqpa__control__type_treat_zero_bounds_as_general:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; solve_qp
.. _doxid-structqpa__control__type_solve_qp:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool solve_qp

if solve_qp is .TRUE., the value of prob.rho_g and prob.rho_b will be increased as many times as are needed to ensure that the output solution is feasible, and thus aims to solve the quadratic program (2)-(4)

.. index:: pair: variable; solve_within_bounds
.. _doxid-structqpa__control__type_solve_within_bounds:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool solve_within_bounds

if solve_within_bounds is .TRUE., the value of prob.rho_b will be increased as many times as are needed to ensure that the output solution is feasible with respect to the simple bounds, and thus aims to solve the bound-constrained quadratic program (4)-(5)

.. index:: pair: variable; randomize
.. _doxid-structqpa__control__type_randomize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool randomize

if randomize is .TRUE., the constraint bounds will be perturbed by small random quantities during the first stage of the solution process. Any randomization will ultimately be removed. Randomization helps when solving degenerate problems

.. index:: pair: variable; array_syntax_worse_than_do_loop
.. _doxid-structqpa__control__type_array_syntax_worse_than_do_loop:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool array_syntax_worse_than_do_loop

if .array_syntax_worse_than_do_loop is true, f77-style do loops will be used rather than f90-style array syntax for vector operations (OBSOLETE)

.. index:: pair: variable; space_critical
.. _doxid-structqpa__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structqpa__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structqpa__control__type_generate_sif_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structqpa__control__type_symmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

indefinite linear equation solver

.. index:: pair: variable; sif_file_name
.. _doxid-structqpa__control__type_sif_file_name:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} sif_file_name

definite linear equation solver

name of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structqpa__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; each_interval
.. _doxid-structqpa__control__type_each_interval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool each_interval

component specifically for parametric problems (not used at present)

.. index:: pair: variable; sls_control
.. _doxid-structqpa__control__type_sls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

