.. index:: pair: struct; expo_control_type
.. _doxid-structexpo__control__type:

expo_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct expo_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          print_gap::INT
          max_it::INT
          max_eval::INT
          alive_unit::INT
          alive_file::NTuple{31,Cchar}
          update_multipliers_itmin::INT
          update_multipliers_tol::T
          infinity::T
          stop_abs_p::T
          stop_rel_p::T
          stop_abs_d::T
          stop_rel_d::T
          stop_abs_c::T
          stop_rel_c::T
          stop_s::T
          initial_mu::T
          mu_reduce::T
          minimum_weight::T
          obj_unbounded::T
          try_advanced_start::T
          try_sqp_start::T
          stop_advanced_start::T
          cpu_time_limit::T
          clock_time_limit::T
          hessian_available::Bool
          subproblem_direct::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}
          bsc_control::bsc_control_type
          tru_control::tru_control_type{T,INT}
          ssls_control::ssls_control_type{T,INT}

.. _details-structexpo__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structexpo__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structexpo__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structexpo__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structexpo__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structexpo__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structexpo__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structexpo__control__type_print_gap:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_gap

the number of iterations between printing

.. index:: pair: variable; max_it
.. _doxid-structexpo__control__type_max_it:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_it

the maximum number of iterations permitted

.. index:: pair: variable; max_eval
.. _doxid-structexpo__control__type_max_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_eval

the maximum number of function evaluations permitted

.. index:: pair: variable; alive_unit
.. _doxid-structexpo__control__type_alive_unit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structexpo__control__type_alive_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; update_multipliers_itmin
.. _doxid-structexpo__control__type_update_multipliers_itmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT update_multipliers_itmin

update the Lagrange multipliers/dual variables from iteration .update_multipliers_itmin (<0 means never) and once the primal infeasibility is below .update_multipliers_tol

.. index:: pair: variable; update_multipliers_tol
.. _doxid-structexpo__control__type_update_multipliers_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T  update_multipliers_tol

see update_multipliers_itmin

.. index:: pair: variable; infinity
.. _doxid-structtrb__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

.. index:: pair: variable; stop_abs_p
.. _doxid-structexpo__control__type_stop_abs_p:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_abs_p

the required absolute and relative accuracies for the primal infeasibility (1)

.. index:: pair: variable; stop_rel_p
.. _doxid-structexpo__control__type_stop_rel_p:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_rel_p

see stop_abs_p

.. index:: pair: variable; stop_abs_d
.. _doxid-structexpo__control__type_stop_abs_d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_abs_d

the required absolute and relative accuracies for the dual infeasibility (2)

.. index:: pair: variable; stop_rel_d
.. _doxid-structexpo__control__type_stop_rel_d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_rel_d

see stop_abs_d

.. index:: pair: variable; stop_abs_c
.. _doxid-structexpo__control__type_stop_abs_c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_abs_c

the required absolute and relative accuracies for the complementary slackness (3)

.. index:: pair: variable; stop_rel_c
.. _doxid-structexpo__control__type_stop_rel_c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_rel_c

see stop_abs_c

.. index:: pair: variable; stop_s
.. _doxid-structexpo__control__type_stop_s:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_s

the smallest the norm of the step can be before termination

.. index:: pair: variable; initial_mu
.. _doxid-structexpo__control__type_initial_mu:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T initial_mu

initial value for the penalty parameter (<=0 means set automatically)

.. index:: pair: variable; mu_reduce
.. _doxid-structexpo__control__type_mu_reduce:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mu_reduce

the amount by which the penalty parameter is decreased

.. index:: pair: variable; obj_unbounded
.. _doxid-structtrb__control__type_obj_unbounded:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; try_advanced_start
.. _doxid-structexpo__control__type_try_advanced_start:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T try_advanced_start

try an advanced start at the end of every iteration when the KKT residuals are smaller than .try_advanced_start (-ve means never)

.. index:: pair: variable; try_sqp_start
.. _doxid-structexpo__control__type_try_sqp_start:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T try_sqp_start

try an advanced SQP start at the end of every iteration when the KKT residuals are smaller than .try_sqp_start (-ve means never)

.. index:: pair: variable; stop_advanced_start
.. _doxid-structexpo__control__type_stop_advanced_start:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_advanced_start

stop the advanced start search once the residuals small tham .stop_advanced_start

.. index:: pair: variable; cpu_time_limit
.. _doxid-structexpo__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structexpo__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structtrb__control__type_hessian_available:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products (coming soon)?

.. index:: pair: variable; subproblem_direct
.. _doxid-structexpo__control__type_subproblem_direct:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method (coming soon) to find the search direction

.. index:: pair: variable; space_critical
.. _doxid-structexpo__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structexpo__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structexpo__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; bsc_control
.. _doxid-structexpo__control__type_bsc_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`bsc_control_type<doxid-structbsc__control__type>` bsc_control

control parameters for BSC

.. index:: pair: variable; tru_control
.. _doxid-structexpo__control__type_tru_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`tru_control_type<doxid-structtru__control__type>` tru_control

control parameters for TRU

.. index:: pair: variable; ssls_control
.. _doxid-structexpo__control__type_ssls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ssls_control_type<doxid-structssls__control__type>` ssls_control

control parameters for SSLS
