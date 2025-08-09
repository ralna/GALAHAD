.. index:: pair: struct; expo_control_type
.. _doxid-structexpo__control__type:

expo_control_type structure
---------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_expo.h>
	
	struct expo_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structexpo__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structexpo__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structexpo__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structexpo__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structexpo__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structexpo__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structexpo__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_it<doxid-structexpo__control__type_max_it>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_eval<doxid-structexpo__control__type_max_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structexpo__control__type_alive_unit>`;
		char :ref:`alive_file<doxid-structexpo__control__type_alive_file>`[31];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`update_multipliers_itmin<doxid-structexpo__control__type_update_multipliers_itmin>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`update_multipliers_tol<doxid-structexpo__control__type_update_multipliers_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structtrb__control__type_infinity>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_p<doxid-structexpo__control__type_stop_abs_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_p<doxid-structexpo__control__type_stop_rel_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_d<doxid-structexpo__control__type_stop_abs_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_d<doxid-structexpo__control__type_stop_rel_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_c<doxid-structexpo__control__type_stop_abs_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_c<doxid-structexpo__control__type_stop_rel_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_s<doxid-structexpo__control__type_stop_s>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_subproblem_rel<doxid-structexpo__control__type_stop_subproblem_rel>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_mu<doxid-structexpo__control__type_initial_mu>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mu_reduce<doxid-structexpo__control__type_mu_reduce>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_unbounded<doxid-structtrb__control__type_obj_unbounded>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`try_advanced_start<doxid-structtrb__control__type_try_advanced_start>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`try_sqp_start<doxid-structtrb__control__type_try_sqp_start>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_advanced_start<doxid-structtrb__control__type_stop_advanced_start>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structexpo__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structexpo__control__type_clock_time_limit>`;
		bool :ref:`hessian_available<doxid-structtrb__control__type_hessian_available>`;
		bool :ref:`subproblem_direct<doxid-structexpo__control__type_subproblem_direct>`;
		bool :ref:`space_critical<doxid-structexpo__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structexpo__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structexpo__control__type_prefix>`[31];
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>` :ref:`bsc_control<doxid-structexpo__control__type_bsc_control>`;
		struct :ref:`tru_control_type<doxid-structtru__control__type>` :ref:`tru_control<doxid-structexpo__control__type_tru_control>`;
		struct :ref:`ssls_control_type<doxid-structssls__control__type>` :ref:`ssls_control<doxid-structexpo__control__type_ssls_control>`;
	};
.. _details-structexpo__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structexpo__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structexpo__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structexpo__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structexpo__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structexpo__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structexpo__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structexpo__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; max_it
.. _doxid-structexpo__control__type_max_it:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_it

the maximum number of iterations permitted

.. index:: pair: variable; max_eval
.. _doxid-structexpo__control__type_max_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_eval

the maximum number of function evaluations permitted

.. index:: pair: variable; alive_unit
.. _doxid-structexpo__control__type_alive_unit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structexpo__control__type_alive_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; update_multipliers_itmin
.. _doxid-structexpo__control__type_update_multipliers_itmin:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` update_multipliers_itmin

update the Lagrange multipliers/dual variables from iteration .update_multipliers_itmin (<0 means never) and once the primal infeasibility is below .update_multipliers_tol

.. index:: pair: variable; update_multipliers_tol
.. _doxid-structexpo__control__type_update_multipliers_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` update_multipliers_tol

see update_multipliers_itmin

.. index:: pair: variable; infinity
.. _doxid-structtrb__control__type_infinity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_abs_p
.. _doxid-structexpo__control__type_stop_abs_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_p

the required absolute and relative accuracies for the primal infeasibility

.. index:: pair: variable; stop_rel_p
.. _doxid-structexpo__control__type_stop_rel_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_p

see stop_abs_p

.. index:: pair: variable; stop_abs_d
.. _doxid-structexpo__control__type_stop_abs_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_d

the required absolute and relative accuracies for the dual infeasibility

.. index:: pair: variable; stop_rel_d
.. _doxid-structexpo__control__type_stop_rel_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_d

see stop_abs_d

.. index:: pair: variable; stop_abs_c
.. _doxid-structexpo__control__type_stop_abs_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_c

the required absolute and relative accuracies for complementary slackness

.. index:: pair: variable; stop_rel_c
.. _doxid-structexpo__control__type_stop_rel_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_c

see stop_abs_c

.. index:: pair: variable; stop_s
.. _doxid-structexpo__control__type_stop_s:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_s

the smallest the norm of the step can be before termination

.. index:: pair: variable; stop_subproblem_rel
.. _doxid-structexpo__control__type_stop_subproblem_rel:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_subproblem_rel

the subproblem minimization that uses GALAHAD TRU will be stopped as soon as the relative decrease in the subproblem gradient falls below .stop_subproblem_rel. If .stop_subproblem_rel is 1.0 or bigger or 0.0 or smaller, this value will be ignored, and the choice of stopping rule delegated to .control_tru.stop_g_relative (see below)

.. index:: pair: variable; initial_mu
.. _doxid-structexpo__control__type_initial_mu:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_mu

initial value for the penalty parameter (<=0 means set automatically)

.. index:: pair: variable; mu_reduce
.. _doxid-structexpo__control__type_mu_reduce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mu_reduce

the amount by which the penalty parameter is decreased

.. index:: pair: variable; obj_unbounded
.. _doxid-structtrb__control__type_obj_unbounded:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; try_advanced_start
.. _doxid-structtrb__control__type_try_advanced_start:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` try_advanced_start

try an advanced start at the end of every iteration when the KKT residuals are smaller than .try_advanced_start (-ve means never)

.. index:: pair: variable; try_sqp_start
.. _doxid-structtrb__control__type_try_sqp_start:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` try_sqp_start

try an advanced SQP start at the end of every iteration when the KKT residuals are smaller than .try_sqp_start (-ve means never)

.. index:: pair: variable; stop_advanced_start
.. _doxid-structtrb__control__type_stop_advanced_start:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_advanced_start

stop the advanced start search once the residuals small tham .stop_advanced_start

.. index:: pair: variable; cpu_time_limit
.. _doxid-structexpo__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structexpo__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structtrb__control__type_hessian_available:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products (coming soon)?

.. index:: pair: variable; subproblem_direct
.. _doxid-structexpo__control__type_subproblem_direct:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method (coming soon) to find the search direction

.. index:: pair: variable; space_critical
.. _doxid-structexpo__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structexpo__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structexpo__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; bsc_control
.. _doxid-structexpo__control__type_bsc_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bsc_control_type<doxid-structbsc__control__type>` bsc_control

control parameters for BSC

.. index:: pair: variable; tru_control
.. _doxid-structexpo__control__type_tru_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`tru_control_type<doxid-structtru__control__type>` tru_control

control parameters for TRU

.. index:: pair: variable; ssls_control
.. _doxid-structexpo__control__type_ssls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ssls_control_type<doxid-structssls__control__type>` ssls_control

control parameters for SSLS
