.. index:: pair: struct; bllsb_control_type
.. _doxid-structbllsb__control__type:

bllsb_control_type structure
----------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bllsb.h>
	
	struct bllsb_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structbllsb__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structbllsb__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structbllsb__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structbllsb__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structbllsb__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structbllsb__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structbllsb__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`infeas_max<doxid-structbllsb__control__type_infeas_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`muzero_fixed<doxid-structbllsb__control__type_muzero_fixed>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`restore_problem<doxid-structbllsb__control__type_restore_problem>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`indicator_type<doxid-structbllsb__control__type_indicator_type>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`arc<doxid-structbllsb__control__type_arc>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`series_order<doxid-structbllsb__control__type_series_order>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structbllsb__control__type_sif_file_device>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`qplib_file_device<doxid-structbllsb__control__type_qplib_file_device>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structbllsb__control__type_infinity>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_p<doxid-structbllsb__control__type_stop_abs_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_p<doxid-structbllsb__control__type_stop_rel_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_d<doxid-structbllsb__control__type_stop_abs_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_d<doxid-structbllsb__control__type_stop_rel_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_c<doxid-structbllsb__control__type_stop_abs_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_c<doxid-structbllsb__control__type_stop_rel_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`perturb_h<doxid-structbllsb__control__type_perturb_h>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`prfeas<doxid-structbllsb__control__type_prfeas>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`dufeas<doxid-structbllsb__control__type_dufeas>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`muzero<doxid-structbllsb__control__type_muzero>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`tau<doxid-structbllsb__control__type_tau>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`gamma_c<doxid-structbllsb__control__type_gamma_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`gamma_f<doxid-structbllsb__control__type_gamma_f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce_infeas<doxid-structbllsb__control__type_reduce_infeas>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_unbounded<doxid-structbllsb__control__type_obj_unbounded>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`potential_unbounded<doxid-structbllsb__control__type_potential_unbounded>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structbllsb__control__type_identical_bounds_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mu_pounce<doxid-structbllsb__control__type_mu_pounce>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`indicator_tol_p<doxid-structbllsb__control__type_indicator_tol_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`indicator_tol_pd<doxid-structbllsb__control__type_indicator_tol_pd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`indicator_tol_tapia<doxid-structbllsb__control__type_indicator_tol_tapia>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structbllsb__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structbllsb__control__type_clock_time_limit>`;
		bool :ref:`remove_dependencies<doxid-structbllsb__control__type_remove_dependencies>`;
		bool :ref:`treat_zero_bounds_as_general<doxid-structbllsb__control__type_treat_zero_bounds_as_general>`;
		bool :ref:`treat_separable_as_general<doxid-structbllsb__control__type_treat_separable_as_general>`;
		bool :ref:`just_feasible<doxid-structbllsb__control__type_just_feasible>`;
		bool :ref:`getdua<doxid-structbllsb__control__type_getdua>`;
		bool :ref:`puiseux<doxid-structbllsb__control__type_puiseux>`;
		bool :ref:`every_order<doxid-structbllsb__control__type_every_order>`;
		bool :ref:`feasol<doxid-structbllsb__control__type_feasol>`;
		bool :ref:`balance_initial_complentarity<doxid-structbllsb__control__type_balance_initial_complentarity>`;
		bool :ref:`crossover<doxid-structbllsb__control__type_crossover>`;
		bool :ref:`space_critical<doxid-structbllsb__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structbllsb__control__type_deallocate_error_fatal>`;
		bool :ref:`generate_sif_file<doxid-structbllsb__control__type_generate_sif_file>`;
		bool :ref:`generate_qplib_file<doxid-structbllsb__control__type_generate_qplib_file>`;
		char :ref:`sif_file_name<doxid-structbllsb__control__type_sif_file_name>`[31];
		char :ref:`qplib_file_name<doxid-structbllsb__control__type_qplib_file_name>`[31];
		char :ref:`prefix<doxid-structbllsb__control__type_prefix>`[31];
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>` :ref:`fdc_control<doxid-structbllsb__control__type_fdc_control>`;
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structbllsb__control__type_sbls_control>`;
		struct :ref:`fit_control_type<doxid-structfit__control__type>` :ref:`fit_control<doxid-structbllsb__control__type_fit_control>`;
		struct :ref:`roots_control_type<doxid-structroots__control__type>` :ref:`roots_control<doxid-structbllsb__control__type_roots_control>`;
		struct :ref:`cro_control_type<doxid-structcro__control__type>` :ref:`cro_control<doxid-structbllsb__control__type_cro_control>`;
	};
.. _details-structbllsb__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structbllsb__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structbllsb__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structbllsb__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structbllsb__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structbllsb__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structbllsb__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structbllsb__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; infeas_max
.. _doxid-structbllsb__control__type_infeas_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` infeas_max

the number of iterations for which the overall infeasibility of the problem is not reduced by at least a factor .reduce_infeas before the problem is flagged as infeasible (see reduce_infeas)

.. index:: pair: variable; muzero_fixed
.. _doxid-structbllsb__control__type_muzero_fixed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` muzero_fixed

the initial value of the barrier parameter will not be changed for the first muzero_fixed iterations

.. index:: pair: variable; restore_problem
.. _doxid-structbllsb__control__type_restore_problem:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are

* 0 nothing restored

* 1 scalar and vector parameters

* 2 all parameters

.. index:: pair: variable; indicator_type
.. _doxid-structbllsb__control__type_indicator_type:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` indicator_type

specifies the type of indicator function used. Possible values are

* 1 primal indicator: a constraint is active if and only if the distance to its nearest bound $\leq$.indicator_p_tol

* 2 primal-dual indicator: a constraint is active if and only if the distance to its nearest bound $\leq$.indicator_tol_pd \* size of corresponding multiplier

* 3 primal-dual indicator: a constraint is active if and only if the distance to its nearest bound $\leq$.indicator_tol_tapia \* distance to same bound at previous iteration

.. index:: pair: variable; arc
.. _doxid-structbllsb__control__type_arc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` arc

which residual trajectory should be used to aim from the current iterate to the solution. Possible values are

* 1 the Zhang linear residual trajectory

* 2 the Zhao-Sun quadratic residual trajectory

* 3 the Zhang arc ultimately switching to the Zhao-Sun residual trajectory

* 4 the mixed linear-quadratic residual trajectory

* 5 the Zhang arc ultimately switching to the mixed linear-quadratic residual trajectory

.. index:: pair: variable; series_order
.. _doxid-structbllsb__control__type_series_order:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` series_order

the order of (Taylor/Puiseux) series to fit to the path data

.. index:: pair: variable; sif_file_device
.. _doxid-structbllsb__control__type_sif_file_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; qplib_file_device
.. _doxid-structbllsb__control__type_qplib_file_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` qplib_file_device

specifies the unit number to write generated QPLIB file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structbllsb__control__type_infinity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_abs_p
.. _doxid-structbllsb__control__type_stop_abs_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_p

the required absolute and relative accuracies for the primal infeasibility

.. index:: pair: variable; stop_rel_p
.. _doxid-structbllsb__control__type_stop_rel_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_p

see stop_abs_p

.. index:: pair: variable; stop_abs_d
.. _doxid-structbllsb__control__type_stop_abs_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_d

the required absolute and relative accuracies for the dual infeasibility

.. index:: pair: variable; stop_rel_d
.. _doxid-structbllsb__control__type_stop_rel_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_d

see stop_abs_d

.. index:: pair: variable; stop_abs_c
.. _doxid-structbllsb__control__type_stop_abs_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_c

the required absolute and relative accuracies for the complementarity

.. index:: pair: variable; stop_rel_c
.. _doxid-structbllsb__control__type_stop_rel_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_c

see stop_abs_c

.. index:: pair: variable; perturb_h
.. _doxid-structbllsb__control__type_perturb_h:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` perturb_h

.perturb_h will be added to the Hessian

.. index:: pair: variable; prfeas
.. _doxid-structbllsb__control__type_prfeas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` prfeas

initial primal variables will not be closer than .prfeas from their bounds

.. index:: pair: variable; dufeas
.. _doxid-structbllsb__control__type_dufeas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` dufeas

initial dual variables will not be closer than .dufeas from their bounds

.. index:: pair: variable; muzero
.. _doxid-structbllsb__control__type_muzero:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` muzero

the initial value of the barrier parameter. If muzero is not positive, it will be reset to an appropriate value

.. index:: pair: variable; tau
.. _doxid-structbllsb__control__type_tau:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` tau

the weight attached to primal-dual infeasibility compared to complementa when assessing step acceptance

.. index:: pair: variable; gamma_c
.. _doxid-structbllsb__control__type_gamma_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` gamma_c

individual complementarities will not be allowed to be smaller than gamma_c times the average value

.. index:: pair: variable; gamma_f
.. _doxid-structbllsb__control__type_gamma_f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` gamma_f

the average complementarity will not be allowed to be smaller than gamma_f times the primal/dual infeasibility

.. index:: pair: variable; reduce_infeas
.. _doxid-structbllsb__control__type_reduce_infeas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce_infeas

if the overall infeasibility of the problem is not reduced by at least a factor .reduce_infeas over .infeas_max iterations, the problem is flagged as infeasible (see infeas_max)

.. index:: pair: variable; obj_unbounded
.. _doxid-structbllsb__control__type_obj_unbounded:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_unbounded

if the objective function value is smaller than obj_unbounded, it will be flagged as unbounded from below.

.. index:: pair: variable; potential_unbounded
.. _doxid-structbllsb__control__type_potential_unbounded:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` potential_unbounded

if W=0 and the potential function value is smaller than .potential_unbounded $\ast$ number of one-sided bounds, the analytic center will be flagged as unbounded

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structbllsb__control__type_identical_bounds_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds $(c_l,c_u)$ or $(x_l,x_u)$ that are closer than .identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; mu_pounce
.. _doxid-structbllsb__control__type_mu_pounce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mu_pounce

start terminal extrapolation when mu reaches mu_pounce

.. index:: pair: variable; indicator_tol_p
.. _doxid-structbllsb__control__type_indicator_tol_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` indicator_tol_p

if .indicator_type = 1, a constraint/bound will be deemed to be active if and only if the distance to its nearest bound $\leq$.indicator_p_tol

.. index:: pair: variable; indicator_tol_pd
.. _doxid-structbllsb__control__type_indicator_tol_pd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` indicator_tol_pd

if .indicator_type = 2, a constraint/bound will be deemed to be active if and only if the distance to its nearest bound $\leq$.indicator_tol_pd \* size of corresponding multiplier

.. index:: pair: variable; indicator_tol_tapia
.. _doxid-structbllsb__control__type_indicator_tol_tapia:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` indicator_tol_tapia

if .indicator_type = 3, a constraint/bound will be deemed to be active if and only if the distance to its nearest bound $\leq$.indicator_tol_tapia \* distance to same bound at previous iteration

.. index:: pair: variable; cpu_time_limit
.. _doxid-structbllsb__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structbllsb__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; remove_dependencies
.. _doxid-structbllsb__control__type_remove_dependencies:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool remove_dependencies

the equality constraints will be preprocessed to remove any linear dependencies if true

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structbllsb__control__type_treat_zero_bounds_as_general:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; treat_separable_as_general
.. _doxid-structbllsb__control__type_treat_separable_as_general:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool treat_separable_as_general

if .just_feasible is true, the algorithm will stop as soon as a feasible point is found. Otherwise, the optimal solution to the problem will be found

.. index:: pair: variable; just_feasible
.. _doxid-structbllsb__control__type_just_feasible:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool just_feasible

if .treat_separable_as_general, is true, any separability in the problem structure will be ignored

.. index:: pair: variable; getdua
.. _doxid-structbllsb__control__type_getdua:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool getdua

if .getdua, is true, advanced initial values are obtained for the dual variables

.. index:: pair: variable; puiseux
.. _doxid-structbllsb__control__type_puiseux:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool puiseux

decide between Puiseux and Taylor series approximations to the arc

.. index:: pair: variable; every_order
.. _doxid-structbllsb__control__type_every_order:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool every_order

try every order of series up to series_order?

.. index:: pair: variable; feasol
.. _doxid-structbllsb__control__type_feasol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasol

if .feasol is true, the final solution obtained will be perturbed so that variables close to their bounds are moved onto these bounds

.. index:: pair: variable; balance_initial_complentarity
.. _doxid-structbllsb__control__type_balance_initial_complentarity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool balance_initial_complentarity

if .balance_initial_complentarity is true, the initial complemetarity is required to be balanced

.. index:: pair: variable; crossover
.. _doxid-structbllsb__control__type_crossover:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool crossover

if .crossover is true, cross over the solution to one defined by linearly-independent constraints if possible

.. index:: pair: variable; reduced_pounce_system
.. _doxid-structbllsb__control__type_crossover:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool reduced_pounce_system

if .reduced_pounce_system is true, eliminate fixed variables when solving the linear system required by the attempted pounce to the solution

.. index:: pair: variable; space_critical
.. _doxid-structbllsb__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structbllsb__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structbllsb__control__type_generate_sif_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; generate_qplib_file
.. _doxid-structbllsb__control__type_generate_qplib_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_qplib_file

if .generate_qplib_file is .true. if a QPLIB file describing the current problem is to be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structbllsb__control__type_sif_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name of generated SIF file containing input problem

.. index:: pair: variable; qplib_file_name
.. _doxid-structbllsb__control__type_qplib_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char qplib_file_name[31]

name of generated QPLIB file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structbllsb__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structbllsb__control__type_fdc_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sbls_control
.. _doxid-structbllsb__control__type_sbls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; sbls_pounce_control
.. _doxid-structbllsb__control__type_sbls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_pounce_control

control parameters for SLS

.. index:: pair: variable; fit_control
.. _doxid-structbllsb__control__type_fit_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fit_control_type<doxid-structfit__control__type>` fit_control

control parameters for FIT

.. index:: pair: variable; roots_control
.. _doxid-structbllsb__control__type_roots_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`roots_control_type<doxid-structroots__control__type>` roots_control

control parameters for ROOTS

.. index:: pair: variable; cro_control
.. _doxid-structbllsb__control__type_cro_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`cro_control_type<doxid-structcro__control__type>` cro_control

control parameters for CRO

