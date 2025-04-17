.. index:: pair: struct; lsqp_control_type
.. _doxid-structlsqp__control__type:

lsqp_control_type structure
---------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lsqp.h>
	
	struct lsqp_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structlsqp__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structlsqp__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structlsqp__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structlsqp__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structlsqp__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structlsqp__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structlsqp__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factor<doxid-structlsqp__control__type_factor>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col<doxid-structlsqp__control__type_max_col>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`indmin<doxid-structlsqp__control__type_indmin>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`valmin<doxid-structlsqp__control__type_valmin>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itref_max<doxid-structlsqp__control__type_itref_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`infeas_max<doxid-structlsqp__control__type_infeas_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`muzero_fixed<doxid-structlsqp__control__type_muzero_fixed>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`restore_problem<doxid-structlsqp__control__type_restore_problem>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`indicator_type<doxid-structlsqp__control__type_indicator_type>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`extrapolate<doxid-structlsqp__control__type_extrapolate>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`path_history<doxid-structlsqp__control__type_path_history>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`path_derivatives<doxid-structlsqp__control__type_path_derivatives>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`fit_order<doxid-structlsqp__control__type_fit_order>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structlsqp__control__type_sif_file_device>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structlsqp__control__type_infinity>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_p<doxid-structlsqp__control__type_stop_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_d<doxid-structlsqp__control__type_stop_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_c<doxid-structlsqp__control__type_stop_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`prfeas<doxid-structlsqp__control__type_prfeas>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`dufeas<doxid-structlsqp__control__type_dufeas>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`muzero<doxid-structlsqp__control__type_muzero>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce_infeas<doxid-structlsqp__control__type_reduce_infeas>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`potential_unbounded<doxid-structlsqp__control__type_potential_unbounded>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol<doxid-structlsqp__control__type_pivot_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol_for_dependencies<doxid-structlsqp__control__type_pivot_tol_for_dependencies>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_pivot<doxid-structlsqp__control__type_zero_pivot>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structlsqp__control__type_identical_bounds_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mu_min<doxid-structlsqp__control__type_mu_min>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`indicator_tol_p<doxid-structlsqp__control__type_indicator_tol_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`indicator_tol_pd<doxid-structlsqp__control__type_indicator_tol_pd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`indicator_tol_tapia<doxid-structlsqp__control__type_indicator_tol_tapia>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structlsqp__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structlsqp__control__type_clock_time_limit>`;
		bool :ref:`remove_dependencies<doxid-structlsqp__control__type_remove_dependencies>`;
		bool :ref:`treat_zero_bounds_as_general<doxid-structlsqp__control__type_treat_zero_bounds_as_general>`;
		bool :ref:`just_feasible<doxid-structlsqp__control__type_just_feasible>`;
		bool :ref:`getdua<doxid-structlsqp__control__type_getdua>`;
		bool :ref:`puiseux<doxid-structlsqp__control__type_puiseux>`;
		bool :ref:`feasol<doxid-structlsqp__control__type_feasol>`;
		bool :ref:`balance_initial_complentarity<doxid-structlsqp__control__type_balance_initial_complentarity>`;
		bool :ref:`use_corrector<doxid-structlsqp__control__type_use_corrector>`;
		bool :ref:`array_syntax_worse_than_do_loop<doxid-structlsqp__control__type_array_syntax_worse_than_do_loop>`;
		bool :ref:`space_critical<doxid-structlsqp__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structlsqp__control__type_deallocate_error_fatal>`;
		bool :ref:`generate_sif_file<doxid-structlsqp__control__type_generate_sif_file>`;
		char :ref:`sif_file_name<doxid-structlsqp__control__type_sif_file_name>`[31];
		char :ref:`prefix<doxid-structlsqp__control__type_prefix>`[31];
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>` :ref:`fdc_control<doxid-structlsqp__control__type_fdc_control>`;
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structlsqp__control__type_sbls_control>`;
	};
.. _details-structlsqp__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlsqp__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlsqp__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structlsqp__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structlsqp__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structlsqp__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structlsqp__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structlsqp__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; factor
.. _doxid-structlsqp__control__type_factor:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factor

the factorization to be used. Possible values are

* 0 automatic

* 1 Schur-complement factorization

* 2 augmented-system factorization

.. index:: pair: variable; max_col
.. _doxid-structlsqp__control__type_max_col:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col

the maximum number of nonzeros in a column of A which is permitted with the Schur-complement factorization

.. index:: pair: variable; indmin
.. _doxid-structlsqp__control__type_indmin:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` indmin

an initial guess as to the integer workspace required by SBLS

.. index:: pair: variable; valmin
.. _doxid-structlsqp__control__type_valmin:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` valmin

an initial guess as to the real workspace required by SBLS

.. index:: pair: variable; itref_max
.. _doxid-structlsqp__control__type_itref_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itref_max

the maximum number of iterative refinements allowed

.. index:: pair: variable; infeas_max
.. _doxid-structlsqp__control__type_infeas_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` infeas_max

the number of iterations for which the overall infeasibility of the problem is not reduced by at least a factor .reduce_infeas before the problem is flagged as infeasible (see reduce_infeas)

.. index:: pair: variable; muzero_fixed
.. _doxid-structlsqp__control__type_muzero_fixed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` muzero_fixed

the initial value of the barrier parameter will not be changed for the first muzero_fixed iterations

.. index:: pair: variable; restore_problem
.. _doxid-structlsqp__control__type_restore_problem:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are

* 0 nothing restored

* 1 scalar and vector parameters

* 2 all parameters

.. index:: pair: variable; indicator_type
.. _doxid-structlsqp__control__type_indicator_type:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` indicator_type

specifies the type of indicator function used. Possible values are

* 1 primal indicator: constraint active if and only if the distance to nearest bound $\leq$.indicator_p_tol

* 2 primal-dual indicator: constraint active if and only if the distance to nearest bound $\leq$.indicator_tol_pd \* size of corresponding multiplier

* 3 primal-dual indicator: constraint active if and only if the distance to the nearest bound $\leq$.indicator_tol_tapia \* distance to same bound at previous iteration

.. index:: pair: variable; extrapolate
.. _doxid-structlsqp__control__type_extrapolate:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` extrapolate

should extrapolation be used to track the central path? Possible values

* 0 never

* 1 after the final major iteration

* 2 at each major iteration (unused at present)

.. index:: pair: variable; path_history
.. _doxid-structlsqp__control__type_path_history:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` path_history

the maximum number of previous path points to use when fitting the data (unused at present)

.. index:: pair: variable; path_derivatives
.. _doxid-structlsqp__control__type_path_derivatives:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` path_derivatives

the maximum order of path derivative to use (unused at present)

.. index:: pair: variable; fit_order
.. _doxid-structlsqp__control__type_fit_order:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` fit_order

the order of (Puiseux) series to fit to the path data: $

.. math::



to fit all data (unused at present)

.. index:: pair: variable; sif_file_device
.. _doxid-structlsqp__control__type_sif_file_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structlsqp__control__type_infinity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_p
.. _doxid-structlsqp__control__type_stop_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_p

the required accuracy for the primal infeasibility

.. index:: pair: variable; stop_d
.. _doxid-structlsqp__control__type_stop_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; stop_c
.. _doxid-structlsqp__control__type_stop_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_c

the required accuracy for the complementarity

.. index:: pair: variable; prfeas
.. _doxid-structlsqp__control__type_prfeas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` prfeas

initial primal variables will not be closer than prfeas from their bounds

.. index:: pair: variable; dufeas
.. _doxid-structlsqp__control__type_dufeas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` dufeas

initial dual variables will not be closer than dufeas from their bounds

.. index:: pair: variable; muzero
.. _doxid-structlsqp__control__type_muzero:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` muzero

the initial value of the barrier parameter. If muzero is not positive, it will be reset to an appropriate value

.. index:: pair: variable; reduce_infeas
.. _doxid-structlsqp__control__type_reduce_infeas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce_infeas

if the overall infeasibility of the problem is not reduced by at least a factor reduce_infeas over .infeas_max iterations, the problem is flagged as infeasible (see infeas_max)

.. index:: pair: variable; potential_unbounded
.. _doxid-structlsqp__control__type_potential_unbounded:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` potential_unbounded

if W=0 and the potential function value is smaller than potential_unbounded \* number of one-sided bounds, the analytic center will be flagged as unbounded

.. index:: pair: variable; pivot_tol
.. _doxid-structlsqp__control__type_pivot_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol

the threshold pivot used by the matrix factorization. See the documentation for SBLS for details

.. index:: pair: variable; pivot_tol_for_dependencies
.. _doxid-structlsqp__control__type_pivot_tol_for_dependencies:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol_for_dependencies

the threshold pivot used by the matrix factorization when attempting to detect linearly dependent constraints. See the documentation for SBLS for details

.. index:: pair: variable; zero_pivot
.. _doxid-structlsqp__control__type_zero_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_pivot

any pivots smaller than zero_pivot in absolute value will be regarded to zero when attempting to detect linearly dependent constraints

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structlsqp__control__type_identical_bounds_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer tha identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; mu_min
.. _doxid-structlsqp__control__type_mu_min:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mu_min

start terminal extrapolation when mu reaches mu_min

.. index:: pair: variable; indicator_tol_p
.. _doxid-structlsqp__control__type_indicator_tol_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` indicator_tol_p

if .indicator_type = 1, a constraint/bound will be deemed to be active if and only if the distance to nearest bound $ $\leq$.indicator_p_tol

.. index:: pair: variable; indicator_tol_pd
.. _doxid-structlsqp__control__type_indicator_tol_pd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` indicator_tol_pd

if .indicator_type = 2, a constraint/bound will be deemed to be active if and only if the distance to nearest bound $ $\leq$.indicator_tol_pd \* size of corresponding multiplier

.. index:: pair: variable; indicator_tol_tapia
.. _doxid-structlsqp__control__type_indicator_tol_tapia:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` indicator_tol_tapia

if .indicator_type = 3, a constraint/bound will be deemed to be active if and only if the distance to nearest bound $ $\leq$.indicator_tol_tapia \* distance to same bound at previous iteration

.. index:: pair: variable; cpu_time_limit
.. _doxid-structlsqp__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structlsqp__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; remove_dependencies
.. _doxid-structlsqp__control__type_remove_dependencies:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool remove_dependencies

the equality constraints will be preprocessed to remove any linear dependencies if true

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structlsqp__control__type_treat_zero_bounds_as_general:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; just_feasible
.. _doxid-structlsqp__control__type_just_feasible:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool just_feasible

if .just_feasible is true, the algorithm will stop as soon as a feasible point is found. Otherwise, the optimal solution to the problem will be found

.. index:: pair: variable; getdua
.. _doxid-structlsqp__control__type_getdua:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool getdua

if .getdua, is true, advanced initial values are obtained for the dual variables

.. index:: pair: variable; puiseux
.. _doxid-structlsqp__control__type_puiseux:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool puiseux

If extrapolation is to be used, decide between Puiseux and Taylor series.

.. index:: pair: variable; feasol
.. _doxid-structlsqp__control__type_feasol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasol

if .feasol is true, the final solution obtained will be perturbed so tha variables close to their bounds are moved onto these bounds

.. index:: pair: variable; balance_initial_complentarity
.. _doxid-structlsqp__control__type_balance_initial_complentarity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool balance_initial_complentarity

if .balance_initial_complentarity is true, the initial complemetarity is required to be balanced

.. index:: pair: variable; use_corrector
.. _doxid-structlsqp__control__type_use_corrector:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool use_corrector

if .use_corrector, a corrector step will be used

.. index:: pair: variable; array_syntax_worse_than_do_loop
.. _doxid-structlsqp__control__type_array_syntax_worse_than_do_loop:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool array_syntax_worse_than_do_loop

if .array_syntax_worse_than_do_loop is true, f77-style do loops will be used rather than f90-style array syntax for vector operations

.. index:: pair: variable; space_critical
.. _doxid-structlsqp__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlsqp__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structlsqp__control__type_generate_sif_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structlsqp__control__type_sif_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structlsqp__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structlsqp__control__type_fdc_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sbls_control
.. _doxid-structlsqp__control__type_sbls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

