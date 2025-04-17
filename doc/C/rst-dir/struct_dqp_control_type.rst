.. index:: pair: struct; dqp_control_type
.. _doxid-structdqp__control__type:

dqp_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_dqp.h>
	
	struct dqp_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structdqp__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structdqp__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structdqp__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structdqp__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structdqp__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structdqp__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structdqp__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dual_starting_point<doxid-structdqp__control__type_dual_starting_point>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structdqp__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_sc<doxid-structdqp__control__type_max_sc>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cauchy_only<doxid-structdqp__control__type_cauchy_only>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`arc_search_maxit<doxid-structdqp__control__type_arc_search_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structdqp__control__type_cg_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`explore_optimal_subspace<doxid-structdqp__control__type_explore_optimal_subspace>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`restore_problem<doxid-structdqp__control__type_restore_problem>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structdqp__control__type_sif_file_device>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`qplib_file_device<doxid-structdqp__control__type_qplib_file_device>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`rho<doxid-structdqp__control__type_rho>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structdqp__control__type_infinity>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_p<doxid-structdqp__control__type_stop_abs_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_p<doxid-structdqp__control__type_stop_rel_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_d<doxid-structdqp__control__type_stop_abs_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_d<doxid-structdqp__control__type_stop_rel_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_c<doxid-structdqp__control__type_stop_abs_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_c<doxid-structdqp__control__type_stop_rel_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_relative<doxid-structdqp__control__type_stop_cg_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_absolute<doxid-structdqp__control__type_stop_cg_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cg_zero_curvature<doxid-structdqp__control__type_cg_zero_curvature>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`max_growth<doxid-structdqp__control__type_max_growth>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structdqp__control__type_identical_bounds_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structdqp__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structdqp__control__type_clock_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_perturbation<doxid-structdqp__control__type_initial_perturbation>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`perturbation_reduction<doxid-structdqp__control__type_perturbation_reduction>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`final_perturbation<doxid-structdqp__control__type_final_perturbation>`;
		bool :ref:`factor_optimal_matrix<doxid-structdqp__control__type_factor_optimal_matrix>`;
		bool :ref:`remove_dependencies<doxid-structdqp__control__type_remove_dependencies>`;
		bool :ref:`treat_zero_bounds_as_general<doxid-structdqp__control__type_treat_zero_bounds_as_general>`;
		bool :ref:`exact_arc_search<doxid-structdqp__control__type_exact_arc_search>`;
		bool :ref:`subspace_direct<doxid-structdqp__control__type_subspace_direct>`;
		bool :ref:`subspace_alternate<doxid-structdqp__control__type_subspace_alternate>`;
		bool :ref:`subspace_arc_search<doxid-structdqp__control__type_subspace_arc_search>`;
		bool :ref:`space_critical<doxid-structdqp__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structdqp__control__type_deallocate_error_fatal>`;
		bool :ref:`generate_sif_file<doxid-structdqp__control__type_generate_sif_file>`;
		bool :ref:`generate_qplib_file<doxid-structdqp__control__type_generate_qplib_file>`;
		char :ref:`symmetric_linear_solver<doxid-structdqp__control__type_symmetric_linear_solver>`[31];
		char :ref:`definite_linear_solver<doxid-structdqp__control__type_definite_linear_solver>`[31];
		char :ref:`unsymmetric_linear_solver<doxid-structdqp__control__type_unsymmetric_linear_solver>`[31];
		char :ref:`sif_file_name<doxid-structdqp__control__type_sif_file_name>`[31];
		char :ref:`qplib_file_name<doxid-structdqp__control__type_qplib_file_name>`[31];
		char :ref:`prefix<doxid-structdqp__control__type_prefix>`[31];
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>` :ref:`fdc_control<doxid-structdqp__control__type_fdc_control>`;
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structdqp__control__type_sls_control>`;
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structdqp__control__type_sbls_control>`;
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>` :ref:`gltr_control<doxid-structdqp__control__type_gltr_control>`;
	};
.. _details-structdqp__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structdqp__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structdqp__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structdqp__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structdqp__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structdqp__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structdqp__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structdqp__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

printing will only occur every print_gap iterations

.. index:: pair: variable; dual_starting_point
.. _doxid-structdqp__control__type_dual_starting_point:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dual_starting_point

which starting point should be used for the dual problem

* -1 user supplied comparing primal vs dual variables

* 0 user supplied

* 1 minimize linearized dual

* 2 minimize simplified quadratic dual

* 3 all free (= all active primal costraints)

* 4 all fixed on bounds (= no active primal costraints)

.. index:: pair: variable; maxit
.. _doxid-structdqp__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; max_sc
.. _doxid-structdqp__control__type_max_sc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_sc

the maximum permitted size of the Schur complement before a refactorization is performed (used in the case where there is no Fredholm Alternative, 0 = refactor every iteration)

.. index:: pair: variable; cauchy_only
.. _doxid-structdqp__control__type_cauchy_only:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cauchy_only

a subspace step will only be taken when the current Cauchy step has changed no more than than cauchy_only active constraints; the subspace step will always be taken if cauchy_only < 0

.. index:: pair: variable; arc_search_maxit
.. _doxid-structdqp__control__type_arc_search_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` arc_search_maxit

how many iterations are allowed per arc search (-ve = as many as require

.. index:: pair: variable; cg_maxit
.. _doxid-structdqp__control__type_cg_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

how many CG iterations to perform per DQP iteration (-ve reverts to n+1)

.. index:: pair: variable; explore_optimal_subspace
.. _doxid-structdqp__control__type_explore_optimal_subspace:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` explore_optimal_subspace

once a potentially optimal subspace has been found, investigate it

* 0 as per an ordinary subspace

* 1 by increasing the maximum number of allowed CG iterations

* 2 by switching to a direct method

.. index:: pair: variable; restore_problem
.. _doxid-structdqp__control__type_restore_problem:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are

* 0 nothing restored

* 1 scalar and vector parameters

* 2 all parameters

.. index:: pair: variable; sif_file_device
.. _doxid-structdqp__control__type_sif_file_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; qplib_file_device
.. _doxid-structdqp__control__type_qplib_file_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` qplib_file_device

specifies the unit number to write generated QPLIB file describing the current problem

.. index:: pair: variable; rho
.. _doxid-structdqp__control__type_rho:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` rho

the penalty weight, rho. The general constraints are not enforced explicitly, but instead included in the objective as a penalty term weighted by rho when rho > 0. If rho <= 0, the general constraints are explicit (that is, there is no penalty term in the objective function)

.. index:: pair: variable; infinity
.. _doxid-structdqp__control__type_infinity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_abs_p
.. _doxid-structdqp__control__type_stop_abs_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_p

the required absolute and relative accuracies for the primal infeasibilies

.. index:: pair: variable; stop_rel_p
.. _doxid-structdqp__control__type_stop_rel_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_p

see stop_abs_p

.. index:: pair: variable; stop_abs_d
.. _doxid-structdqp__control__type_stop_abs_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_d

the required absolute and relative accuracies for the dual infeasibility

.. index:: pair: variable; stop_rel_d
.. _doxid-structdqp__control__type_stop_rel_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_d

see stop_abs_d

.. index:: pair: variable; stop_abs_c
.. _doxid-structdqp__control__type_stop_abs_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_c

the required absolute and relative accuracies for the complementarity

.. index:: pair: variable; stop_rel_c
.. _doxid-structdqp__control__type_stop_rel_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_c

see stop_abs_c

.. index:: pair: variable; stop_cg_relative
.. _doxid-structdqp__control__type_stop_cg_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_relative

the CG iteration will be stopped as soon as the current norm of the preconditioned gradient is smaller than max( stop_cg_relative \* initial preconditioned gradient, stop_cg_absolute )

.. index:: pair: variable; stop_cg_absolute
.. _doxid-structdqp__control__type_stop_cg_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_absolute

see stop_cg_relative

.. index:: pair: variable; cg_zero_curvature
.. _doxid-structdqp__control__type_cg_zero_curvature:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cg_zero_curvature

threshold below which curvature is regarded as zero if CG is used

.. index:: pair: variable; max_growth
.. _doxid-structdqp__control__type_max_growth:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` max_growth

maximum growth factor allowed without a refactorization

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structdqp__control__type_identical_bounds_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; cpu_time_limit
.. _doxid-structdqp__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structdqp__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; initial_perturbation
.. _doxid-structdqp__control__type_initial_perturbation:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_perturbation

the initial penalty weight (for DLP only)

.. index:: pair: variable; perturbation_reduction
.. _doxid-structdqp__control__type_perturbation_reduction:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` perturbation_reduction

the penalty weight reduction factor (for DLP only)

.. index:: pair: variable; final_perturbation
.. _doxid-structdqp__control__type_final_perturbation:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` final_perturbation

the final penalty weight (for DLP only)

.. index:: pair: variable; factor_optimal_matrix
.. _doxid-structdqp__control__type_factor_optimal_matrix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool factor_optimal_matrix

are the factors of the optimal augmented matrix required? (for DLP only)

.. index:: pair: variable; remove_dependencies
.. _doxid-structdqp__control__type_remove_dependencies:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool remove_dependencies

the equality constraints will be preprocessed to remove any linear dependencies if true

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structdqp__control__type_treat_zero_bounds_as_general:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; exact_arc_search
.. _doxid-structdqp__control__type_exact_arc_search:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool exact_arc_search

if .exact_arc_search is true, an exact piecewise arc search will be performed. Otherwise an ineaxt search using a backtracing Armijo strategy will be employed

.. index:: pair: variable; subspace_direct
.. _doxid-structdqp__control__type_subspace_direct:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subspace_direct

if .subspace_direct is true, the subspace step will be calculated using a direct (factorization) method, while if it is false, an iterative (conjugate-gradient) method will be used.

.. index:: pair: variable; subspace_alternate
.. _doxid-structdqp__control__type_subspace_alternate:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subspace_alternate

if .subspace_alternate is true, the subspace step will alternate between a direct (factorization) method and an iterative (GLTR conjugate-gradient) method. This will override .subspace_direct

.. index:: pair: variable; subspace_arc_search
.. _doxid-structdqp__control__type_subspace_arc_search:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subspace_arc_search

if .subspace_arc_search is true, a piecewise arc search will be performed along the subspace step. Otherwise the search will stop at the firstconstraint encountered

.. index:: pair: variable; space_critical
.. _doxid-structdqp__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structdqp__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structdqp__control__type_generate_sif_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; generate_qplib_file
.. _doxid-structdqp__control__type_generate_qplib_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_qplib_file

if .generate_qplib_file is .true. if a QPLIB file describing the current problem is to be generated

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structdqp__control__type_symmetric_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric-indefinite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', and 'sytr', although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; definite_linear_solver
.. _doxid-structdqp__control__type_definite_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

the name of the definite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', 'potr',  'sytr' and 'pbtr', although only 'potr',  'sytr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structdqp__control__type_unsymmetric_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char unsymmetric_linear_solver[31]

the name of the unsymmetric linear equation solver used. Possible choices are currently: 'gls', 'ma48' and 'getr', although only 'getr' is installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_uls<details-uls__solvers>`.

.. index:: pair: variable; sif_file_name
.. _doxid-structdqp__control__type_sif_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name of generated SIF file containing input problem

.. index:: pair: variable; qplib_file_name
.. _doxid-structdqp__control__type_qplib_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char qplib_file_name[31]

name of generated QPLIB file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structdqp__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structdqp__control__type_fdc_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sls_control
.. _doxid-structdqp__control__type_sls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; sbls_control
.. _doxid-structdqp__control__type_sbls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; gltr_control
.. _doxid-structdqp__control__type_gltr_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

