.. index:: pair: struct; qpb_control_type
.. _doxid-structqpb__control__type:

qpb_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct qpb_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          maxit::INT
          itref_max::INT
          cg_maxit::INT
          indicator_type::INT
          restore_problem::INT
          extrapolate::INT
          path_history::INT
          factor::INT
          max_col::INT
          indmin::INT
          valmin::INT
          infeas_max::INT
          precon::INT
          nsemib::INT
          path_derivatives::INT
          fit_order::INT
          sif_file_device::INT
          infinity::T
          stop_p::T
          stop_d::T
          stop_c::T
          theta_d::T
          theta_c::T
          beta::T
          prfeas::T
          dufeas::T
          muzero::T
          reduce_infeas::T
          obj_unbounded::T
          pivot_tol::T
          pivot_tol_for_dependencies::T
          zero_pivot::T
          identical_bounds_tol::T
          inner_stop_relative::T
          inner_stop_absolute::T
          initial_radius::T
          mu_min::T
          inner_fraction_opt::T
          indicator_tol_p::T
          indicator_tol_pd::T
          indicator_tol_tapia::T
          cpu_time_limit::T
          clock_time_limit::T
          remove_dependencies::Bool
          treat_zero_bounds_as_general::Bool
          center::Bool
          primal::Bool
          puiseux::Bool
          feasol::Bool
          array_syntax_worse_than_do_loop::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          generate_sif_file::Bool
          sif_file_name::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          lsqp_control::lsqp_control_type{T,INT}
          fdc_control::fdc_control_type{T,INT}
          sbls_control::sbls_control_type{T,INT}
          gltr_control::gltr_control_type{T,INT}
          fit_control::fit_control_type{INT}

.. _details-structqpb__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structqpb__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structqpb__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structqpb__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structqpb__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structqpb__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structqpb__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structqpb__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; itref_max
.. _doxid-structqpb__control__type_itref_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT itref_max

the maximum number of iterative refinements allowed

.. index:: pair: variable; cg_maxit
.. _doxid-structqpb__control__type_cg_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_maxit

the maximum number of CG iterations allowed. If cg_maxit < 0, this number will be reset to the dimension of the system + 1

.. index:: pair: variable; indicator_type
.. _doxid-structqpb__control__type_indicator_type:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT indicator_type

specifies the type of indicator function used. Pssible values are

* 1 primal indicator: constraint active <=> distance to nearest bound <= .indicator_p_tol

* 2 primal-dual indicator: constraint active <=> distance to nearest bound <= .indicator_tol_pd \* size of corresponding multiplier

* 3 primal-dual indicator: constraint active <=> distance to nearest bound <= .indicator_tol_tapia \* distance to same bound at previous iteration

.. index:: pair: variable; restore_problem
.. _doxid-structqpb__control__type_restore_problem:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are

* 0 nothing restored

* 1 scalar and vector parameters

* 2 all parameters

.. index:: pair: variable; extrapolate
.. _doxid-structqpb__control__type_extrapolate:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT extrapolate

should extrapolation be used to track the central path? Possible values

* 0 never

* 1 after the final major iteration

* 2 at each major iteration

.. index:: pair: variable; path_history
.. _doxid-structqpb__control__type_path_history:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT path_history

the maximum number of previous path points to use when fitting the data

.. index:: pair: variable; factor
.. _doxid-structqpb__control__type_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factor

the factorization to be used. Possible values are

* 0 automatic

* 1 Schur-complement factorization

* 2 augmented-system factorization

.. index:: pair: variable; max_col
.. _doxid-structqpb__control__type_max_col:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_col

the maximum number of nonzeros in a column of A which is permitted with the Schur-complement factorization

.. index:: pair: variable; indmin
.. _doxid-structqpb__control__type_indmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT indmin

an initial guess as to the integer workspace required by SBLS

.. index:: pair: variable; valmin
.. _doxid-structqpb__control__type_valmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT valmin

an initial guess as to the real workspace required by SBLS

.. index:: pair: variable; infeas_max
.. _doxid-structqpb__control__type_infeas_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT infeas_max

the number of iterations for which the overall infeasibility of the problem is not reduced by at least a factor .reduce_infeas before the problem is flagged as infeasible (see reduce_infeas)

.. index:: pair: variable; precon
.. _doxid-structqpb__control__type_precon:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT precon

the preconditioner to be used for the CG is defined by precon. Possible values are

* 0 automatic

* 1 no preconditioner, i.e, the identity within full factorization

* 2 full factorization

* 3 band within full factorization

* 4 diagonal using the barrier terms within full factorization

.. index:: pair: variable; nsemib
.. _doxid-structqpb__control__type_nsemib:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nsemib

the semi-bandwidth of a band preconditioner, if appropriate

.. index:: pair: variable; path_derivatives
.. _doxid-structqpb__control__type_path_derivatives:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT path_derivatives

the maximum order of path derivative to use

.. index:: pair: variable; fit_order
.. _doxid-structqpb__control__type_fit_order:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT fit_order

the order of (Puiseux) series to fit to the path data: <=0 to fit all data

.. index:: pair: variable; sif_file_device
.. _doxid-structqpb__control__type_sif_file_device:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structqpb__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_p
.. _doxid-structqpb__control__type_stop_p:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_p

the required accuracy for the primal infeasibility

.. index:: pair: variable; stop_d
.. _doxid-structqpb__control__type_stop_d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; stop_c
.. _doxid-structqpb__control__type_stop_c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_c

the required accuracy for the complementarity

.. index:: pair: variable; theta_d
.. _doxid-structqpb__control__type_theta_d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T theta_d

tolerances used to terminate the inner iteration (for given mu): dual feasibility <= MAX( theta_d \* mu \*\* beta, 0.99 \* stop_d ) complementarity <= MAX( theta_c \* mu \*\* beta, 0.99 \* stop_d )

.. index:: pair: variable; theta_c
.. _doxid-structqpb__control__type_theta_c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T theta_c

see theta_d

.. index:: pair: variable; beta
.. _doxid-structqpb__control__type_beta:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T beta

see theta_d

.. index:: pair: variable; prfeas
.. _doxid-structqpb__control__type_prfeas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T prfeas

initial primal variables will not be closer than prfeas from their bound

.. index:: pair: variable; dufeas
.. _doxid-structqpb__control__type_dufeas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T dufeas

initial dual variables will not be closer than dufeas from their bounds

.. index:: pair: variable; muzero
.. _doxid-structqpb__control__type_muzero:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T muzero

the initial value of the barrier parameter. If muzero is not positive, it will be reset to an appropriate value

.. index:: pair: variable; reduce_infeas
.. _doxid-structqpb__control__type_reduce_infeas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T reduce_infeas

if the overall infeasibility of the problem is not reduced by at least a factor reduce_infeas over .infeas_max iterations, the problem is flagged as infeasible (see infeas_max)

.. index:: pair: variable; obj_unbounded
.. _doxid-structqpb__control__type_obj_unbounded:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_unbounded

if the objective function value is smaller than obj_unbounded, it will be flagged as unbounded from below.

.. index:: pair: variable; pivot_tol
.. _doxid-structqpb__control__type_pivot_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol

the threshold pivot used by the matrix factorization. See the documentation for SBLS for details

.. index:: pair: variable; pivot_tol_for_dependencies
.. _doxid-structqpb__control__type_pivot_tol_for_dependencies:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol_for_dependencies

the threshold pivot used by the matrix factorization when attempting to detect linearly dependent constraints. See the documentation for FDC for details

.. index:: pair: variable; zero_pivot
.. _doxid-structqpb__control__type_zero_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_pivot

any pivots smaller than zero_pivot in absolute value will be regarded to zero when attempting to detect linearly dependent constraints

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structqpb__control__type_identical_bounds_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T identical_bounds_tol

any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; inner_stop_relative
.. _doxid-structqpb__control__type_inner_stop_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_stop_relative

the search direction is considered as an acceptable approximation to the minimizer of the model if the gradient of the model in the preconditioning(inverse) norm is less than max( inner_stop_relative \* initial preconditioning(inverse) gradient norm, inner_stop_absolute )

.. index:: pair: variable; inner_stop_absolute
.. _doxid-structqpb__control__type_inner_stop_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_stop_absolute

see inner_stop_relative

.. index:: pair: variable; initial_radius
.. _doxid-structqpb__control__type_initial_radius:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T initial_radius

the initial trust-region radius

.. index:: pair: variable; mu_min
.. _doxid-structqpb__control__type_mu_min:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mu_min

start terminal extrapolation when mu reaches mu_min

.. index:: pair: variable; inner_fraction_opt
.. _doxid-structqpb__control__type_inner_fraction_opt:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_fraction_opt

a search direction which gives at least inner_fraction_opt times the optimal model decrease will be found

.. index:: pair: variable; indicator_tol_p
.. _doxid-structqpb__control__type_indicator_tol_p:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T indicator_tol_p

if .indicator_type = 1, a constraint/bound will be deemed to be active <=> distance to nearest bound <= .indicator_p_tol

.. index:: pair: variable; indicator_tol_pd
.. _doxid-structqpb__control__type_indicator_tol_pd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T indicator_tol_pd

if .indicator_type = 2, a constraint/bound will be deemed to be active <=> distance to nearest bound <= .indicator_tol_pd \* size of corresponding multiplier

.. index:: pair: variable; indicator_tol_tapia
.. _doxid-structqpb__control__type_indicator_tol_tapia:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T indicator_tol_tapia

if .indicator_type = 3, a constraint/bound will be deemed to be active <=> distance to nearest bound <= .indicator_tol_tapia \* distance to same bound at previous iteration

.. index:: pair: variable; cpu_time_limit
.. _doxid-structqpb__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structqpb__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; remove_dependencies
.. _doxid-structqpb__control__type_remove_dependencies:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool remove_dependencies

the equality constraints will be preprocessed to remove any linear dependencies if true

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structqpb__control__type_treat_zero_bounds_as_general:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; center
.. _doxid-structqpb__control__type_center:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool center

if .center is true, the algorithm will use the analytic center of the feasible set as its initial feasible point. Otherwise, a feasible point as close as possible to the initial point will be used. We recommend using the analytic center

.. index:: pair: variable; primal
.. _doxid-structqpb__control__type_primal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool primal

if .primal, is true, a primal barrier method will be used in place of t primal-dual method

.. index:: pair: variable; puiseux
.. _doxid-structqpb__control__type_puiseux:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool puiseux

If extrapolation is to be used, decide between Puiseux and Taylor series.

.. index:: pair: variable; feasol
.. _doxid-structqpb__control__type_feasol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasol

if .feasol is true, the final solution obtained will be perturbed so that variables close to their bounds are moved onto these bounds

.. index:: pair: variable; array_syntax_worse_than_do_loop
.. _doxid-structqpb__control__type_array_syntax_worse_than_do_loop:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool array_syntax_worse_than_do_loop

if .array_syntax_worse_than_do_loop is true, f77-style do loops will be used rather than f90-style array syntax for vector operations

.. index:: pair: variable; space_critical
.. _doxid-structqpb__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structqpb__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structqpb__control__type_generate_sif_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structqpb__control__type_sif_file_name:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} sif_file_name

name of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structqpb__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; lsqp_control
.. _doxid-structqpb__control__type_lsqp_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lsqp_control_type<doxid-structlsqp__control__type>` lsqp_control

control parameters for LSQP

.. index:: pair: variable; fdc_control
.. _doxid-structqpb__control__type_fdc_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sbls_control
.. _doxid-structqpb__control__type_sbls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; gltr_control
.. _doxid-structqpb__control__type_gltr_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

.. index:: pair: variable; fit_control
.. _doxid-structqpb__control__type_fit_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fit_control_type<doxid-structfit__control__type>` fit_control

control parameters for FIT

