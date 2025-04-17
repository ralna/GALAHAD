.. index:: pair: struct; wcp_control_type
.. _doxid-structwcp__control__type:

wcp_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct wcp_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          maxit::INT
          initial_point::INT
          factor::INT
          max_col::INT
          indmin::INT
          valmin::INT
          itref_max::INT
          infeas_max::INT
          perturbation_strategy::INT
          restore_problem::INT
          infinity::T
          stop_p::T
          stop_d::T
          stop_c::T
          prfeas::T
          dufeas::T
          mu_target::T
          mu_accept_fraction::T
          mu_increase_factor::T
          required_infeas_reduction::T
          implicit_tol::T
          pivot_tol::T
          pivot_tol_for_dependencies::T
          zero_pivot::T
          perturb_start::T
          alpha_scale::T
          identical_bounds_tol::T
          reduce_perturb_factor::T
          reduce_perturb_multiplier::T
          insufficiently_feasible::T
          perturbation_small::T
          cpu_time_limit::T
          clock_time_limit::T
          remove_dependencies::Bool
          treat_zero_bounds_as_general::Bool
          just_feasible::Bool
          balance_initial_complementarity::Bool
          use_corrector::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          record_x_status::Bool
          record_c_status::Bool
          prefix::NTuple{31,Cchar}
          fdc_control::fdc_control_type{T,INT}
          sbls_control::sbls_control_type{T,INT}


.. _details-structwcp__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structwcp__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structwcp__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structwcp__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structwcp__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structwcp__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structwcp__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structwcp__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; initial_point
.. _doxid-structwcp__control__type_initial_point:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT initial_point

how to choose the initial point. Possible values are

* 0 the values input in X, shifted to be at least prfeas from their nearest bound, will be used

* 1 the nearest point to the "bound average" 0.5(X_l+X_u) that satisfies the linear constraints will be used

.. index:: pair: variable; factor
.. _doxid-structwcp__control__type_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factor

the factorization to be used. Possible values are

* 0 automatic

* 1 Schur-complement factorization

* 2 augmented-system factorization

.. index:: pair: variable; max_col
.. _doxid-structwcp__control__type_max_col:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_col

the maximum number of nonzeros in a column of A which is permitted with the Schur-complement factorization

.. index:: pair: variable; indmin
.. _doxid-structwcp__control__type_indmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT indmin

an initial guess as to the integer workspace required by SBLS

.. index:: pair: variable; valmin
.. _doxid-structwcp__control__type_valmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT valmin

an initial guess as to the real workspace required by SBLS

.. index:: pair: variable; itref_max
.. _doxid-structwcp__control__type_itref_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT itref_max

the maximum number of iterative refinements allowed

.. index:: pair: variable; infeas_max
.. _doxid-structwcp__control__type_infeas_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT infeas_max

the number of iterations for which the overall infeasibility of the problem is not reduced by at least a factor .required_infeas_reduction before the problem is flagged as infeasible (see required_infeas_reducti

.. index:: pair: variable; perturbation_strategy
.. _doxid-structwcp__control__type_perturbation_strategy:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT perturbation_strategy

the strategy used to reduce relaxed constraint bounds. Possible values are

* 0 do not perturb the constraints

* 1 reduce all perturbations by the same amount with linear reduction

* 2 reduce each perturbation as much as possible with linear reduction

* 3 reduce all perturbations by the same amount with superlinear reduction

* 4 reduce each perturbation as much as possible with superlinear reduction

.. index:: pair: variable; restore_problem
.. _doxid-structwcp__control__type_restore_problem:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are

* 0 nothing restored

* 1 scalar and vector parameters

* 2 all parameters

.. index:: pair: variable; infinity
.. _doxid-structwcp__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_p
.. _doxid-structwcp__control__type_stop_p:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_p

the required accuracy for the primal infeasibility

.. index:: pair: variable; stop_d
.. _doxid-structwcp__control__type_stop_d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; stop_c
.. _doxid-structwcp__control__type_stop_c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_c

the required accuracy for the complementarity

.. index:: pair: variable; prfeas
.. _doxid-structwcp__control__type_prfeas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T prfeas

initial primal variables will not be closer than prfeas from their bound

.. index:: pair: variable; dufeas
.. _doxid-structwcp__control__type_dufeas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T dufeas

initial dual variables will not be closer than dufeas from their bounds

.. index:: pair: variable; mu_target
.. _doxid-structwcp__control__type_mu_target:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mu_target

the target value of the barrier parameter. If mu_target is not positive, it will be reset to an appropriate value

.. index:: pair: variable; mu_accept_fraction
.. _doxid-structwcp__control__type_mu_accept_fraction:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mu_accept_fraction

the complemtary slackness x_i.z_i will be judged to lie within an acceptable margin around its target value mu as soon as mu_accept_fraction \* mu <= x_i.z_i <= ( 1 / mu_accept_fraction ) \* mu; the perturbations will be reduced as soon as all of the complemtary slacknesses x_i.z_i lie within acceptable bounds. mu_accept_fraction will be reset to ensure that it lies in the interval (0,1]

.. index:: pair: variable; mu_increase_factor
.. _doxid-structwcp__control__type_mu_increase_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mu_increase_factor

the target value of the barrier parameter will be increased by mu_increase_factor for infeasible constraints every time the perturbations are adjusted

.. index:: pair: variable; required_infeas_reduction
.. _doxid-structwcp__control__type_required_infeas_reduction:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T required_infeas_reduction

if the overall infeasibility of the problem is not reduced by at least a factor required_infeas_reduction over .infeas_max iterations, the problem is flagged as infeasible (see infeas_max)

.. index:: pair: variable; implicit_tol
.. _doxid-structwcp__control__type_implicit_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T implicit_tol

any primal or dual variable that is less feasible than implicit_tol will be regarded as defining an implicit constraint

.. index:: pair: variable; pivot_tol
.. _doxid-structwcp__control__type_pivot_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol

the threshold pivot used by the matrix factorization. See the documentation for SBLS for details (obsolete)

.. index:: pair: variable; pivot_tol_for_dependencies
.. _doxid-structwcp__control__type_pivot_tol_for_dependencies:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol_for_dependencies

the threshold pivot used by the matrix factorization when attempting to detect linearly dependent constraints. See the documentation for SBLS for details (obsolete)

.. index:: pair: variable; zero_pivot
.. _doxid-structwcp__control__type_zero_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_pivot

any pivots smaller than zero_pivot in absolute value will be regarded to zero when attempting to detect linearly dependent constraints (obsolete)

.. index:: pair: variable; perturb_start
.. _doxid-structwcp__control__type_perturb_start:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T perturb_start

the constraint bounds will initially be relaxed by .perturb_start; this perturbation will subsequently be reduced to zero. If perturb_start < 0, the amount by which the bounds are relaxed will be computed automatically

.. index:: pair: variable; alpha_scale
.. _doxid-structwcp__control__type_alpha_scale:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T alpha_scale

the test for rank defficiency will be to factorize ( alpha_scale I A^T ) ( A 0 )

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structwcp__control__type_identical_bounds_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T identical_bounds_tol

any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer tha identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; reduce_perturb_factor
.. _doxid-structwcp__control__type_reduce_perturb_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T reduce_perturb_factor

the constraint perturbation will be reduced as follows:

* - if the variable lies outside a bound, the corresponding perturbation will be reduced to reduce_perturb_factor \* current pertubation

  * ( 1 - reduce_perturb_factor ) \* violation

* - otherwise, if the variable lies within insufficiently_feasible of its bound the pertubation will be reduced to reduce_perturb_multiplier \* current pertubation

* - otherwise if will be set to zero

.. index:: pair: variable; reduce_perturb_multiplier
.. _doxid-structwcp__control__type_reduce_perturb_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T reduce_perturb_multiplier

see reduce_perturb_factor

.. index:: pair: variable; insufficiently_feasible
.. _doxid-structwcp__control__type_insufficiently_feasible:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T insufficiently_feasible

see reduce_perturb_factor

.. index:: pair: variable; perturbation_small
.. _doxid-structwcp__control__type_perturbation_small:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T perturbation_small

if the maximum constraint pertubation is smaller than perturbation_small and the violation is smaller than implicit_tol, the method will deduce that there is a feasible point but no interior

.. index:: pair: variable; cpu_time_limit
.. _doxid-structwcp__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structwcp__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; remove_dependencies
.. _doxid-structwcp__control__type_remove_dependencies:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool remove_dependencies

the equality constraints will be preprocessed to remove any linear dependencies if true

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structwcp__control__type_treat_zero_bounds_as_general:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; just_feasible
.. _doxid-structwcp__control__type_just_feasible:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool just_feasible

if .just_feasible is true, the algorithm will stop as soon as a feasible point is found. Otherwise, the optimal solution to the problem will be found

.. index:: pair: variable; balance_initial_complementarity
.. _doxid-structwcp__control__type_balance_initial_complementarity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool balance_initial_complementarity

if .balance_initial_complementarity is .true. the initial complemetarity will be balanced

.. index:: pair: variable; use_corrector
.. _doxid-structwcp__control__type_use_corrector:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool use_corrector

if .use_corrector, a corrector step will be used

.. index:: pair: variable; space_critical
.. _doxid-structwcp__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structwcp__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; record_x_status
.. _doxid-structwcp__control__type_record_x_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool record_x_status

if .record_x_status is true, the array inform.X_status will be allocated and the status of the bound constraints will be reported on exit.

.. index:: pair: variable; record_c_status
.. _doxid-structwcp__control__type_record_c_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool record_c_status

if .record_c_status is true, the array inform.C_status will be allocated and the status of the general constraints will be reported on exit.

.. index:: pair: variable; prefix
.. _doxid-structwcp__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structwcp__control__type_fdc_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	fdc_control_type{T,INT} :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sbls_control
.. _doxid-structwcp__control__type_sbls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	sbls_control_type{T,INT} :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS
