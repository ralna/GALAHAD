.. index:: pair: struct; snls_control_type
.. _doxid-structsnls__control__type:

snls_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct snls_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          print_gap::INT
          maxit::INT
          alive_unit::INT
          alive_file::NTuple{31,Cchar}
          jacobian_available::INT
          subproblem_solver::INT
          non_monotone::INT
          weight_update_strategy::INT
          stop_r_absolute::T
          stop_r_relative::T
          stop_pg_absolute::T
          stop_pg_relative::T
          stop_s::T
          stop_pg_switch::T
          initial_weight::T
          minimum_weight::T
          eta_successful::T
          eta_very_successful::T
          eta_too_successful::T
          weight_decrease_min::T
          weight_decrease::T
          weight_increase::T
          weight_increase_max::T
          switch_to_newton::T
          cpu_time_limit::T
          clock_time_limit::T
          newton_acceleration::Bool
          magic_step::Bool
          print_obj::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}
          slls_control::slls_control_type{T,INT}
          sllsb_control::sllsb_control_type{T,INT}

.. _details-structsnls__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsnls__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsnls__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structsnls__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structsnls__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structsnls__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structsnls__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structsnls__control__type_print_gap:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structsnls__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

the maximum number of iterations performed

.. index:: pair: variable; alive_unit
.. _doxid-structsnls__control__type_alive_unit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structsnls__control__type_alive_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; jacobian_available
.. _doxid-structsnls__control__type_jacobian_available:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT jacobian_available

is the Jacobian matrix of first derivatives available ($\geq$ 2), is access only via matrix-vector products (=1) or is it not available ($\leq$ 0) ?

.. index:: pair: variable; subproblem_solver
.. _doxid-structsnls__control__type_subproblem_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT subproblem_solver

the method used to solve the crucial step-determination subproblem.
Possible values are

* 1 a projected-gradient method using GALAHAD's ``slls`` will be used

* 2 an interior-point method using GALAHAD's ``sllsb`` will be used

* 3 an interior-point method will initially be used, but a switch to a projected-gradient method will occur when sufficient progress has occurred (see .stop_pg_switch).

.. index:: pair: variable; non_monotone
.. _doxid-structsnls__control__type_non_monotone:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT non_monotone

non-monotone $\leq$ 0 monotone strategy used, anything else non-monotone strategy with this history length used

.. index:: pair: variable; weight_update_strategy
.. _doxid-structsnls__control__type_weight_update_strategy:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT weight_update_strategy

define the weight-update strategy: 1 (basic), 2 (reset to zero when very
successful), 3 (imitate TR), 4 (increase lower bound), 5 (GPT)

.. index:: pair: variable; stop_r_absolute
.. _doxid-structsnls__control__type_stop_r_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_r_absolute

overall convergence tolerances. The iteration will terminate when
$||r(x)||_2 \leq$ MAX( .stop_r_absolute, .stop_r_relative $*
\|r(x_0)\|_2$ or when the norm of the gradient, $g(x) = J^T(x) W r(x)$
satisfies $\|P[x-g(x)]-x\|_2 \leq$
MAX( .stop_pg_absolute, .stop_pg_relative $* \|P[x_0-g(x_0)]-x_0\|_2$
or if the norm of step is less than .stop_s, where $x_0$ is the initial point.

.. index:: pair: variable; stop_r_relative
.. _doxid-structsnls__control__type_stop_r_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_r_relative

see stop_r_absolute

.. index:: pair: variable; stop_pg_absolute
.. _doxid-structsnls__control__type_stop_pg_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_pg_absolute

see stop_r_absolute

.. index:: pair: variable; stop_pg_relative
.. _doxid-structsnls__control__type_stop_pg_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_pg_relative

see stop_r_absolute

.. index:: pair: variable; stop_s
.. _doxid-structsnls__control__type_stop_s:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_s

see stop_r_absolute


.. index:: pair: variable; stop_pg_switch
.. _doxid-structsnls__control__type_stop_pg_switch:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_pg_switch

the step-computation solver will switch from an interior-point
method to a projected-gradient one if .subproblem_solver = 3 (see above) and
$\|P[x-g(x)]-x\|_2 \leq$
MAX( .stop_pg_absolute, .stop_pg_switch $* \|P[x_0-g(x_0)]-x_0\|_2.

.. index:: pair: variable; initial_weight
.. _doxid-structsnls__control__type_initial_weight:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T initial_weight

initial value for the regularization weight (-ve => $1/\|g_0\|)$)

.. index:: pair: variable; minimum_weight
.. _doxid-structsnls__control__type_minimum_weight:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T minimum_weight

minimum permitted regularization weight

.. index:: pair: variable; eta_successful
.. _doxid-structsnls__control__type_eta_successful:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_successful

a potential iterate will only be accepted if the actual decrease f - f(x_new) is larger than .eta_successful times that predicted by a quadratic model of the decrease. The regularization weight will be decreaed if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful

.. index:: pair: variable; eta_very_successful
.. _doxid-structsnls__control__type_eta_very_successful:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structsnls__control__type_eta_too_successful:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_too_successful

see eta_successful

.. index:: pair: variable; weight_decrease_min
.. _doxid-structsnls__control__type_weight_decrease_min:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight_decrease_min

on very successful iterations, the regularization weight will be reduced by the factor .weight_decrease but no more than .weight_decrease_min while if the iteration is unsucceful, the weight will be increased by a factor .weight_increase but no more than .weight_increase_max (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli and Toint, 2011)

.. index:: pair: variable; weight_decrease
.. _doxid-structsnls__control__type_weight_decrease:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight_decrease

see weight_decrease_min

.. index:: pair: variable; weight_increase
.. _doxid-structsnls__control__type_weight_increase:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight_increase

see weight_decrease_min

.. index:: pair: variable; weight_increase_max
.. _doxid-structsnls__control__type_weight_increase_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight_increase_max

see weight_decrease_min

.. index:: pair: variable; switch_to_newton
.. _doxid-structsnls__control__type_switch_to_newton:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T switch_to_newton

if the value of the two-norm of the projected gradient is less than
.switch_to_newton, a switch is made from the Gauss-Newton model to the Newton 
one when .newton_acceleration is true

.. index:: pair: variable; cpu_time_limit
.. _doxid-structsnls__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structsnls__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; newton_acceleration
.. _doxid-structsnls__control__type_newton_acceleration:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool newton_acceleration

if they are available, second derivatives should be used to accelerate the convergence of the algorithm

.. index:: pair: variable; magic_step
.. _doxid-structsnls__control__type_magic_step:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool magic_step

allow the user to perform a "magic" step to improve the objective

.. index:: pair: variable; print_obj
.. _doxid-structsnls__control__type_print_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool print_obj

print values of the objective/gradient rather than $\|r\|$ and its gradient

.. index:: pair: variable; space_critical
.. _doxid-structsnls__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structsnls__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structsnls__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; slls_control
.. _doxid-structsnls__control__type_slls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`slls_control_type<doxid-structslls__control__type>` slls_control

control parameters for SLLS

.. index:: pair: variable; sllsb_control
.. _doxid-structsnls__control__type_sllsb_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sllsb_control_type<doxid-structsllsb__control__type>` sllsb_control

control parameters for SLLSB

