.. index:: pair: struct; snls_control_type
.. _doxid-structsnls__control__type:

snls_control_type structure
---------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_snls.h>
	
	struct snls_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structsnls__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structsnls__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structsnls__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structsnls__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structsnls__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structsnls__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structsnls__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structsnls__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structsnls__control__type_alive_unit>`;
		char :ref:`alive_file<doxid-structsnls__control__type_alive_file>`[31];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`jacobian_available<doxid-structsnls__control__type_jacobian_available>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`subproblem_solver<doxid-structsnls__control__type_subproblem_solver>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`non_monotone<doxid-structsnls__control__type_non_monotone>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`weight_update_strategy<doxid-structsnls__control__type_weight_update_strategy>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_r_absolute<doxid-structsnls__control__type_stop_r_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_r_relative<doxid-structsnls__control__type_stop_r_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_pg_absolute<doxid-structsnls__control__type_stop_pg_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_pg_relative<doxid-structsnls__control__type_stop_pg_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_s<doxid-structsnls__control__type_stop_s>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_pg_switch<doxid-structsnls__control__type_stop_pg_switch>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_weight<doxid-structsnls__control__type_initial_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`minimum_weight<doxid-structsnls__control__type_minimum_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_successful<doxid-structsnls__control__type_eta_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_very_successful<doxid-structsnls__control__type_eta_very_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_too_successful<doxid-structsnls__control__type_eta_too_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_decrease_min<doxid-structsnls__control__type_weight_decrease_min>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_decrease<doxid-structsnls__control__type_weight_decrease>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_increase<doxid-structsnls__control__type_weight_increase>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_increase_max<doxid-structsnls__control__type_weight_increase_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`switch_to_newton<doxid-structsnls__control__type_switch_to_newton>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structsnls__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structsnls__control__type_clock_time_limit>`;
		bool :ref:`newton_acceleration<doxid-structsnls__control__type_newton_acceleration>`;
		bool :ref:`magic_step<doxid-structsnls__control__type_magic_step>`;
		bool :ref:`print_obj<doxid-structsnls__control__type_print_obj>`;
		bool :ref:`space_critical<doxid-structsnls__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structsnls__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structsnls__control__type_prefix>`[31];
		struct :ref:`slls_control_type<doxid-structslls__control__type>` :ref:`slls_control<doxid-structsnls__control__type_slls_control>`;
		struct :ref:`sllsb_control_type<doxid-structsllsb__control__type>` :ref:`sllsb_control<doxid-structsnls__control__type_sllsb_control>`;
	};
.. _details-structsnls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsnls__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsnls__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structsnls__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structsnls__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structsnls__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structsnls__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structsnls__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structsnls__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

the maximum number of iterations performed

.. index:: pair: variable; alive_unit
.. _doxid-structsnls__control__type_alive_unit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structsnls__control__type_alive_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; jacobian_available
.. _doxid-structsnls__control__type_jacobian_available:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` jacobian_available

is the Jacobian matrix of first derivatives available ($\geq$ 2), is access only via matrix-vector products (=1) or is it not available ($\leq$ 0) ?

.. index:: pair: variable; subproblem_solver
.. _doxid-structsnls__control__type_subproblem_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` subproblem_solver

the method used to solve the crucial step-determination subproblem.
Possible values are

* 1 a projected-gradient method using GALAHAD's ``slls`` will be used

* 2 an interior-point method using GALAHAD's ``sllsb`` will be used

* 3 an interior-point method will initially be used, but a switch to a projected-gradient method will occur when sufficient progress has occurred (see .stop_pg_switch).

.. index:: pair: variable; non_monotone
.. _doxid-structsnls__control__type_non_monotone:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` non_monotone

non-monotone $\leq$ 0 monotone strategy used, anything else non-monotone strategy with this history length used

.. index:: pair: variable; weight_update_strategy
.. _doxid-structsnls__control__type_weight_update_strategy:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` weight_update_strategy

define the weight-update strategy: 1 (basic), 2 (reset to zero when very successful), 3 (imitate TR), 4 (increase lower bound), 5 (GPT)

.. index:: pair: variable; stop_r_absolute
.. _doxid-structsnls__control__type_stop_r_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_r_absolute

overall convergence tolerances. The iteration will terminate when
$||r(x)||_2 \leq$ MAX( .stop_r_absolute, .stop_r_relative $*
\|r(x_0)\|_2$ or when the norm of the gradient, $g(x) = J^T(x) W r(x)$
satisfies $\|P[x-g(x)]-x\|_2 \leq$
MAX( .stop_pg_absolute, .stop_pg_relative $* \|P[x_0-g(x_0)]-x_0\|_2$
or if the norm of step is less than .stop_s, where $x_0$ is the initial point.

.. index:: pair: variable; stop_r_relative
.. _doxid-structsnls__control__type_stop_r_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_r_relative

see stop_r_absolute

.. index:: pair: variable; stop_pg_absolute
.. _doxid-structsnls__control__type_stop_pg_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_pg_absolute

see stop_r_absolute

.. index:: pair: variable; stop_pg_relative
.. _doxid-structsnls__control__type_stop_pg_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_pg_relative

see stop_r_absolute

.. index:: pair: variable; stop_s
.. _doxid-structsnls__control__type_stop_s:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_s

see stop_r_absolute

.. index:: pair: variable; stop_pg_switch
.. _doxid-structsnls__control__type_stop_pg_switch:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_pg_switch

the step-computation solver will switch from an interior-point
method to a projected-gradient one if .subproblem_solver = 3 (see above) and
$\|P[x-g(x)]-x\|_2 \leq$
MAX( .stop_pg_absolute, .stop_pg_switch $* \|P[x_0-g(x_0)]-x_0\|_2$.

.. index:: pair: variable; initial_weight
.. _doxid-structsnls__control__type_initial_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_weight

initial value for the regularization weight (-ve => $1/\|g_0\|)$)

.. index:: pair: variable; minimum_weight
.. _doxid-structsnls__control__type_minimum_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` minimum_weight

minimum permitted regularization weight

.. index:: pair: variable; eta_successful
.. _doxid-structsnls__control__type_eta_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_successful

a potential iterate will only be accepted if the actual decrease f - f(x_new) is larger than .eta_successful times that predicted by a quadratic model of the decrease. The regularization weight will be decreaed if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful

.. index:: pair: variable; eta_very_successful
.. _doxid-structsnls__control__type_eta_very_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structsnls__control__type_eta_too_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_too_successful

see eta_successful

.. index:: pair: variable; weight_decrease_min
.. _doxid-structsnls__control__type_weight_decrease_min:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_decrease_min

on very successful iterations, the regularization weight will be reduced by the factor .weight_decrease but no more than .weight_decrease_min while if the iteration is unsucceful, the weight will be increased by a factor .weight_increase but no more than .weight_increase_max (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli and Toint, 2011)

.. index:: pair: variable; weight_decrease
.. _doxid-structsnls__control__type_weight_decrease:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_decrease

see weight_decrease_min

.. index:: pair: variable; weight_increase
.. _doxid-structsnls__control__type_weight_increase:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_increase

see weight_decrease_min

.. index:: pair: variable; weight_increase_max
.. _doxid-structsnls__control__type_weight_increase_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_increase_max

see weight_decrease_min

.. index:: pair: variable; switch_to_newton
.. _doxid-structsnls__control__type_switch_to_newton:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` switch_to_newton

if the value of the two-norm of the projected gradient is less than
.switch_to_newton, a switch is made from the Gauss-Newton model to the Newton 
one when .newton_acceleration is true

.. index:: pair: variable; cpu_time_limit
.. _doxid-structsnls__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structsnls__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; newton_acceleration
.. _doxid-structsnls__control__type_newton_acceleration:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool newton_acceleration

if they are available, second derivatives should be used to accelerate the convergence of the algorithm

.. index:: pair: variable; magic_step
.. _doxid-structsnls__control__type_magic_step:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool magic_step

allow the user to perform a "magic" step to improve the objective

.. index:: pair: variable; print_obj
.. _doxid-structsnls__control__type_print_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool print_obj

print values of the objective/gradient rather than $\|r\|$ and its gradient

.. index:: pair: variable; space_critical
.. _doxid-structsnls__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structsnls__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structsnls__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; slls_control
.. _doxid-structsnls__control__type_slls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`slls_control_type<doxid-structslls__control__type>` slls_control

control parameters for SLLS

.. index:: pair: variable; sllsb_control
.. _doxid-structsnls__control__type_sllsb_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sllsb_control_type<doxid-structsllsb__control__type>` sllsb_control

control parameters for SLLSB
