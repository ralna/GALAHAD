.. index:: pair: struct; nls_subproblem_control_type
.. _doxid-structnls__subproblem__control__type:

nls_subproblem_control_type structure
-------------------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_nls.h>
	
	struct nls_subproblem_control_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structnls__subproblem__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structnls__subproblem__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structnls__subproblem__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structnls__subproblem__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structnls__subproblem__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structnls__subproblem__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structnls__subproblem__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structnls__subproblem__control__type_alive_unit>`;
		char :ref:`alive_file<doxid-structnls__subproblem__control__type_alive_file>`[31];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`jacobian_available<doxid-structnls__subproblem__control__type_jacobian_available>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`hessian_available<doxid-structnls__subproblem__control__type_hessian_available>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`model<doxid-structnls__subproblem__control__type_model>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`norm<doxid-structnls__subproblem__control__type_norm>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`non_monotone<doxid-structnls__subproblem__control__type_non_monotone>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`weight_update_strategy<doxid-structnls__subproblem__control__type_weight_update_strategy>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_c_absolute<doxid-structnls__subproblem__control__type_stop_c_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_c_relative<doxid-structnls__subproblem__control__type_stop_c_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_g_absolute<doxid-structnls__subproblem__control__type_stop_g_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_g_relative<doxid-structnls__subproblem__control__type_stop_g_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_s<doxid-structnls__subproblem__control__type_stop_s>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`power<doxid-structnls__subproblem__control__type_power>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_weight<doxid-structnls__subproblem__control__type_initial_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`minimum_weight<doxid-structnls__subproblem__control__type_minimum_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_inner_weight<doxid-structnls__subproblem__control__type_initial_inner_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_successful<doxid-structnls__subproblem__control__type_eta_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_very_successful<doxid-structnls__subproblem__control__type_eta_very_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_too_successful<doxid-structnls__subproblem__control__type_eta_too_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_decrease_min<doxid-structnls__subproblem__control__type_weight_decrease_min>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_decrease<doxid-structnls__subproblem__control__type_weight_decrease>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_increase<doxid-structnls__subproblem__control__type_weight_increase>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_increase_max<doxid-structnls__subproblem__control__type_weight_increase_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce_gap<doxid-structnls__subproblem__control__type_reduce_gap>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`tiny_gap<doxid-structnls__subproblem__control__type_tiny_gap>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`large_root<doxid-structnls__subproblem__control__type_large_root>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`switch_to_newton<doxid-structnls__subproblem__control__type_switch_to_newton>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structnls__subproblem__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structnls__subproblem__control__type_clock_time_limit>`;
		bool :ref:`subproblem_direct<doxid-structnls__subproblem__control__type_subproblem_direct>`;
		bool :ref:`renormalize_weight<doxid-structnls__subproblem__control__type_renormalize_weight>`;
		bool :ref:`magic_step<doxid-structnls__subproblem__control__type_magic_step>`;
		bool :ref:`print_obj<doxid-structnls__subproblem__control__type_print_obj>`;
		bool :ref:`space_critical<doxid-structnls__subproblem__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structnls__subproblem__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structnls__subproblem__control__type_prefix>`[31];
		struct :ref:`rqs_control_type<doxid-structrqs__control__type>` :ref:`rqs_control<doxid-structnls__subproblem__control__type_rqs_control>`;
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>` :ref:`glrt_control<doxid-structnls__subproblem__control__type_glrt_control>`;
		struct :ref:`psls_control_type<doxid-structpsls__control__type>` :ref:`psls_control<doxid-structnls__subproblem__control__type_psls_control>`;
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>` :ref:`bsc_control<doxid-structnls__subproblem__control__type_bsc_control>`;
		struct :ref:`roots_control_type<doxid-structroots__control__type>` :ref:`roots_control<doxid-structnls__subproblem__control__type_roots_control>`;
	};
.. _details-structnls__subproblem__control__type:

detailed documentation
----------------------

subproblem_control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; error
.. _doxid-structnls__subproblem__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structnls__subproblem__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structnls__subproblem__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structnls__subproblem__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structnls__subproblem__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structnls__subproblem__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structnls__subproblem__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

the maximum number of iterations performed

.. index:: pair: variable; alive_unit
.. _doxid-structnls__subproblem__control__type_alive_unit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structnls__subproblem__control__type_alive_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; jacobian_available
.. _doxid-structnls__subproblem__control__type_jacobian_available:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` jacobian_available

is the Jacobian matrix of first derivatives available ($\geq$ 2), is access only via matrix-vector products (=1) or is it not available ($\leq$ 0) ?

.. index:: pair: variable; hessian_available
.. _doxid-structnls__subproblem__control__type_hessian_available:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` hessian_available

is the Hessian matrix of second derivatives available ($\geq$ 2), is access only via matrix-vector products (=1) or is it not available ($\leq$ 0) ?

.. index:: pair: variable; model
.. _doxid-structnls__subproblem__control__type_model:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` model

the model used.

Possible values are

* 0 dynamic (*not yet implemented*)

* 1 first-order (no Hessian)

* 2 barely second-order (identity Hessian)

* 3 Gauss-Newton ($J^T J$ Hessian)

* 4 second-order (exact Hessian)

* 5 Gauss-Newton to Newton transition

* 6 tensor Gauss-Newton treated as a least-squares model

* 7 tensor Gauss-Newton treated as a general model

* 8 tensor Gauss-Newton transition from a least-squares to a general mode

.. index:: pair: variable; norm
.. _doxid-structnls__subproblem__control__type_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` norm

the regularization norm used.

The norm is defined via $\|v\|^2 = v^T S v$, and will define the preconditioner used for iterative methods. Possible values for $S$ are

* -3 user's own regularization norm

* -2 $S$ = limited-memory BFGS matrix (with .PSLS_control.lbfgs_vectors history) (*not yet implemented*)

* -1 identity (= Euclidan two-norm)

* 0 automatic (*not yet implemented*)

* 1 diagonal, $S$ = diag( max($J^TJ$ Hessian, .PSLS_control.min_diagonal ) )

* 2 diagonal, $S$ = diag( max( Hessian, .PSLS_control.min_diagonal ) )

* 3 banded, $S$ = band( Hessian ) with semi-bandwidth .PSLS_control.semi_bandwidth

* 4 re-ordered band, P=band(order(A)) with semi-bandwidth .PSLS_control.semi_bandwidth

* 5 full factorization, $S$ = Hessian, Schnabel-Eskow modification

* 6 full factorization, $S$ = Hessian, GMPS modification (*not yet implemented*)

* 7 incomplete factorization of Hessian, Lin-More'

* 8 incomplete factorization of Hessian, HSL_MI28

* 9 incomplete factorization of Hessian, Munskgaard (*not yet implemented*)

* 10 expanding band of Hessian (*not yet implemented*)

.. index:: pair: variable; non_monotone
.. _doxid-structnls__subproblem__control__type_non_monotone:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` non_monotone

non-monotone $\leq$ 0 monotone strategy used, anything else non-monotone strategy with this history length used

.. index:: pair: variable; weight_update_strategy
.. _doxid-structnls__subproblem__control__type_weight_update_strategy:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` weight_update_strategy

define the weight-update strategy: 1 (basic), 2 (reset to zero when very successful), 3 (imitate TR), 4 (increase lower bound), 5 (GPT)

.. index:: pair: variable; stop_c_absolute
.. _doxid-structnls__subproblem__control__type_stop_c_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_c_absolute

overall convergence tolerances. The iteration will terminate when $||c(x)||_2 \leq$ MAX( .stop_c_absolute, .stop_c_relative $* \|c(x_{\mbox{initial}})\|_2$, or when the norm of the gradient, $g = J^T(x) c(x) / \|c(x)\|_2$, of \|\|c\|\|_2, satisfies $\|g\|_2 \leq$ MAX( .stop_g_absolute, .stop_g_relative $* \|g_{\mbox{initial}}\|_2$, or if the step is less than .stop_s

.. index:: pair: variable; stop_c_relative
.. _doxid-structnls__subproblem__control__type_stop_c_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_c_relative

see stop_c_absolute

.. index:: pair: variable; stop_g_absolute
.. _doxid-structnls__subproblem__control__type_stop_g_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_g_absolute

see stop_c_absolute

.. index:: pair: variable; stop_g_relative
.. _doxid-structnls__subproblem__control__type_stop_g_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_g_relative

see stop_c_absolute

.. index:: pair: variable; stop_s
.. _doxid-structnls__subproblem__control__type_stop_s:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_s

see stop_c_absolute

.. index:: pair: variable; power
.. _doxid-structnls__subproblem__control__type_power:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` power

the regularization power (<2 => chosen according to the model)

.. index:: pair: variable; initial_weight
.. _doxid-structnls__subproblem__control__type_initial_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_weight

initial value for the regularization weight (-ve => $1/\|g_0\|)$)

.. index:: pair: variable; minimum_weight
.. _doxid-structnls__subproblem__control__type_minimum_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` minimum_weight

minimum permitted regularization weight

.. index:: pair: variable; initial_inner_weight
.. _doxid-structnls__subproblem__control__type_initial_inner_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_inner_weight

initial value for the inner regularization weight for tensor GN (-ve => 0)

.. index:: pair: variable; eta_successful
.. _doxid-structnls__subproblem__control__type_eta_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_successful

a potential iterate will only be accepted if the actual decrease f - f(x_new) is larger than .eta_successful times that predicted by a quadratic model of the decrease. The regularization weight will be decreaed if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful

.. index:: pair: variable; eta_very_successful
.. _doxid-structnls__subproblem__control__type_eta_very_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structnls__subproblem__control__type_eta_too_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_too_successful

see eta_successful

.. index:: pair: variable; weight_decrease_min
.. _doxid-structnls__subproblem__control__type_weight_decrease_min:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_decrease_min

on very successful iterations, the regularization weight will be reduced by the factor .weight_decrease but no more than .weight_decrease_min while if the iteration is unsucceful, the weight will be increased by a factor .weight_increase but no more than .weight_increase_max (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli and Toint, 2011)

.. index:: pair: variable; weight_decrease
.. _doxid-structnls__subproblem__control__type_weight_decrease:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_decrease

see weight_decrease_min

.. index:: pair: variable; weight_increase
.. _doxid-structnls__subproblem__control__type_weight_increase:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_increase

see weight_decrease_min

.. index:: pair: variable; weight_increase_max
.. _doxid-structnls__subproblem__control__type_weight_increase_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_increase_max

see weight_decrease_min

.. index:: pair: variable; reduce_gap
.. _doxid-structnls__subproblem__control__type_reduce_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce_gap

expert parameters as suggested in Gould, Porcelli and Toint, "Updating t
regularization parameter in the adaptive cubic regularization algorithm" RAL-TR-2011-007, Rutherford Appleton Laboratory, England (2011), `http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf <http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf>`__ (these are denoted beta, epsilon_chi and alpha_max in the paper)

.. index:: pair: variable; tiny_gap
.. _doxid-structnls__subproblem__control__type_tiny_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` tiny_gap

see reduce_gap

.. index:: pair: variable; large_root
.. _doxid-structnls__subproblem__control__type_large_root:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` large_root

see reduce_gap

.. index:: pair: variable; switch_to_newton
.. _doxid-structnls__subproblem__control__type_switch_to_newton:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` switch_to_newton

if the Gauss-Newto to Newton model is specified, switch to Newton as soon as the norm of the gradient g is smaller than switch_to_newton

.. index:: pair: variable; cpu_time_limit
.. _doxid-structnls__subproblem__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structnls__subproblem__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; subproblem_direct
.. _doxid-structnls__subproblem__control__type_subproblem_direct:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method to find the search direction

.. index:: pair: variable; renormalize_weight
.. _doxid-structnls__subproblem__control__type_renormalize_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool renormalize_weight

should the weight be renormalized to account for a change in scaling?

.. index:: pair: variable; magic_step
.. _doxid-structnls__subproblem__control__type_magic_step:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool magic_step

allow the user to perform a "magic" step to improve the objective

.. index:: pair: variable; print_obj
.. _doxid-structnls__subproblem__control__type_print_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool print_obj

print values of the objective/gradient rather than \|\|c\|\| and its gradien

.. index:: pair: variable; space_critical
.. _doxid-structnls__subproblem__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structnls__subproblem__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structnls__subproblem__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; rqs_control
.. _doxid-structnls__subproblem__control__type_rqs_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rqs_control_type<doxid-structrqs__control__type>` rqs_control

control parameters for RQS

.. index:: pair: variable; glrt_control
.. _doxid-structnls__subproblem__control__type_glrt_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`glrt_control_type<doxid-structglrt__control__type>` glrt_control

control parameters for GLRT

.. index:: pair: variable; psls_control
.. _doxid-structnls__subproblem__control__type_psls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_control_type<doxid-structpsls__control__type>` psls_control

control parameters for PSLS

.. index:: pair: variable; bsc_control
.. _doxid-structnls__subproblem__control__type_bsc_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bsc_control_type<doxid-structbsc__control__type>` bsc_control

control parameters for BSC

.. index:: pair: variable; roots_control
.. _doxid-structnls__subproblem__control__type_roots_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`roots_control_type<doxid-structroots__control__type>` roots_control

control parameters for ROOTS

