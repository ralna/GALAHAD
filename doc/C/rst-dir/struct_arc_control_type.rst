.. index:: pair: struct; arc_control_type
.. _doxid-structarc__control__type:

arc_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_arc.h>
	
	struct arc_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structarc__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structarc__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structarc__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structarc__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structarc__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structarc__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structarc__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structarc__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structarc__control__type_alive_unit>`;
		char :ref:`alive_file<doxid-structarc__control__type_alive_file>`[31];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`non_monotone<doxid-structarc__control__type_non_monotone>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`model<doxid-structarc__control__type_model>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`norm<doxid-structarc__control__type_norm>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structarc__control__type_semi_bandwidth>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`lbfgs_vectors<doxid-structarc__control__type_lbfgs_vectors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_dxg<doxid-structarc__control__type_max_dxg>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`icfs_vectors<doxid-structarc__control__type_icfs_vectors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_lsize<doxid-structarc__control__type_mi28_lsize>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_rsize<doxid-structarc__control__type_mi28_rsize>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`advanced_start<doxid-structarc__control__type_advanced_start>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_g_absolute<doxid-structarc__control__type_stop_g_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_g_relative<doxid-structarc__control__type_stop_g_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_s<doxid-structarc__control__type_stop_s>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_weight<doxid-structarc__control__type_initial_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`minimum_weight<doxid-structarc__control__type_minimum_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce_gap<doxid-structarc__control__type_reduce_gap>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`tiny_gap<doxid-structarc__control__type_tiny_gap>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`large_root<doxid-structarc__control__type_large_root>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_successful<doxid-structarc__control__type_eta_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_very_successful<doxid-structarc__control__type_eta_very_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_too_successful<doxid-structarc__control__type_eta_too_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_decrease_min<doxid-structarc__control__type_weight_decrease_min>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_decrease<doxid-structarc__control__type_weight_decrease>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_increase<doxid-structarc__control__type_weight_increase>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_increase_max<doxid-structarc__control__type_weight_increase_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_unbounded<doxid-structarc__control__type_obj_unbounded>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structarc__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structarc__control__type_clock_time_limit>`;
		bool :ref:`hessian_available<doxid-structarc__control__type_hessian_available>`;
		bool :ref:`subproblem_direct<doxid-structarc__control__type_subproblem_direct>`;
		bool :ref:`renormalize_weight<doxid-structarc__control__type_renormalize_weight>`;
		bool :ref:`quadratic_ratio_test<doxid-structarc__control__type_quadratic_ratio_test>`;
		bool :ref:`space_critical<doxid-structarc__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structarc__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structarc__control__type_prefix>`[31];
		struct :ref:`rqs_control_type<doxid-structrqs__control__type>` :ref:`rqs_control<doxid-structarc__control__type_rqs_control>`;
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>` :ref:`glrt_control<doxid-structarc__control__type_glrt_control>`;
		struct :ref:`dps_control_type<doxid-structdps__control__type>` :ref:`dps_control<doxid-structarc__control__type_dps_control>`;
		struct :ref:`psls_control_type<doxid-structpsls__control__type>` :ref:`psls_control<doxid-structarc__control__type_psls_control>`;
		struct :ref:`lms_control_type<doxid-structlms__control__type>` :ref:`lms_control<doxid-structarc__control__type_lms_control>`;
		struct :ref:`lms_control_type<doxid-structlms__control__type>` :target:`lms_control_prec<doxid-structarc__control__type_lms_control_prec>`;
		struct :ref:`sha_control_type<doxid-structsha__control__type>` :ref:`sha_control<doxid-structarc__control__type_sha_control>`;
	};
.. _details-structarc__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structarc__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structarc__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structarc__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structarc__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structarc__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structarc__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structarc__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structarc__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

the maximum number of iterations performed

.. index:: pair: variable; alive_unit
.. _doxid-structarc__control__type_alive_unit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structarc__control__type_alive_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; non_monotone
.. _doxid-structarc__control__type_non_monotone:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` non_monotone

the descent strategy used.

Possible values are

* <= 0 a monotone strategy is used.

* anything else, a non-monotone strategy with history length .non_monotine is used.

.. index:: pair: variable; model
.. _doxid-structarc__control__type_model:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` model

the model used.

Possible values are

* 0 dynamic (*not yet implemented*)

* 1 first-order (no Hessian)

* 2 second-order (exact Hessian)

* 3 barely second-order (identity Hessian)

* 4 secant second-order (limited-memory BFGS, with .lbfgs_vectors history) (*not yet implemented*)

* 5 secant second-order (limited-memory SR1, with .lbfgs_vectors history) (*not yet implemented*)

.. index:: pair: variable; norm
.. _doxid-structarc__control__type_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` norm

the regularization norm used.

The norm is defined via $\|v\|^2 = v^T P v$, and will define the preconditioner used for iterative methods. Possible values for $P$ are

* -3 users own preconditioner

* -2 $P =$ limited-memory BFGS matrix (with .lbfgs_vectors history)

* -1 identity (= Euclidan two-norm)

* 0 automatic (*not yet implemented*)

* 1 diagonal, $P =$ diag( max( Hessian, .min_diagonal ) )

* 2 banded, $P =$ band( Hessian ) with semi-bandwidth .semi_bandwidth

* 3 re-ordered band, P=band(order(A)) with semi-bandwidth .semi_bandwidth

* 4 full factorization, $P =$ Hessian, Schnabel-Eskow modification

* 5 full factorization, $P =$ Hessian, GMPS modification (*not yet implemented*)

* 6 incomplete factorization of Hessian, Lin-More'

* 7 incomplete factorization of Hessian, HSL_MI28

* 8 incomplete factorization of Hessian, Munskgaard (*not yet implemented*)

* 9 expanding band of Hessian (*not yet implemented*)

* 10 diagonalizing norm from GALAHAD_DPS (*subproblem_direct only*)

.. index:: pair: variable; semi_bandwidth
.. _doxid-structarc__control__type_semi_bandwidth:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

specify the semi-bandwidth of the band matrix P if required

.. index:: pair: variable; lbfgs_vectors
.. _doxid-structarc__control__type_lbfgs_vectors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` lbfgs_vectors

number of vectors used by the L-BFGS matrix P if required

.. index:: pair: variable; max_dxg
.. _doxid-structarc__control__type_max_dxg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_dxg

number of vectors used by the sparsity-based secant Hessian if required

.. index:: pair: variable; icfs_vectors
.. _doxid-structarc__control__type_icfs_vectors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` icfs_vectors

number of vectors used by the Lin-More' incomplete factorization matrix P if required

.. index:: pair: variable; mi28_lsize
.. _doxid-structarc__control__type_mi28_lsize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_lsize

the maximum number of fill entries within each column of the incomplete factor L computed by HSL_MI28. In general, increasing .mi28_lsize improve the quality of the preconditioner but increases the time to compute and then apply the preconditioner. Values less than 0 are treated as 0

.. index:: pair: variable; mi28_rsize
.. _doxid-structarc__control__type_mi28_rsize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_rsize

the maximum number of entries within each column of the strictly lower triangular matrix $R$ used in the computation of the preconditioner by HSL_MI28. Rank-1 arrays of size .mi28_rsize \* n are allocated internally to hold $R$. Thus the amount of memory used, as well as the amount of work involved in computing the preconditioner, depends on .mi28_rsize. Setting .mi28_rsize > 0 generally leads to a higher quality preconditioner than using .mi28_rsize = 0, and choosing .mi28_rsize >= .mi28_lsize is generally recommended

.. index:: pair: variable; advanced_start
.. _doxid-structarc__control__type_advanced_start:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` advanced_start

try to pick a good initial regularization weight using .advanced_start iterates of a variant on the strategy of Sartenaer SISC 18(6) 1990:1788-1803

.. index:: pair: variable; stop_g_absolute
.. _doxid-structarc__control__type_stop_g_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_g_absolute

overall convergence tolerances. The iteration will terminate when the norm of the gradient of the objective function is smaller than MAX( .stop_g_absolute, .stop_g_relative \* norm of the initial gradient ) or if the step is less than .stop_s

.. index:: pair: variable; stop_g_relative
.. _doxid-structarc__control__type_stop_g_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_g_relative

see stop_g_absolute

.. index:: pair: variable; stop_s
.. _doxid-structarc__control__type_stop_s:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_s

see stop_g_absolute

.. index:: pair: variable; initial_weight
.. _doxid-structarc__control__type_initial_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_weight

Initial value for the regularisation weight (-ve => 1/\|\|g_0\|\|)

.. index:: pair: variable; minimum_weight
.. _doxid-structarc__control__type_minimum_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` minimum_weight

minimum permitted regularisation weight

.. index:: pair: variable; reduce_gap
.. _doxid-structarc__control__type_reduce_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce_gap

expert parameters as suggested in Gould, Porcelli & Toint, "Updating the
regularization parameter in the adaptive cubic regularization algorithm" RAL-TR-2011-007, Rutherford Appleton Laboratory, England (2011), `http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf <http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf>`__ (these are denoted beta, epsilon_chi and alpha_max in the paper)

.. index:: pair: variable; tiny_gap
.. _doxid-structarc__control__type_tiny_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` tiny_gap

see reduce_gap

.. index:: pair: variable; large_root
.. _doxid-structarc__control__type_large_root:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` large_root

see reduce_gap

.. index:: pair: variable; eta_successful
.. _doxid-structarc__control__type_eta_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_successful

a potential iterate will only be accepted if the actual decrease f - f(x_new) is larger than .eta_successful times that predicted by a quadratic model of the decrease. The regularization weight will be decreased if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful (the first is eta in Gould, Porcell and Toint, 2011)

.. index:: pair: variable; eta_very_successful
.. _doxid-structarc__control__type_eta_very_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structarc__control__type_eta_too_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_too_successful

see eta_successful

.. index:: pair: variable; weight_decrease_min
.. _doxid-structarc__control__type_weight_decrease_min:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_decrease_min

on very successful iterations, the regularization weight will be reduced by the factor .weight_decrease but no more than .weight_decrease_min while if the iteration is unsuccessful, the weight will be increased by a factor .weight_increase but no more than .weight_increase_max (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli and Toint, 2011)

.. index:: pair: variable; weight_decrease
.. _doxid-structarc__control__type_weight_decrease:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_decrease

see weight_decrease_min

.. index:: pair: variable; weight_increase
.. _doxid-structarc__control__type_weight_increase:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_increase

see weight_decrease_min

.. index:: pair: variable; weight_increase_max
.. _doxid-structarc__control__type_weight_increase_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_increase_max

see weight_decrease_min

.. index:: pair: variable; obj_unbounded
.. _doxid-structarc__control__type_obj_unbounded:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_unbounded

the smallest value the onjective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structarc__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structarc__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structarc__control__type_hessian_available:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; subproblem_direct
.. _doxid-structarc__control__type_subproblem_direct:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method to find the search direction

.. index:: pair: variable; renormalize_weight
.. _doxid-structarc__control__type_renormalize_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool renormalize_weight

should the weight be renormalized to account for a change in preconditioner?

.. index:: pair: variable; quadratic_ratio_test
.. _doxid-structarc__control__type_quadratic_ratio_test:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool quadratic_ratio_test

should the test for acceptance involve the quadratic model or the cubic?

.. index:: pair: variable; space_critical
.. _doxid-structarc__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structarc__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structarc__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; rqs_control
.. _doxid-structarc__control__type_rqs_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rqs_control_type<doxid-structrqs__control__type>` rqs_control

control parameters for RQS

.. index:: pair: variable; glrt_control
.. _doxid-structarc__control__type_glrt_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`glrt_control_type<doxid-structglrt__control__type>` glrt_control

control parameters for GLRT

.. index:: pair: variable; dps_control
.. _doxid-structarc__control__type_dps_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`dps_control_type<doxid-structdps__control__type>` dps_control

control parameters for DPS

.. index:: pair: variable; psls_control
.. _doxid-structarc__control__type_psls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_control_type<doxid-structpsls__control__type>` psls_control

control parameters for PSLS

.. index:: pair: variable; lms_control
.. _doxid-structarc__control__type_lms_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control
control parameters for LMS

.. index:: pair: variable; lms_control_prec
.. _doxid-structarc__control__type_lms_control_prec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control_prec

control parameters for LMS used for preconditioning

.. index:: pair: variable; sha_control
.. _doxid-structarc__control__type_sha_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sha_control_type<doxid-structsha__control__type>` sha_control

control parameters for SHA

