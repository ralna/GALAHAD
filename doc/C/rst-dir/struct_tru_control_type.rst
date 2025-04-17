.. index:: pair: struct; tru_control_type
.. _doxid-structtru__control__type:

tru_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_tru.h>
	
	struct tru_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structtru__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structtru__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structtru__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structtru__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structtru__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structtru__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structtru__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structtru__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structtru__control__type_alive_unit>`;
		char :ref:`alive_file<doxid-structtru__control__type_alive_file>`[31];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`non_monotone<doxid-structtru__control__type_non_monotone>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`model<doxid-structtru__control__type_model>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`norm<doxid-structtru__control__type_norm>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structtru__control__type_semi_bandwidth>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`lbfgs_vectors<doxid-structtru__control__type_lbfgs_vectors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_dxg<doxid-structtru__control__type_max_dxg>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`icfs_vectors<doxid-structtru__control__type_icfs_vectors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_lsize<doxid-structtru__control__type_mi28_lsize>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_rsize<doxid-structtru__control__type_mi28_rsize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_g_absolute<doxid-structtru__control__type_stop_g_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_g_relative<doxid-structtru__control__type_stop_g_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_s<doxid-structtru__control__type_stop_s>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`advanced_start<doxid-structtru__control__type_advanced_start>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_radius<doxid-structtru__control__type_initial_radius>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`maximum_radius<doxid-structtru__control__type_maximum_radius>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_successful<doxid-structtru__control__type_eta_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_very_successful<doxid-structtru__control__type_eta_very_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_too_successful<doxid-structtru__control__type_eta_too_successful>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius_increase<doxid-structtru__control__type_radius_increase>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius_reduce<doxid-structtru__control__type_radius_reduce>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius_reduce_max<doxid-structtru__control__type_radius_reduce_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_unbounded<doxid-structtru__control__type_obj_unbounded>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structtru__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structtru__control__type_clock_time_limit>`;
		bool :ref:`hessian_available<doxid-structtru__control__type_hessian_available>`;
		bool :ref:`subproblem_direct<doxid-structtru__control__type_subproblem_direct>`;
		bool :ref:`retrospective_trust_region<doxid-structtru__control__type_retrospective_trust_region>`;
		bool :ref:`renormalize_radius<doxid-structtru__control__type_renormalize_radius>`;
		bool :ref:`space_critical<doxid-structtru__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structtru__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structtru__control__type_prefix>`[31];
		struct :ref:`trs_control_type<doxid-structtrs__control__type>` :ref:`trs_control<doxid-structtru__control__type_trs_control>`;
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>` :ref:`gltr_control<doxid-structtru__control__type_gltr_control>`;
		struct :ref:`dps_control_type<doxid-structdps__control__type>` :ref:`dps_control<doxid-structtru__control__type_dps_control>`;
		struct :ref:`psls_control_type<doxid-structpsls__control__type>` :ref:`psls_control<doxid-structtru__control__type_psls_control>`;
		struct :ref:`lms_control_type<doxid-structlms__control__type>` :ref:`lms_control<doxid-structtru__control__type_lms_control>`;
		struct :ref:`lms_control_type<doxid-structlms__control__type>` :target:`lms_control_prec<doxid-structtru__control__type_lms_control_prec>`;
		struct :ref:`sec_control_type<doxid-structsec__control__type>` :ref:`sec_control<doxid-structtru__control__type_sec_control>`;
		struct :ref:`sha_control_type<doxid-structsha__control__type>` :ref:`sha_control<doxid-structtru__control__type_sha_control>`;
	};
.. _details-structtru__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structtru__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structtru__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structtru__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structtru__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structtru__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structtru__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structtru__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structtru__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

the maximum number of iterations allowed

.. index:: pair: variable; alive_unit
.. _doxid-structtru__control__type_alive_unit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structtru__control__type_alive_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; non_monotone
.. _doxid-structtru__control__type_non_monotone:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` non_monotone

the descent strategy used.

Possible values are

* <= 0 a monotone strategy is used.

* anything else, a non-monotone strategy with history length .non_monotine is used.

.. index:: pair: variable; model
.. _doxid-structtru__control__type_model:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` model

the model used.

Possible values are

* 0 dynamic (*not yet implemented*)

* 1 first-order (no Hessian)

* 2 second-order (exact Hessian)

* 3 barely second-order (identity Hessian)

* 4 secant second-order (sparsity-based)

* 5 secant second-order (limited-memory BFGS, with .lbfgs_vectors history)

* 6 secant second-order (limited-memory SR1, with .lbfgs_vectors history)

.. index:: pair: variable; norm
.. _doxid-structtru__control__type_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` norm

the trust-region norm used.

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
.. _doxid-structtru__control__type_semi_bandwidth:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

specify the semi-bandwidth of the band matrix $P$ if required

.. index:: pair: variable; lbfgs_vectors
.. _doxid-structtru__control__type_lbfgs_vectors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` lbfgs_vectors

number of vectors used by the L-BFGS matrix $P$ if required

.. index:: pair: variable; max_dxg
.. _doxid-structtru__control__type_max_dxg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_dxg

number of vectors used by the sparsity-based secant Hessian if required

.. index:: pair: variable; icfs_vectors
.. _doxid-structtru__control__type_icfs_vectors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` icfs_vectors

number of vectors used by the Lin-More' incomplete factorization matrix $P$ if required

.. index:: pair: variable; mi28_lsize
.. _doxid-structtru__control__type_mi28_lsize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_lsize

the maximum number of fill entries within each column of the incomplete factor L computed by HSL_MI28. In general, increasing .mi28_lsize improve the quality of the preconditioner but increases the time to compute and then apply the preconditioner. Values less than 0 are treated as 0

.. index:: pair: variable; mi28_rsize
.. _doxid-structtru__control__type_mi28_rsize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_rsize

the maximum number of entries within each column of the strictly lower triangular matrix $R$ used in the computation of the preconditioner by HSL_MI28. Rank-1 arrays of size .mi28_rsize \* n are allocated internally to hold $R$. Thus the amount of memory used, as well as the amount of work involved in computing the preconditioner, depends on .mi28_rsize. Setting .mi28_rsize > 0 generally leads to a higher quality preconditioner than using .mi28_rsize = 0, and choosing .mi28_rsize >= .mi28_lsize is generally recommended

.. index:: pair: variable; stop_g_absolute
.. _doxid-structtru__control__type_stop_g_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_g_absolute

overall convergence tolerances. The iteration will terminate when the norm of the gradient of the objective function is smaller than MAX( .stop_g_absolute, .stop_g_relative \* norm of the initial gradient ) or if the step is less than .stop_s

.. index:: pair: variable; stop_g_relative
.. _doxid-structtru__control__type_stop_g_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_g_relative

see stop_g_absolute

.. index:: pair: variable; stop_s
.. _doxid-structtru__control__type_stop_s:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_s

see stop_g_absolute

.. index:: pair: variable; advanced_start
.. _doxid-structtru__control__type_advanced_start:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` advanced_start

try to pick a good initial trust-region radius using .advanced_start iterates of a variant on the strategy of Sartenaer SISC 18(6) 1990:1788-1803

.. index:: pair: variable; initial_radius
.. _doxid-structtru__control__type_initial_radius:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_radius

initial value for the trust-region radius

.. index:: pair: variable; maximum_radius
.. _doxid-structtru__control__type_maximum_radius:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` maximum_radius

maximum permitted trust-region radius

.. index:: pair: variable; eta_successful
.. _doxid-structtru__control__type_eta_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_successful

a potential iterate will only be accepted if the actual decrease $f - f(x_{new})$ is larger than .eta_successful times that predicted by a quadratic model of the decrease. The trust-region radius will be increased if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful

.. index:: pair: variable; eta_very_successful
.. _doxid-structtru__control__type_eta_very_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structtru__control__type_eta_too_successful:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_too_successful

see eta_successful

.. index:: pair: variable; radius_increase
.. _doxid-structtru__control__type_radius_increase:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius_increase

on very successful iterations, the trust-region radius will be increased by the factor .radius_increase, while if the iteration is unsucceful, the radius will be decreased by a factor .radius_reduce but no more than .radius_reduce_max

.. index:: pair: variable; radius_reduce
.. _doxid-structtru__control__type_radius_reduce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius_reduce

see radius_increase;

.. index:: pair: variable; radius_reduce_max
.. _doxid-structtru__control__type_radius_reduce_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius_reduce_max

see radius_increase;

.. index:: pair: variable; obj_unbounded
.. _doxid-structtru__control__type_obj_unbounded:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structtru__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structtru__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structtru__control__type_hessian_available:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; subproblem_direct
.. _doxid-structtru__control__type_subproblem_direct:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method to find the search direction

.. index:: pair: variable; retrospective_trust_region
.. _doxid-structtru__control__type_retrospective_trust_region:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool retrospective_trust_region

is a retrospective strategy to be used to update the trust-region radius?

.. index:: pair: variable; renormalize_radius
.. _doxid-structtru__control__type_renormalize_radius:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool renormalize_radius

should the radius be renormalized to account for a change in preconditioner?

.. index:: pair: variable; space_critical
.. _doxid-structtru__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structtru__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structtru__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; trs_control
.. _doxid-structtru__control__type_trs_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trs_control_type<doxid-structtrs__control__type>` trs_control

control parameters for TRS

.. index:: pair: variable; gltr_control
.. _doxid-structtru__control__type_gltr_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

.. index:: pair: variable; dps_control
.. _doxid-structtru__control__type_dps_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`dps_control_type<doxid-structdps__control__type>` dps_control

control parameters for DPS

.. index:: pair: variable; psls_control
.. _doxid-structtru__control__type_psls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_control_type<doxid-structpsls__control__type>` psls_control

control parameters for PSLS

.. index:: pair: variable; lms_control
.. _doxid-structtru__control__type_lms_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control

control parameters for LMS

.. index:: pair: variable; lms_control_prec
.. _doxid-structtru__control__type_lms_control_prec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control_prec

control parameters for LMS used for preconditioning

.. index:: pair: variable; sec_control
.. _doxid-structtru__control__type_sec_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sec_control_type<doxid-structsec__control__type>` sec_control

control parameters for SEC

.. index:: pair: variable; sha_control
.. _doxid-structtru__control__type_sha_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sha_control_type<doxid-structsha__control__type>` sha_control

control parameters for SHA

