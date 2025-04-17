.. index:: pair: struct; trb_control_type
.. _doxid-structtrb__control__type:

trb_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct trb_control_type{T,INT}
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
          more_toraldo::INT
          non_monotone::INT
          model::INT
          norm::INT
          semi_bandwidth::INT
          lbfgs_vectors::INT
          max_dxg::INT
          icfs_vectors::INT
          mi28_lsize::INT
          mi28_rsize::INT
          advanced_start::INT
          infinity::T
          stop_pg_absolute::T
          stop_pg_relative::T
          stop_s::T
          initial_radius::T
          maximum_radius::T
          stop_rel_cg::T
          eta_successful::T
          eta_very_successful::T
          eta_too_successful::T
          radius_increase::T
          radius_reduce::T
          radius_reduce_max::T
          obj_unbounded::T
          cpu_time_limit::T
          clock_time_limit::T
          hessian_available::Bool
          subproblem_direct::Bool
          retrospective_trust_region::Bool
          renormalize_radius::Bool
          two_norm_tr::Bool
          exact_gcp::Bool
          accurate_bqp::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}
          trs_control::trs_control_type{T,INT}
          gltr_control::gltr_control_type{T,INT}
          psls_control::psls_control_type{T,INT}
          lms_control::lms_control_type{INT}
          lms_control_prec::lms_control_type{INT}
          sha_control::sha_control_type{INT}

.. _details-structtrb__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structtrb__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structtrb__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structtrb__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structtrb__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structtrb__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structtrb__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structtrb__control__type_print_gap:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structtrb__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

the maximum number of iterations performed

.. index:: pair: variable; alive_unit
.. _doxid-structtrb__control__type_alive_unit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structtrb__control__type_alive_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; more_toraldo
.. _doxid-structtrb__control__type_more_toraldo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more_toraldo

more_toraldo >= 1 gives the number of More'-Toraldo projected searches to be used to improve upon the Cauchy point, anything else is for the standard add-one-at-a-time CG search

.. index:: pair: variable; non_monotone
.. _doxid-structtrb__control__type_non_monotone:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT non_monotone

non-monotone <= 0 monotone strategy used, anything else non-monotone strategy with this history length used

.. index:: pair: variable; model
.. _doxid-structtrb__control__type_model:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT model

the model used.

Possible values are

* 0 dynamic (*not yet implemented*)

* 1 first-order (no Hessian)

* 2 second-order (exact Hessian)

* 3 barely second-order (identity Hessian)

* 4 secant second-order (sparsity-based)

* 5 secant second-order (limited-memory BFGS, with .lbfgs_vectors history) (*not yet implemented*)

* 6 secant second-order (limited-memory SR1, with .lbfgs_vectors history) (*not yet implemented*)

.. index:: pair: variable; norm
.. _doxid-structtrb__control__type_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT norm

The norm is defined via $\|v\|^2 = v^T P v$, and will define the preconditioner used for iterative methods. Possible values for $P$ are.

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

.. index:: pair: variable; semi_bandwidth
.. _doxid-structtrb__control__type_semi_bandwidth:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth

specify the semi-bandwidth of the band matrix P if required

.. index:: pair: variable; lbfgs_vectors
.. _doxid-structtrb__control__type_lbfgs_vectors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lbfgs_vectors

number of vectors used by the L-BFGS matrix P if required

.. index:: pair: variable; max_dxg
.. _doxid-structtrb__control__type_max_dxg:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_dxg

number of vectors used by the sparsity-based secant Hessian if required

.. index:: pair: variable; icfs_vectors
.. _doxid-structtrb__control__type_icfs_vectors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT icfs_vectors

number of vectors used by the Lin-More' incomplete factorization matrix P if required

.. index:: pair: variable; mi28_lsize
.. _doxid-structtrb__control__type_mi28_lsize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mi28_lsize

the maximum number of fill entries within each column of the incomplete factor L computed by HSL_MI28. In general, increasing .mi28_lsize improve the quality of the preconditioner but increases the time to compute and then apply the preconditioner. Values less than 0 are treated as 0

.. index:: pair: variable; mi28_rsize
.. _doxid-structtrb__control__type_mi28_rsize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mi28_rsize

the maximum number of entries within each column of the strictly lower triangular matrix $R$ used in the computation of the preconditioner by HSL_MI28. Rank-1 arrays of size .mi28_rsize \* n are allocated internally to hold $R$. Thus the amount of memory used, as well as the amount of work involved in computing the preconditioner, depends on .mi28_rsize. Setting .mi28_rsize > 0 generally leads to a higher quality preconditioner than using .mi28_rsize = 0, and choosing .mi28_rsize >= .mi28_lsize is generally recommended

.. index:: pair: variable; advanced_start
.. _doxid-structtrb__control__type_advanced_start:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT advanced_start

iterates of a variant on the strategy of Sartenaer SISC 18(6)1990:1788-1803

.. index:: pair: variable; infinity
.. _doxid-structtrb__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_pg_absolute
.. _doxid-structtrb__control__type_stop_pg_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_pg_absolute

overall convergence tolerances. The iteration will terminate when the norm of the gradient of the objective function is smaller than MAX( .stop_pg_absolute, .stop_pg_relative \* norm of the initial gradient ) or if the step is less than .stop_s

.. index:: pair: variable; stop_pg_relative
.. _doxid-structtrb__control__type_stop_pg_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_pg_relative

see stop_pg_absolute

.. index:: pair: variable; stop_s
.. _doxid-structtrb__control__type_stop_s:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_s

see stop_pg_absolute

.. index:: pair: variable; initial_radius
.. _doxid-structtrb__control__type_initial_radius:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T initial_radius

initial value for the trust-region radius

.. index:: pair: variable; maximum_radius
.. _doxid-structtrb__control__type_maximum_radius:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T maximum_radius

maximum permitted trust-region radius

.. index:: pair: variable; stop_rel_cg
.. _doxid-structtrb__control__type_stop_rel_cg:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_rel_cg

required relative reduction in the resuiduals from CG

.. index:: pair: variable; eta_successful
.. _doxid-structtrb__control__type_eta_successful:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_successful

a potential iterate will only be accepted if the actual decrease f - f(x_new) is larger than .eta_successful times that predicted by a quadratic model of the decrease. The trust-region radius will be increased if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful

.. index:: pair: variable; eta_very_successful
.. _doxid-structtrb__control__type_eta_very_successful:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structtrb__control__type_eta_too_successful:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_too_successful

see eta_successful

.. index:: pair: variable; radius_increase
.. _doxid-structtrb__control__type_radius_increase:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T radius_increase

on very successful iterations, the trust-region radius will be increased the factor .radius_increase, while if the iteration is unsucceful, the radius will be decreased by a factor .radius_reduce but no more than .radius_reduce_max

.. index:: pair: variable; radius_reduce
.. _doxid-structtrb__control__type_radius_reduce:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T radius_reduce

see radius_increase

.. index:: pair: variable; radius_reduce_max
.. _doxid-structtrb__control__type_radius_reduce_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T radius_reduce_max

see radius_increase

.. index:: pair: variable; obj_unbounded
.. _doxid-structtrb__control__type_obj_unbounded:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structtrb__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structtrb__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structtrb__control__type_hessian_available:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; subproblem_direct
.. _doxid-structtrb__control__type_subproblem_direct:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method to find the search direction

.. index:: pair: variable; retrospective_trust_region
.. _doxid-structtrb__control__type_retrospective_trust_region:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool retrospective_trust_region

is a retrospective strategy to be used to update the trust-region radius

.. index:: pair: variable; renormalize_radius
.. _doxid-structtrb__control__type_renormalize_radius:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool renormalize_radius

should the radius be renormalized to account for a change in preconditioner?

.. index:: pair: variable; two_norm_tr
.. _doxid-structtrb__control__type_two_norm_tr:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool two_norm_tr

should an ellipsoidal trust-region be used rather than an infinity norm one?

.. index:: pair: variable; exact_gcp
.. _doxid-structtrb__control__type_exact_gcp:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool exact_gcp

is the exact Cauchy point required rather than an approximation?

.. index:: pair: variable; accurate_bqp
.. _doxid-structtrb__control__type_accurate_bqp:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool accurate_bqp

should the minimizer of the quadratic model within the intersection of the trust-region and feasible box be found (to a prescribed accuracy) rather than a (much) cheaper approximation?

.. index:: pair: variable; space_critical
.. _doxid-structtrb__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structtrb__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structtrb__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; trs_control
.. _doxid-structtrb__control__type_trs_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`trs_control_type<doxid-structtrs__control__type>` trs_control

control parameters for TRS

.. index:: pair: variable; gltr_control
.. _doxid-structtrb__control__type_gltr_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

.. index:: pair: variable; psls_control
.. _doxid-structtrb__control__type_psls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`psls_control_type<doxid-structpsls__control__type>` psls_control

control parameters for PSLS

.. index:: pair: variable; lms_control
.. _doxid-structtrb__control__type_lms_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control

control parameters for LMS

.. index:: pair: variable; lms_control_prec
.. _doxid-structtrb__control__type_lms_control_prec:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control_prec

control parameters for LMS used for preconditioning

.. index:: pair: variable; sha_control
.. _doxid-structtrb__control__type_sha_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sha_control_type<doxid-structsha__control__type>` sha_control

control parameters for SHA

