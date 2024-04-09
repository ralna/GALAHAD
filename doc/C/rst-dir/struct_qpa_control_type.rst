.. index:: pair: struct; qpa_control_type
.. _doxid-structqpa__control__type:

qpa_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_qpa.h>
	
	struct qpa_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structqpa__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structqpa__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structqpa__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structqpa__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structqpa__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structqpa__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structqpa__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factor<doxid-structqpa__control__type_1a42eac05293c31fef9b90c92698928d7d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col<doxid-structqpa__control__type_1abca2db33b9520095e98790d45a1be93f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_sc<doxid-structqpa__control__type_1a58895c9f8fe04a04578073223b30af2a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`indmin<doxid-structqpa__control__type_1a5031bbc31f94e4cba6a540a3182b6d80>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`valmin<doxid-structqpa__control__type_1a0e142fa8dc9c363c3c2993b6129b0955>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itref_max<doxid-structqpa__control__type_1a903ba4ef0869186a65d4c32459a6a0ed>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`infeas_check_interval<doxid-structqpa__control__type_1ab4584330c0becd648f624a63da263422>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structqpa__control__type_1a7a1029142a22f3e2a1963c3428276849>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`precon<doxid-structqpa__control__type_1ab22aef71d1fbb41e515107086ef88b54>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nsemib<doxid-structqpa__control__type_1a2215bbe54edcdc418de8b79a995c9b75>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`full_max_fill<doxid-structqpa__control__type_1a4eaa8ad7ef1b5f8ef3d66f481ef2a4ec>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`deletion_strategy<doxid-structqpa__control__type_1a05a547ad2b2708fdef102264e4200d38>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`restore_problem<doxid-structqpa__control__type_1a19f10912888ac7ebd153fb21baaeaefa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`monitor_residuals<doxid-structqpa__control__type_1a086f355fe16121b32849f7f54a6d821d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cold_start<doxid-structqpa__control__type_1ad5e3138a19f7400e9d5c8105fa724831>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structqpa__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structqpa__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`feas_tol<doxid-structqpa__control__type_1aa636051ebb14fdebe323a3843e7e24e3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_unbounded<doxid-structqpa__control__type_1a7eed67e26bc4e17ca334031b7fd608a6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`increase_rho_g_factor<doxid-structqpa__control__type_1aacaa758f766934307f78d98b04064d50>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infeas_g_improved_by_factor<doxid-structqpa__control__type_1a2441f4b793197c9f7ff879131baf2655>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`increase_rho_b_factor<doxid-structqpa__control__type_1aab5a0d54763ee4bca9465d05d2ccb539>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infeas_b_improved_by_factor<doxid-structqpa__control__type_1a2a0c4d140899491cdf36bebdb25cf39f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol<doxid-structqpa__control__type_1a133347eb5f45a24a77b63b4afd4212e8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol_for_dependencies<doxid-structqpa__control__type_1a14e253b53c59b0850d7d3b5245d89df9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_pivot<doxid-structqpa__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`inner_stop_relative<doxid-structqpa__control__type_1a7aedce06c5903f675a7c1496f42ff834>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`inner_stop_absolute<doxid-structqpa__control__type_1a80ae9dce0bca96a4691b0a222e3257b5>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier_tol<doxid-structqpa__control__type_1a6f12f6f5f94ae38d4a4152b1150dd9af>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structqpa__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structqpa__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8>`;
		bool :ref:`treat_zero_bounds_as_general<doxid-structqpa__control__type_1a59b23877e9c8cb49f1f6261e24660295>`;
		bool :ref:`solve_qp<doxid-structqpa__control__type_1abbf84109b56423a0eb15b3c096349eff>`;
		bool :ref:`solve_within_bounds<doxid-structqpa__control__type_1a62862be91f70fd8284c0faff374bc022>`;
		bool :ref:`randomize<doxid-structqpa__control__type_1a7a24e4455c51460f9504bab5a850ea74>`;
		bool :ref:`array_syntax_worse_than_do_loop<doxid-structqpa__control__type_1a67975e9960ae3d4d79bf18b240b9f614>`;
		bool :ref:`space_critical<doxid-structqpa__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structqpa__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		bool :ref:`generate_sif_file<doxid-structqpa__control__type_1aa75b3a16d146c0d7ad57bf9817033843>`;
		char :ref:`symmetric_linear_solver<doxid-structqpa__control__type_1af297ace351b9307640715643cde57384>`[31];
		char :ref:`sif_file_name<doxid-structqpa__control__type_1aaa95e830b709da79d9790471bab54193>`[31];
		char :ref:`prefix<doxid-structqpa__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		bool :ref:`each_interval<doxid-structqpa__control__type_1a1c67e9287c98ef959b05e041803f16a9>`;
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structqpa__control__type_1a31b308b91955ee385daacc3de00f161b>`;
	};
.. _details-structqpa__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structqpa__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structqpa__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structqpa__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structqpa__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structqpa__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structqpa__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structqpa__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; factor
.. _doxid-structqpa__control__type_1a42eac05293c31fef9b90c92698928d7d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factor

the factorization to be used. Possible values are 0 automatic 1 Schur-complement factorization 2 augmented-system factorization

.. index:: pair: variable; max_col
.. _doxid-structqpa__control__type_1abca2db33b9520095e98790d45a1be93f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col

the maximum number of nonzeros in a column of A which is permitted with the Schur-complement factorization

.. index:: pair: variable; max_sc
.. _doxid-structqpa__control__type_1a58895c9f8fe04a04578073223b30af2a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_sc

the maximum permitted size of the Schur complement before a refactorization is performed

.. index:: pair: variable; indmin
.. _doxid-structqpa__control__type_1a5031bbc31f94e4cba6a540a3182b6d80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` indmin

an initial guess as to the integer workspace required by SLS (OBSOLETE)

.. index:: pair: variable; valmin
.. _doxid-structqpa__control__type_1a0e142fa8dc9c363c3c2993b6129b0955:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` valmin

an initial guess as to the real workspace required by SLS (OBSOLETE)

.. index:: pair: variable; itref_max
.. _doxid-structqpa__control__type_1a903ba4ef0869186a65d4c32459a6a0ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itref_max

the maximum number of iterative refinements allowed (OBSOLETE)

.. index:: pair: variable; infeas_check_interval
.. _doxid-structqpa__control__type_1ab4584330c0becd648f624a63da263422:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` infeas_check_interval

the infeasibility will be checked for improvement every infeas_check_interval iterations (see infeas_g_improved_by_factor and infeas_b_improved_by_factor below)

.. index:: pair: variable; cg_maxit
.. _doxid-structqpa__control__type_1a7a1029142a22f3e2a1963c3428276849:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

the maximum number of CG iterations allowed. If cg_maxit < 0, this number will be reset to the dimension of the system + 1

.. index:: pair: variable; precon
.. _doxid-structqpa__control__type_1ab22aef71d1fbb41e515107086ef88b54:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` precon

the preconditioner to be used for the CG is defined by precon. Possible values are 0 automatic 1 no preconditioner, i.e, the identity within full factorization 2 full factorization 3 band within full factorization 4 diagonal using the barrier terms within full factorization

.. index:: pair: variable; nsemib
.. _doxid-structqpa__control__type_1a2215bbe54edcdc418de8b79a995c9b75:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nsemib

the semi-bandwidth of a band preconditioner, if appropriate

.. index:: pair: variable; full_max_fill
.. _doxid-structqpa__control__type_1a4eaa8ad7ef1b5f8ef3d66f481ef2a4ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` full_max_fill

if the ratio of the number of nonzeros in the factors of the reference matrix to the number of nonzeros in the matrix itself exceeds full_max_fill, and the preconditioner is being selected automatically (precon = 0), a banded approximation will be used instead

.. index:: pair: variable; deletion_strategy
.. _doxid-structqpa__control__type_1a05a547ad2b2708fdef102264e4200d38:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` deletion_strategy

the constraint deletion strategy to be used. Possible values are:

0 most violated of all 1 LIFO (last in, first out) k LIFO(k) most violated of the last k in LIFO

.. index:: pair: variable; restore_problem
.. _doxid-structqpa__control__type_1a19f10912888ac7ebd153fb21baaeaefa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are 0 nothing restored 1 scalar and vector parameters 2 all parameters

.. index:: pair: variable; monitor_residuals
.. _doxid-structqpa__control__type_1a086f355fe16121b32849f7f54a6d821d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` monitor_residuals

the frequency at which residuals will be monitored

.. index:: pair: variable; cold_start
.. _doxid-structqpa__control__type_1ad5e3138a19f7400e9d5c8105fa724831:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cold_start

indicates whether a cold or warm start should be made. Possible values are

0 warm start - the values set in C_stat and B_stat indicate which constraints will be included in the initial working set. 1 cold start from the value set in X; constraints active at X will determine the initial working set. 2 cold start with no active constraints 3 cold start with only equality constraints active 4 cold start with as many active constraints as possible

.. index:: pair: variable; sif_file_device
.. _doxid-structqpa__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structqpa__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; feas_tol
.. _doxid-structqpa__control__type_1aa636051ebb14fdebe323a3843e7e24e3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` feas_tol

any constraint violated by less than feas_tol will be considered to be satisfied

.. index:: pair: variable; obj_unbounded
.. _doxid-structqpa__control__type_1a7eed67e26bc4e17ca334031b7fd608a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_unbounded

if the objective function value is smaller than obj_unbounded, it will be flagged as unbounded from below.

.. index:: pair: variable; increase_rho_g_factor
.. _doxid-structqpa__control__type_1aacaa758f766934307f78d98b04064d50:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` increase_rho_g_factor

if the problem is currently infeasible and solve_qp (see below) is .TRUE. the current penalty parameter for the general constraints will be increased by increase_rho_g_factor when needed

.. index:: pair: variable; infeas_g_improved_by_factor
.. _doxid-structqpa__control__type_1a2441f4b793197c9f7ff879131baf2655:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infeas_g_improved_by_factor

if the infeasibility of the general constraints has not dropped by a fac of infeas_g_improved_by_factor over the previous infeas_check_interval iterations, the current corresponding penalty parameter will be increase

.. index:: pair: variable; increase_rho_b_factor
.. _doxid-structqpa__control__type_1aab5a0d54763ee4bca9465d05d2ccb539:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` increase_rho_b_factor

if the problem is currently infeasible and solve_qp or solve_within_boun (see below) is .TRUE., the current penalty parameter for the simple boun constraints will be increased by increase_rho_b_factor when needed

.. index:: pair: variable; infeas_b_improved_by_factor
.. _doxid-structqpa__control__type_1a2a0c4d140899491cdf36bebdb25cf39f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infeas_b_improved_by_factor

if the infeasibility of the simple bounds has not dropped by a factor of infeas_b_improved_by_factor over the previous infeas_check_interval iterations, the current corresponding penalty parameter will be increase

.. index:: pair: variable; pivot_tol
.. _doxid-structqpa__control__type_1a133347eb5f45a24a77b63b4afd4212e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol

the threshold pivot used by the matrix factorization. See the documentation for SLS for details (OBSOLE

.. index:: pair: variable; pivot_tol_for_dependencies
.. _doxid-structqpa__control__type_1a14e253b53c59b0850d7d3b5245d89df9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol_for_dependencies

the threshold pivot used by the matrix factorization when attempting to detect linearly dependent constraints.

.. index:: pair: variable; zero_pivot
.. _doxid-structqpa__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_pivot

any pivots smaller than zero_pivot in absolute value will be regarded to zero when attempting to detect linearly dependent constraints (OBSOLE

.. index:: pair: variable; inner_stop_relative
.. _doxid-structqpa__control__type_1a7aedce06c5903f675a7c1496f42ff834:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` inner_stop_relative

the search direction is considered as an acceptable approximation to the minimizer of the model if the gradient of the model in the preconditioning(inverse) norm is less than max( inner_stop_relative \* initial preconditioning(inverse) gradient norm, inner_stop_absolute )

.. index:: pair: variable; inner_stop_absolute
.. _doxid-structqpa__control__type_1a80ae9dce0bca96a4691b0a222e3257b5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` inner_stop_absolute

see inner_stop_relative

.. index:: pair: variable; multiplier_tol
.. _doxid-structqpa__control__type_1a6f12f6f5f94ae38d4a4152b1150dd9af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier_tol

any dual variable or Lagrange multiplier which is less than multiplier_t outside its optimal interval will be regarded as being acceptable when checking for optimality

.. index:: pair: variable; cpu_time_limit
.. _doxid-structqpa__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structqpa__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structqpa__control__type_1a59b23877e9c8cb49f1f6261e24660295:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; solve_qp
.. _doxid-structqpa__control__type_1abbf84109b56423a0eb15b3c096349eff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool solve_qp

if solve_qp is .TRUE., the value of prob.rho_g and prob.rho_b will be increased as many times as are needed to ensure that the output solution is feasible, and thus aims to solve the quadratic program (2)-(4)

.. index:: pair: variable; solve_within_bounds
.. _doxid-structqpa__control__type_1a62862be91f70fd8284c0faff374bc022:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool solve_within_bounds

if solve_within_bounds is .TRUE., the value of prob.rho_b will be increased as many times as are needed to ensure that the output solution is feasible with respect to the simple bounds, and thus aims to solve the bound-constrained quadratic program (4)-(5)

.. index:: pair: variable; randomize
.. _doxid-structqpa__control__type_1a7a24e4455c51460f9504bab5a850ea74:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool randomize

if randomize is .TRUE., the constraint bounds will be perturbed by small random quantities during the first stage of the solution process. Any randomization will ultimately be removed. Randomization helps when solving degenerate problems

.. index:: pair: variable; array_syntax_worse_than_do_loop
.. _doxid-structqpa__control__type_1a67975e9960ae3d4d79bf18b240b9f614:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool array_syntax_worse_than_do_loop

if .array_syntax_worse_than_do_loop is true, f77-style do loops will be used rather than f90-style array syntax for vector operations (OBSOLETE)

.. index:: pair: variable; space_critical
.. _doxid-structqpa__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structqpa__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structqpa__control__type_1aa75b3a16d146c0d7ad57bf9817033843:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structqpa__control__type_1af297ace351b9307640715643cde57384:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric-indefinite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', and 'sytr', although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; sif_file_name
.. _doxid-structqpa__control__type_1aaa95e830b709da79d9790471bab54193:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

definite linear equation solver

name of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structqpa__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; each_interval
.. _doxid-structqpa__control__type_1a1c67e9287c98ef959b05e041803f16a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool each_interval

component specifically for parametric problems (not used at present)

.. index:: pair: variable; sls_control
.. _doxid-structqpa__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

