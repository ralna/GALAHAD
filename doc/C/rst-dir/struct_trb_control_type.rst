.. index:: pair: struct; trb_control_type
.. _doxid-structtrb__control__type:

trb_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_trb.h>
	
	struct trb_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structtrb__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structtrb__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structtrb__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structtrb__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structtrb__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structtrb__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structtrb__control__type_1a31edaef6b722ef2721633484405a649b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structtrb__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structtrb__control__type_1a3fc6359d77a53a63d57ea600b51eac13>`;
		char :ref:`alive_file<doxid-structtrb__control__type_1ac631699a26f321b14dbed37115f3c006>`[31];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`more_toraldo<doxid-structtrb__control__type_1ad8f949bd9ff13ead3970fc52ed44bb7a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`non_monotone<doxid-structtrb__control__type_1a856b2df558071805c217b6d72a1e215b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`model<doxid-structtrb__control__type_1a027a1f1731d22465c926ce57be2364c3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`norm<doxid-structtrb__control__type_1a5b6da5fd1d9c6f86967fa0b4197e3498>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structtrb__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`lbfgs_vectors<doxid-structtrb__control__type_1a90eb3c326cdd5cd8f81f084c4ec5bf30>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_dxg<doxid-structtrb__control__type_1a0d14c8b2992107c3e0f8099cf7f3d04f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`icfs_vectors<doxid-structtrb__control__type_1adb095f545799aab1d69fcdca912d4afd>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_lsize<doxid-structtrb__control__type_1a97a46af6187162b529821f79d1559827>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_rsize<doxid-structtrb__control__type_1a8cd04d404e41a2a09c29eeb2de78cd85>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`advanced_start<doxid-structtrb__control__type_1a7565611061db14e471a4f68e6dabbc17>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structtrb__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_pg_absolute<doxid-structtrb__control__type_1aed4a4de60fddff880a1ef290ab230bde>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_pg_relative<doxid-structtrb__control__type_1a8d441f68e5b4af51d1824e0df1ac4bc8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_s<doxid-structtrb__control__type_1a56612668b00bf042745379f43166cd27>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_radius<doxid-structtrb__control__type_1aa1a65cb31a449551c7819e7e886ca028>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`maximum_radius<doxid-structtrb__control__type_1ad67b5e31569fee1255347e8d1782ce9d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_cg<doxid-structtrb__control__type_1a1360e9ccd514178810a976ac1d072692>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_successful<doxid-structtrb__control__type_1ac0774abb09bb59381960d771cb38b8ef>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_very_successful<doxid-structtrb__control__type_1a5e55cf3fe7846b0f9b23919b0f95469e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_too_successful<doxid-structtrb__control__type_1a6af4c9666b9342fa75b665bfb8cef524>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius_increase<doxid-structtrb__control__type_1a5fdd7428d08e428c376420582cbff66e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius_reduce<doxid-structtrb__control__type_1a5c424056838394ef7a658c5376614567>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius_reduce_max<doxid-structtrb__control__type_1ab47548da92f2f23bae395a0b960d7fba>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_unbounded<doxid-structtrb__control__type_1a7eed67e26bc4e17ca334031b7fd608a6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structtrb__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structtrb__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8>`;
		bool :ref:`hessian_available<doxid-structtrb__control__type_1a0fa05e3076ccb30e3b859c1e4be08981>`;
		bool :ref:`subproblem_direct<doxid-structtrb__control__type_1a8c10db7cf72a4e3e52c9601007f7b1de>`;
		bool :ref:`retrospective_trust_region<doxid-structtrb__control__type_1a7a834a2cc8deb90becaf0245edb7eea9>`;
		bool :ref:`renormalize_radius<doxid-structtrb__control__type_1a761e5ba3ea38a06456f35a1690e77a2e>`;
		bool :ref:`two_norm_tr<doxid-structtrb__control__type_1a5e920badd523b39eb5e854aef7c07b30>`;
		bool :ref:`exact_gcp<doxid-structtrb__control__type_1ad095bacb69fb3fc3ac3de46b9fad96f0>`;
		bool :ref:`accurate_bqp<doxid-structtrb__control__type_1ac97f9f336fb2e903b69c0bbb59ccd240>`;
		bool :ref:`space_critical<doxid-structtrb__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structtrb__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structtrb__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`trs_control_type<doxid-structtrs__control__type>` :ref:`trs_control<doxid-structtrb__control__type_1a0fb493acc030672e71f68fa1ef1727e8>`;
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>` :ref:`gltr_control<doxid-structtrb__control__type_1aa48d482633f3788830b1d8dc85fa91d6>`;
		struct :ref:`psls_control_type<doxid-structpsls__control__type>` :ref:`psls_control<doxid-structtrb__control__type_1a6fe2b1a82e177fbd1a39d9de9652a2c5>`;
		struct :ref:`lms_control_type<doxid-structlms__control__type>` :ref:`lms_control<doxid-structtrb__control__type_1a7149e4807d93e93adf2eb1e3e42c6fb6>`;
		struct :ref:`lms_control_type<doxid-structlms__control__type>` :ref:`lms_control_prec<doxid-structtrb__control__type_1aaad01db9888c4f77c450ff45fac4dfec>`;
		struct :ref:`sha_control_type<doxid-structsha__control__type>` :ref:`sha_control<doxid-structtrb__control__type_1a0e0b1319a0f3da41507bfb343a26ab96>`;
	};
.. _details-structtrb__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structtrb__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structtrb__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structtrb__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structtrb__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structtrb__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structtrb__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structtrb__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structtrb__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

the maximum number of iterations performed

.. index:: pair: variable; alive_unit
.. _doxid-structtrb__control__type_1a3fc6359d77a53a63d57ea600b51eac13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structtrb__control__type_1ac631699a26f321b14dbed37115f3c006:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; more_toraldo
.. _doxid-structtrb__control__type_1ad8f949bd9ff13ead3970fc52ed44bb7a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` more_toraldo

more_toraldo >= 1 gives the number of More'-Toraldo projected searches to be used to improve upon the Cauchy point, anything else is for the standard add-one-at-a-time CG search

.. index:: pair: variable; non_monotone
.. _doxid-structtrb__control__type_1a856b2df558071805c217b6d72a1e215b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` non_monotone

non-monotone <= 0 monotone strategy used, anything else non-monotone strategy with this history length used

.. index:: pair: variable; model
.. _doxid-structtrb__control__type_1a027a1f1731d22465c926ce57be2364c3:

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

* 5 secant second-order (limited-memory BFGS, with .lbfgs_vectors history) (*not yet implemented*)

* 6 secant second-order (limited-memory SR1, with .lbfgs_vectors history) (*not yet implemented*)

.. index:: pair: variable; norm
.. _doxid-structtrb__control__type_1a5b6da5fd1d9c6f86967fa0b4197e3498:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` norm

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
.. _doxid-structtrb__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

specify the semi-bandwidth of the band matrix P if required

.. index:: pair: variable; lbfgs_vectors
.. _doxid-structtrb__control__type_1a90eb3c326cdd5cd8f81f084c4ec5bf30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` lbfgs_vectors

number of vectors used by the L-BFGS matrix P if required

.. index:: pair: variable; max_dxg
.. _doxid-structtrb__control__type_1a0d14c8b2992107c3e0f8099cf7f3d04f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_dxg

number of vectors used by the sparsity-based secant Hessian if required

.. index:: pair: variable; icfs_vectors
.. _doxid-structtrb__control__type_1adb095f545799aab1d69fcdca912d4afd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` icfs_vectors

number of vectors used by the Lin-More' incomplete factorization matrix P if required

.. index:: pair: variable; mi28_lsize
.. _doxid-structtrb__control__type_1a97a46af6187162b529821f79d1559827:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_lsize

the maximum number of fill entries within each column of the incomplete factor L computed by HSL_MI28. In general, increasing .mi28_lsize improve the quality of the preconditioner but increases the time to compute and then apply the preconditioner. Values less than 0 are treated as 0

.. index:: pair: variable; mi28_rsize
.. _doxid-structtrb__control__type_1a8cd04d404e41a2a09c29eeb2de78cd85:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_rsize

the maximum number of entries within each column of the strictly lower triangular matrix $R$ used in the computation of the preconditioner by HSL_MI28. Rank-1 arrays of size .mi28_rsize \* n are allocated internally to hold $R$. Thus the amount of memory used, as well as the amount of work involved in computing the preconditioner, depends on .mi28_rsize. Setting .mi28_rsize > 0 generally leads to a higher quality preconditioner than using .mi28_rsize = 0, and choosing .mi28_rsize >= .mi28_lsize is generally recommended

.. index:: pair: variable; advanced_start
.. _doxid-structtrb__control__type_1a7565611061db14e471a4f68e6dabbc17:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` advanced_start

iterates of a variant on the strategy of Sartenaer SISC 18(6)1990:1788-1803

.. index:: pair: variable; infinity
.. _doxid-structtrb__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_pg_absolute
.. _doxid-structtrb__control__type_1aed4a4de60fddff880a1ef290ab230bde:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_pg_absolute

overall convergence tolerances. The iteration will terminate when the norm of the gradient of the objective function is smaller than MAX( .stop_pg_absolute, .stop_pg_relative \* norm of the initial gradient ) or if the step is less than .stop_s

.. index:: pair: variable; stop_pg_relative
.. _doxid-structtrb__control__type_1a8d441f68e5b4af51d1824e0df1ac4bc8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_pg_relative

see stop_pg_absolute

.. index:: pair: variable; stop_s
.. _doxid-structtrb__control__type_1a56612668b00bf042745379f43166cd27:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_s

see stop_pg_absolute

.. index:: pair: variable; initial_radius
.. _doxid-structtrb__control__type_1aa1a65cb31a449551c7819e7e886ca028:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_radius

initial value for the trust-region radius

.. index:: pair: variable; maximum_radius
.. _doxid-structtrb__control__type_1ad67b5e31569fee1255347e8d1782ce9d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` maximum_radius

maximum permitted trust-region radius

.. index:: pair: variable; stop_rel_cg
.. _doxid-structtrb__control__type_1a1360e9ccd514178810a976ac1d072692:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_cg

required relative reduction in the resuiduals from CG

.. index:: pair: variable; eta_successful
.. _doxid-structtrb__control__type_1ac0774abb09bb59381960d771cb38b8ef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_successful

a potential iterate will only be accepted if the actual decrease f - f(x_new) is larger than .eta_successful times that predicted by a quadratic model of the decrease. The trust-region radius will be increased if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful

.. index:: pair: variable; eta_very_successful
.. _doxid-structtrb__control__type_1a5e55cf3fe7846b0f9b23919b0f95469e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structtrb__control__type_1a6af4c9666b9342fa75b665bfb8cef524:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_too_successful

see eta_successful

.. index:: pair: variable; radius_increase
.. _doxid-structtrb__control__type_1a5fdd7428d08e428c376420582cbff66e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius_increase

on very successful iterations, the trust-region radius will be increased the factor .radius_increase, while if the iteration is unsucceful, the radius will be decreased by a factor .radius_reduce but no more than .radius_reduce_max

.. index:: pair: variable; radius_reduce
.. _doxid-structtrb__control__type_1a5c424056838394ef7a658c5376614567:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius_reduce

see radius_increase

.. index:: pair: variable; radius_reduce_max
.. _doxid-structtrb__control__type_1ab47548da92f2f23bae395a0b960d7fba:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius_reduce_max

see radius_increase

.. index:: pair: variable; obj_unbounded
.. _doxid-structtrb__control__type_1a7eed67e26bc4e17ca334031b7fd608a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structtrb__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structtrb__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structtrb__control__type_1a0fa05e3076ccb30e3b859c1e4be08981:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; subproblem_direct
.. _doxid-structtrb__control__type_1a8c10db7cf72a4e3e52c9601007f7b1de:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method to find the search direction

.. index:: pair: variable; retrospective_trust_region
.. _doxid-structtrb__control__type_1a7a834a2cc8deb90becaf0245edb7eea9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool retrospective_trust_region

is a retrospective strategy to be used to update the trust-region radius

.. index:: pair: variable; renormalize_radius
.. _doxid-structtrb__control__type_1a761e5ba3ea38a06456f35a1690e77a2e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool renormalize_radius

should the radius be renormalized to account for a change in preconditioner?

.. index:: pair: variable; two_norm_tr
.. _doxid-structtrb__control__type_1a5e920badd523b39eb5e854aef7c07b30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool two_norm_tr

should an ellipsoidal trust-region be used rather than an infinity norm one?

.. index:: pair: variable; exact_gcp
.. _doxid-structtrb__control__type_1ad095bacb69fb3fc3ac3de46b9fad96f0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool exact_gcp

is the exact Cauchy point required rather than an approximation?

.. index:: pair: variable; accurate_bqp
.. _doxid-structtrb__control__type_1ac97f9f336fb2e903b69c0bbb59ccd240:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool accurate_bqp

should the minimizer of the quadratic model within the intersection of the trust-region and feasible box be found (to a prescribed accuracy) rather than a (much) cheaper approximation?

.. index:: pair: variable; space_critical
.. _doxid-structtrb__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structtrb__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structtrb__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; trs_control
.. _doxid-structtrb__control__type_1a0fb493acc030672e71f68fa1ef1727e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trs_control_type<doxid-structtrs__control__type>` trs_control

control parameters for TRS

.. index:: pair: variable; gltr_control
.. _doxid-structtrb__control__type_1aa48d482633f3788830b1d8dc85fa91d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

.. index:: pair: variable; psls_control
.. _doxid-structtrb__control__type_1a6fe2b1a82e177fbd1a39d9de9652a2c5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_control_type<doxid-structpsls__control__type>` psls_control

control parameters for PSLS

.. index:: pair: variable; lms_control
.. _doxid-structtrb__control__type_1a7149e4807d93e93adf2eb1e3e42c6fb6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control

control parameters for LMS

.. index:: pair: variable; lms_control_prec
.. _doxid-structtrb__control__type_1aaad01db9888c4f77c450ff45fac4dfec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control_prec

control parameters for LMS used for preconditioning

.. index:: pair: variable; sha_control
.. _doxid-structtrb__control__type_1a0e0b1319a0f3da41507bfb343a26ab96:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sha_control_type<doxid-structsha__control__type>` sha_control

control parameters for SHA

