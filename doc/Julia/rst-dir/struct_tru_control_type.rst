.. index:: pair: struct; tru_control_type
.. _doxid-structtru__control__type:

tru_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct tru_control_type{T,INT}
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
          non_monotone::INT
          model::INT
          norm::INT
          semi_bandwidth::INT
          lbfgs_vectors::INT
          max_dxg::INT
          icfs_vectors::INT
          mi28_lsize::INT
          mi28_rsize::INT
          stop_g_absolute::T
          stop_g_relative::T
          stop_s::T
          advanced_start::INT
          initial_radius::T
          maximum_radius::T
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
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}
          trs_control::trs_control_type{T,INT}
          gltr_control::gltr_control_type{T,INT}
          dps_control::dps_control_type{T,INT}
          psls_control::psls_control_type{T,INT}
          lms_control::lms_control_type{INT}
          lms_control_prec::lms_control_type{INT}
          sec_control::sec_control_type{T,INT}
          sha_control::sha_control_type{INT}

.. _details-structtru__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structtru__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structtru__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structtru__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structtru__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structtru__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structtru__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structtru__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structtru__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

the maximum number of iterations allowed

.. index:: pair: variable; alive_unit
.. _doxid-structtru__control__type_1a3fc6359d77a53a63d57ea600b51eac13:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structtru__control__type_1ac631699a26f321b14dbed37115f3c006:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; non_monotone
.. _doxid-structtru__control__type_1a856b2df558071805c217b6d72a1e215b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT non_monotone

the descent strategy used.

Possible values are

* <= 0 a monotone strategy is used.

* anything else, a non-monotone strategy with history length .non_monotine is used.

.. index:: pair: variable; model
.. _doxid-structtru__control__type_1a027a1f1731d22465c926ce57be2364c3:

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

* 5 secant second-order (limited-memory BFGS, with .lbfgs_vectors history)

* 6 secant second-order (limited-memory SR1, with .lbfgs_vectors history)

.. index:: pair: variable; norm
.. _doxid-structtru__control__type_1a5b6da5fd1d9c6f86967fa0b4197e3498:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT norm

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
.. _doxid-structtru__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth

specify the semi-bandwidth of the band matrix $P$ if required

.. index:: pair: variable; lbfgs_vectors
.. _doxid-structtru__control__type_1a90eb3c326cdd5cd8f81f084c4ec5bf30:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lbfgs_vectors

number of vectors used by the L-BFGS matrix $P$ if required

.. index:: pair: variable; max_dxg
.. _doxid-structtru__control__type_1a0d14c8b2992107c3e0f8099cf7f3d04f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_dxg

number of vectors used by the sparsity-based secant Hessian if required

.. index:: pair: variable; icfs_vectors
.. _doxid-structtru__control__type_1adb095f545799aab1d69fcdca912d4afd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT icfs_vectors

number of vectors used by the Lin-More' incomplete factorization matrix $P$ if required

.. index:: pair: variable; mi28_lsize
.. _doxid-structtru__control__type_1a97a46af6187162b529821f79d1559827:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mi28_lsize

the maximum number of fill entries within each column of the incomplete factor L computed by HSL_MI28. In general, increasing .mi28_lsize improve the quality of the preconditioner but increases the time to compute and then apply the preconditioner. Values less than 0 are treated as 0

.. index:: pair: variable; mi28_rsize
.. _doxid-structtru__control__type_1a8cd04d404e41a2a09c29eeb2de78cd85:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mi28_rsize

the maximum number of entries within each column of the strictly lower triangular matrix $R$ used in the computation of the preconditioner by HSL_MI28. Rank-1 arrays of size .mi28_rsize \* n are allocated internally to hold $R$. Thus the amount of memory used, as well as the amount of work involved in computing the preconditioner, depends on .mi28_rsize. Setting .mi28_rsize > 0 generally leads to a higher quality preconditioner than using .mi28_rsize = 0, and choosing .mi28_rsize >= .mi28_lsize is generally recommended

.. index:: pair: variable; stop_g_absolute
.. _doxid-structtru__control__type_1a6182fed3e6c11b9aa39e1460c1def7f8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_g_absolute

overall convergence tolerances. The iteration will terminate when the norm of the gradient of the objective function is smaller than MAX( .stop_g_absolute, .stop_g_relative \* norm of the initial gradient ) or if the step is less than .stop_s

.. index:: pair: variable; stop_g_relative
.. _doxid-structtru__control__type_1aeb89f1dc942cea0814ee1e8d645467d3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_g_relative

see stop_g_absolute

.. index:: pair: variable; stop_s
.. _doxid-structtru__control__type_1a56612668b00bf042745379f43166cd27:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_s

see stop_g_absolute

.. index:: pair: variable; advanced_start
.. _doxid-structtru__control__type_1a7565611061db14e471a4f68e6dabbc17:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT advanced_start

try to pick a good initial trust-region radius using .advanced_start iterates of a variant on the strategy of Sartenaer SISC 18(6) 1990:1788-1803

.. index:: pair: variable; initial_radius
.. _doxid-structtru__control__type_1aa1a65cb31a449551c7819e7e886ca028:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T initial_radius

initial value for the trust-region radius

.. index:: pair: variable; maximum_radius
.. _doxid-structtru__control__type_1ad67b5e31569fee1255347e8d1782ce9d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T maximum_radius

maximum permitted trust-region radius

.. index:: pair: variable; eta_successful
.. _doxid-structtru__control__type_1ac0774abb09bb59381960d771cb38b8ef:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_successful

a potential iterate will only be accepted if the actual decrease $f - f(x_{new})$ is larger than .eta_successful times that predicted by a quadratic model of the decrease. The trust-region radius will be increased if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful

.. index:: pair: variable; eta_very_successful
.. _doxid-structtru__control__type_1a5e55cf3fe7846b0f9b23919b0f95469e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structtru__control__type_1a6af4c9666b9342fa75b665bfb8cef524:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eta_too_successful

see eta_successful

.. index:: pair: variable; radius_increase
.. _doxid-structtru__control__type_1a5fdd7428d08e428c376420582cbff66e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T radius_increase

on very successful iterations, the trust-region radius will be increased by the factor .radius_increase, while if the iteration is unsucceful, the radius will be decreased by a factor .radius_reduce but no more than .radius_reduce_max

.. index:: pair: variable; radius_reduce
.. _doxid-structtru__control__type_1a5c424056838394ef7a658c5376614567:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T radius_reduce

see radius_increase;

.. index:: pair: variable; radius_reduce_max
.. _doxid-structtru__control__type_1ab47548da92f2f23bae395a0b960d7fba:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T radius_reduce_max

see radius_increase;

.. index:: pair: variable; obj_unbounded
.. _doxid-structtru__control__type_1a7eed67e26bc4e17ca334031b7fd608a6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structtru__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structtru__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structtru__control__type_1a0fa05e3076ccb30e3b859c1e4be08981:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; subproblem_direct
.. _doxid-structtru__control__type_1a8c10db7cf72a4e3e52c9601007f7b1de:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method to find the search direction

.. index:: pair: variable; retrospective_trust_region
.. _doxid-structtru__control__type_1a7a834a2cc8deb90becaf0245edb7eea9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool retrospective_trust_region

is a retrospective strategy to be used to update the trust-region radius?

.. index:: pair: variable; renormalize_radius
.. _doxid-structtru__control__type_1a761e5ba3ea38a06456f35a1690e77a2e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool renormalize_radius

should the radius be renormalized to account for a change in preconditioner?

.. index:: pair: variable; space_critical
.. _doxid-structtru__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structtru__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structtru__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; trs_control
.. _doxid-structtru__control__type_1a0fb493acc030672e71f68fa1ef1727e8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`trs_control_type<doxid-structtrs__control__type>` trs_control

control parameters for TRS

.. index:: pair: variable; gltr_control
.. _doxid-structtru__control__type_1aa48d482633f3788830b1d8dc85fa91d6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

.. index:: pair: variable; dps_control
.. _doxid-structtru__control__type_1a400a915be09fbe2f8114fc9f7f9dddf1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`dps_control_type<doxid-structdps__control__type>` dps_control

control parameters for DPS

.. index:: pair: variable; psls_control
.. _doxid-structtru__control__type_1a6fe2b1a82e177fbd1a39d9de9652a2c5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`psls_control_type<doxid-structpsls__control__type>` psls_control

control parameters for PSLS

.. index:: pair: variable; lms_control
.. _doxid-structtru__control__type_1a7149e4807d93e93adf2eb1e3e42c6fb6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control

control parameters for LMS

.. index:: pair: variable; lms_control_prec
.. _doxid-structtru__control__type_1aaad01db9888c4f77c450ff45fac4dfec:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lms_control_type<doxid-structlms__control__type>` lms_control_prec

control parameters for LMS used for preconditioning

.. index:: pair: variable; sec_control
.. _doxid-structtru__control__type_1a3b18636f4c3c9a62ddcaf9dfb5b56da6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sec_control_type<doxid-structsec__control__type>` sec_control

control parameters for SEC

.. index:: pair: variable; sha_control
.. _doxid-structtru__control__type_1a0e0b1319a0f3da41507bfb343a26ab96:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sha_control_type<doxid-structsha__control__type>` sha_control

control parameters for SHA

