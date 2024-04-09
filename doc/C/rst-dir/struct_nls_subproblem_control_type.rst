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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structnls__subproblem__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structnls__subproblem__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structnls__subproblem__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structnls__subproblem__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structnls__subproblem__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structnls__subproblem__control__type_1a31edaef6b722ef2721633484405a649b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structnls__subproblem__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structnls__subproblem__control__type_1a3fc6359d77a53a63d57ea600b51eac13>`;
		char :ref:`alive_file<doxid-structnls__subproblem__control__type_1ac631699a26f321b14dbed37115f3c006>`[31];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`jacobian_available<doxid-structnls__subproblem__control__type_1a331884a4968f11decfca0a687fc59dde>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`hessian_available<doxid-structnls__subproblem__control__type_1ae665ac50d8c985bf502ba2b90363826a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`model<doxid-structnls__subproblem__control__type_1a027a1f1731d22465c926ce57be2364c3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`norm<doxid-structnls__subproblem__control__type_1a5b6da5fd1d9c6f86967fa0b4197e3498>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`non_monotone<doxid-structnls__subproblem__control__type_1a856b2df558071805c217b6d72a1e215b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`weight_update_strategy<doxid-structnls__subproblem__control__type_1a91395480b85edaac83529fc7f1605289>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_c_absolute<doxid-structnls__subproblem__control__type_1add1b86a6f37634dcbf1e5f55807eac2b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_c_relative<doxid-structnls__subproblem__control__type_1a8e60a718c9c3b3bf9daf2e2be3c7dd68>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_g_absolute<doxid-structnls__subproblem__control__type_1a6182fed3e6c11b9aa39e1460c1def7f8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_g_relative<doxid-structnls__subproblem__control__type_1aeb89f1dc942cea0814ee1e8d645467d3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_s<doxid-structnls__subproblem__control__type_1a56612668b00bf042745379f43166cd27>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`power<doxid-structnls__subproblem__control__type_1aa8611d2e53cef87afe803f0e486f3c98>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_weight<doxid-structnls__subproblem__control__type_1aa8b73dcc65ae7dc7b3331980f77c5fb4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`minimum_weight<doxid-structnls__subproblem__control__type_1a044b125f7d2b5409dde4253030798367>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_inner_weight<doxid-structnls__subproblem__control__type_1ab32efd2263e7fbebfe87846bc977f1ab>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_successful<doxid-structnls__subproblem__control__type_1ac0774abb09bb59381960d771cb38b8ef>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_very_successful<doxid-structnls__subproblem__control__type_1a5e55cf3fe7846b0f9b23919b0f95469e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`eta_too_successful<doxid-structnls__subproblem__control__type_1a6af4c9666b9342fa75b665bfb8cef524>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_decrease_min<doxid-structnls__subproblem__control__type_1a481b6aeef2f3d9a0078665db1e512c85>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_decrease<doxid-structnls__subproblem__control__type_1a46e8590e1c6ebb8c2a673d854762424d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_increase<doxid-structnls__subproblem__control__type_1a92d59bc5d9b2899fbb318ea033e85540>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight_increase_max<doxid-structnls__subproblem__control__type_1a43230e4cfd494bbdcb897074b0b9768b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce_gap<doxid-structnls__subproblem__control__type_1a462c383f8c4b96b23d585f292bd5e0e2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`tiny_gap<doxid-structnls__subproblem__control__type_1a28c746bfd481575683553dc28abe3e30>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`large_root<doxid-structnls__subproblem__control__type_1a7abb4c844c9fddec04f47c856f4a2383>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`switch_to_newton<doxid-structnls__subproblem__control__type_1a3e388718674f5243006272b3911fae22>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structnls__subproblem__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structnls__subproblem__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8>`;
		bool :ref:`subproblem_direct<doxid-structnls__subproblem__control__type_1a8c10db7cf72a4e3e52c9601007f7b1de>`;
		bool :ref:`renormalize_weight<doxid-structnls__subproblem__control__type_1aa451859552980839fa9fe17df018c042>`;
		bool :ref:`magic_step<doxid-structnls__subproblem__control__type_1adb993f5277ee10213227e86ba94fd934>`;
		bool :ref:`print_obj<doxid-structnls__subproblem__control__type_1a95ba685ae01db06b2e9a16412097612a>`;
		bool :ref:`space_critical<doxid-structnls__subproblem__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structnls__subproblem__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structnls__subproblem__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`rqs_control_type<doxid-structrqs__control__type>` :ref:`rqs_control<doxid-structnls__subproblem__control__type_1a7339f172b9757ed89a7595a29c4bca2b>`;
		struct :ref:`glrt_control_type<doxid-structglrt__control__type>` :ref:`glrt_control<doxid-structnls__subproblem__control__type_1ac872f3f7f159cedfac4d6898d51842a3>`;
		struct :ref:`psls_control_type<doxid-structpsls__control__type>` :ref:`psls_control<doxid-structnls__subproblem__control__type_1a6fe2b1a82e177fbd1a39d9de9652a2c5>`;
		struct :ref:`bsc_control_type<doxid-structbsc__control__type>` :ref:`bsc_control<doxid-structnls__subproblem__control__type_1a899f35f15eea56c23fe8fb22857e01f7>`;
		struct :ref:`roots_control_type<doxid-structroots__control__type>` :ref:`roots_control<doxid-structnls__subproblem__control__type_1a08df6708e7b5364ff3e8fbde29f29014>`;
	};
.. _details-structnls__subproblem__control__type:

detailed documentation
----------------------

subproblem_control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; error
.. _doxid-structnls__subproblem__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structnls__subproblem__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structnls__subproblem__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required.

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structnls__subproblem__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structnls__subproblem__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structnls__subproblem__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structnls__subproblem__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

the maximum number of iterations performed

.. index:: pair: variable; alive_unit
.. _doxid-structnls__subproblem__control__type_1a3fc6359d77a53a63d57ea600b51eac13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structnls__subproblem__control__type_1ac631699a26f321b14dbed37115f3c006:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; jacobian_available
.. _doxid-structnls__subproblem__control__type_1a331884a4968f11decfca0a687fc59dde:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` jacobian_available

is the Jacobian matrix of first derivatives available ($\geq$ 2), is access only via matrix-vector products (=1) or is it not available ($\leq$ 0) ?

.. index:: pair: variable; hessian_available
.. _doxid-structnls__subproblem__control__type_1ae665ac50d8c985bf502ba2b90363826a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` hessian_available

is the Hessian matrix of second derivatives available ($\geq$ 2), is access only via matrix-vector products (=1) or is it not available ($\leq$ 0) ?

.. index:: pair: variable; model
.. _doxid-structnls__subproblem__control__type_1a027a1f1731d22465c926ce57be2364c3:

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
.. _doxid-structnls__subproblem__control__type_1a5b6da5fd1d9c6f86967fa0b4197e3498:

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
.. _doxid-structnls__subproblem__control__type_1a856b2df558071805c217b6d72a1e215b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` non_monotone

non-monotone $\leq$ 0 monotone strategy used, anything else non-monotone strategy with this history length used

.. index:: pair: variable; weight_update_strategy
.. _doxid-structnls__subproblem__control__type_1a91395480b85edaac83529fc7f1605289:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` weight_update_strategy

define the weight-update strategy: 1 (basic), 2 (reset to zero when very successful), 3 (imitate TR), 4 (increase lower bound), 5 (GPT)

.. index:: pair: variable; stop_c_absolute
.. _doxid-structnls__subproblem__control__type_1add1b86a6f37634dcbf1e5f55807eac2b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_c_absolute

overall convergence tolerances. The iteration will terminate when $||c(x)||_2 \leq$ MAX( .stop_c_absolute, .stop_c_relative $* \|c(x_{\mbox{initial}})\|_2$, or when the norm of the gradient, $g = J^T(x) c(x) / \|c(x)\|_2$, of \|\|c\|\|_2, satisfies $\|g\|_2 \leq$ MAX( .stop_g_absolute, .stop_g_relative $* \|g_{\mbox{initial}}\|_2$, or if the step is less than .stop_s

.. index:: pair: variable; stop_c_relative
.. _doxid-structnls__subproblem__control__type_1a8e60a718c9c3b3bf9daf2e2be3c7dd68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_c_relative

see stop_c_absolute

.. index:: pair: variable; stop_g_absolute
.. _doxid-structnls__subproblem__control__type_1a6182fed3e6c11b9aa39e1460c1def7f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_g_absolute

see stop_c_absolute

.. index:: pair: variable; stop_g_relative
.. _doxid-structnls__subproblem__control__type_1aeb89f1dc942cea0814ee1e8d645467d3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_g_relative

see stop_c_absolute

.. index:: pair: variable; stop_s
.. _doxid-structnls__subproblem__control__type_1a56612668b00bf042745379f43166cd27:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_s

see stop_c_absolute

.. index:: pair: variable; power
.. _doxid-structnls__subproblem__control__type_1aa8611d2e53cef87afe803f0e486f3c98:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` power

the regularization power (<2 => chosen according to the model)

.. index:: pair: variable; initial_weight
.. _doxid-structnls__subproblem__control__type_1aa8b73dcc65ae7dc7b3331980f77c5fb4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_weight

initial value for the regularization weight (-ve => $1/\|g_0\|)$)

.. index:: pair: variable; minimum_weight
.. _doxid-structnls__subproblem__control__type_1a044b125f7d2b5409dde4253030798367:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` minimum_weight

minimum permitted regularization weight

.. index:: pair: variable; initial_inner_weight
.. _doxid-structnls__subproblem__control__type_1ab32efd2263e7fbebfe87846bc977f1ab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_inner_weight

initial value for the inner regularization weight for tensor GN (-ve => 0)

.. index:: pair: variable; eta_successful
.. _doxid-structnls__subproblem__control__type_1ac0774abb09bb59381960d771cb38b8ef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_successful

a potential iterate will only be accepted if the actual decrease f - f(x_new) is larger than .eta_successful times that predicted by a quadratic model of the decrease. The regularization weight will be decreaed if this relative decrease is greater than .eta_very_successful but smaller than .eta_too_successful

.. index:: pair: variable; eta_very_successful
.. _doxid-structnls__subproblem__control__type_1a5e55cf3fe7846b0f9b23919b0f95469e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_very_successful

see eta_successful

.. index:: pair: variable; eta_too_successful
.. _doxid-structnls__subproblem__control__type_1a6af4c9666b9342fa75b665bfb8cef524:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` eta_too_successful

see eta_successful

.. index:: pair: variable; weight_decrease_min
.. _doxid-structnls__subproblem__control__type_1a481b6aeef2f3d9a0078665db1e512c85:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_decrease_min

on very successful iterations, the regularization weight will be reduced by the factor .weight_decrease but no more than .weight_decrease_min while if the iteration is unsucceful, the weight will be increased by a factor .weight_increase but no more than .weight_increase_max (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli and Toint, 2011)

.. index:: pair: variable; weight_decrease
.. _doxid-structnls__subproblem__control__type_1a46e8590e1c6ebb8c2a673d854762424d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_decrease

see weight_decrease_min

.. index:: pair: variable; weight_increase
.. _doxid-structnls__subproblem__control__type_1a92d59bc5d9b2899fbb318ea033e85540:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_increase

see weight_decrease_min

.. index:: pair: variable; weight_increase_max
.. _doxid-structnls__subproblem__control__type_1a43230e4cfd494bbdcb897074b0b9768b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight_increase_max

see weight_decrease_min

.. index:: pair: variable; reduce_gap
.. _doxid-structnls__subproblem__control__type_1a462c383f8c4b96b23d585f292bd5e0e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce_gap

expert parameters as suggested in Gould, Porcelli and Toint, "Updating t
regularization parameter in the adaptive cubic regularization algorithm" RAL-TR-2011-007, Rutherford Appleton Laboratory, England (2011), `http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf <http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf>`__ (these are denoted beta, epsilon_chi and alpha_max in the paper)

.. index:: pair: variable; tiny_gap
.. _doxid-structnls__subproblem__control__type_1a28c746bfd481575683553dc28abe3e30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` tiny_gap

see reduce_gap

.. index:: pair: variable; large_root
.. _doxid-structnls__subproblem__control__type_1a7abb4c844c9fddec04f47c856f4a2383:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` large_root

see reduce_gap

.. index:: pair: variable; switch_to_newton
.. _doxid-structnls__subproblem__control__type_1a3e388718674f5243006272b3911fae22:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` switch_to_newton

if the Gauss-Newto to Newton model is specified, switch to Newton as soon as the norm of the gradient g is smaller than switch_to_newton

.. index:: pair: variable; cpu_time_limit
.. _doxid-structnls__subproblem__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structnls__subproblem__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; subproblem_direct
.. _doxid-structnls__subproblem__control__type_1a8c10db7cf72a4e3e52c9601007f7b1de:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subproblem_direct

use a direct (factorization) or (preconditioned) iterative method to find the search direction

.. index:: pair: variable; renormalize_weight
.. _doxid-structnls__subproblem__control__type_1aa451859552980839fa9fe17df018c042:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool renormalize_weight

should the weight be renormalized to account for a change in scaling?

.. index:: pair: variable; magic_step
.. _doxid-structnls__subproblem__control__type_1adb993f5277ee10213227e86ba94fd934:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool magic_step

allow the user to perform a "magic" step to improve the objective

.. index:: pair: variable; print_obj
.. _doxid-structnls__subproblem__control__type_1a95ba685ae01db06b2e9a16412097612a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool print_obj

print values of the objective/gradient rather than \|\|c\|\| and its gradien

.. index:: pair: variable; space_critical
.. _doxid-structnls__subproblem__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structnls__subproblem__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structnls__subproblem__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; rqs_control
.. _doxid-structnls__subproblem__control__type_1a7339f172b9757ed89a7595a29c4bca2b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rqs_control_type<doxid-structrqs__control__type>` rqs_control

control parameters for RQS

.. index:: pair: variable; glrt_control
.. _doxid-structnls__subproblem__control__type_1ac872f3f7f159cedfac4d6898d51842a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`glrt_control_type<doxid-structglrt__control__type>` glrt_control

control parameters for GLRT

.. index:: pair: variable; psls_control
.. _doxid-structnls__subproblem__control__type_1a6fe2b1a82e177fbd1a39d9de9652a2c5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_control_type<doxid-structpsls__control__type>` psls_control

control parameters for PSLS

.. index:: pair: variable; bsc_control
.. _doxid-structnls__subproblem__control__type_1a899f35f15eea56c23fe8fb22857e01f7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bsc_control_type<doxid-structbsc__control__type>` bsc_control

control parameters for BSC

.. index:: pair: variable; roots_control
.. _doxid-structnls__subproblem__control__type_1a08df6708e7b5364ff3e8fbde29f29014:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`roots_control_type<doxid-structroots__control__type>` roots_control

control parameters for ROOTS

