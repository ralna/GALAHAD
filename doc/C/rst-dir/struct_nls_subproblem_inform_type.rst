.. index:: pair: struct; nls_subproblem_inform_type
.. _doxid-structnls__subproblem__inform__type:

nls_subproblem_inform_type structure
------------------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_nls.h>
	
	struct nls_subproblem_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structnls__subproblem__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structnls__subproblem__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structnls__subproblem__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		char :ref:`bad_eval<doxid-structnls__subproblem__inform__type_1a184c27298dc565470437c213a2bd2f3e>`[13];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structnls__subproblem__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structnls__subproblem__inform__type_1ad37cf7ad93af3413bc01b6515aad692a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`c_eval<doxid-structnls__subproblem__inform__type_1ab8312e1defeefffdcc0b5956bcb31ad4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`j_eval<doxid-structnls__subproblem__inform__type_1a47a079918ad01b32fd15ed6a0b8bd581>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structnls__subproblem__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_max<doxid-structnls__subproblem__inform__type_1a97dadabf3b7bdf921c4dcd1f43129f05>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structnls__subproblem__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`;
		int64_t :ref:`max_entries_factors<doxid-structnls__subproblem__inform__type_1a177e429e737cfa2cd3df051a65fcfb68>`;
		int64_t :ref:`factorization_integer<doxid-structnls__subproblem__inform__type_1a29cd3a5b0f30227170f825116d9ade9e>`;
		int64_t :ref:`factorization_real<doxid-structnls__subproblem__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorization_average<doxid-structnls__subproblem__inform__type_1a42d0c89df887685f68327d07c6e92f05>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structnls__subproblem__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_c<doxid-structnls__subproblem__inform__type_1a4969b17b30edb63a6bbcb89c7c10a340>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_g<doxid-structnls__subproblem__inform__type_1ae1bc0a751c6ede62421bbc49fbe7d9fe>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight<doxid-structnls__subproblem__inform__type_1adcd20aeaf7042e972ddab56f3867ce70>`;
		struct :ref:`nls_time_type<doxid-structnls__time__type>` :ref:`time<doxid-structnls__subproblem__inform__type_1a44dc03b1a33bf900f668c713cbac9498>`;
		struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` :ref:`rqs_inform<doxid-structnls__subproblem__inform__type_1a68497e7bbd1695ac9b830fc8fe594d60>`;
		struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` :ref:`glrt_inform<doxid-structnls__subproblem__inform__type_1aa5a47a840c1f9680ac8b9e4db3eb9e88>`;
		struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` :ref:`psls_inform<doxid-structnls__subproblem__inform__type_1a57ca5ed37882eb917736f845d3cdb8ee>`;
		struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` :ref:`bsc_inform<doxid-structnls__subproblem__inform__type_1ab95c5e6786b9d93eb147f64fbf14da17>`;
		struct :ref:`roots_inform_type<doxid-structroots__inform__type>` :ref:`roots_inform<doxid-structnls__subproblem__inform__type_1a68574d04a336f7be88a151fa8b975885>`;
	};
.. _details-structnls__subproblem__inform__type:

detailed documentation
----------------------

subproblem_inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structnls__subproblem__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See NLS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structnls__subproblem__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structnls__subproblem__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; bad_eval
.. _doxid-structnls__subproblem__inform__type_1a184c27298dc565470437c213a2bd2f3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_eval[13]

the name of the user-supplied evaluation routine for which an error occurred

.. index:: pair: variable; iter
.. _doxid-structnls__subproblem__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structnls__subproblem__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of CG iterations performed

.. index:: pair: variable; c_eval
.. _doxid-structnls__subproblem__inform__type_1ab8312e1defeefffdcc0b5956bcb31ad4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` c_eval

the total number of evaluations of the residual function c(x)

.. index:: pair: variable; j_eval
.. _doxid-structnls__subproblem__inform__type_1a47a079918ad01b32fd15ed6a0b8bd581:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` j_eval

the total number of evaluations of the Jacobian J(x) of c(x)

.. index:: pair: variable; h_eval
.. _doxid-structnls__subproblem__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the scaled Hessian H(x,y) of c(x)

.. index:: pair: variable; factorization_max
.. _doxid-structnls__subproblem__inform__type_1a97dadabf3b7bdf921c4dcd1f43129f05:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; factorization_status
.. _doxid-structnls__subproblem__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; max_entries_factors
.. _doxid-structnls__subproblem__inform__type_1a177e429e737cfa2cd3df051a65fcfb68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structnls__subproblem__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structnls__subproblem__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; factorization_average
.. _doxid-structnls__subproblem__inform__type_1a42d0c89df887685f68327d07c6e92f05:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorization_average

the average number of factorizations per sub-problem solve

.. index:: pair: variable; obj
.. _doxid-structnls__subproblem__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function $\frac{1}{2}\|c(x)\|^2_W$ at the best estimate the solution, x, determined by NLS_solve

.. index:: pair: variable; norm_c
.. _doxid-structnls__subproblem__inform__type_1a4969b17b30edb63a6bbcb89c7c10a340:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_c

the norm of the residual $\|c(x)\|_W$ at the best estimate of the solution x, determined by NLS_solve

.. index:: pair: variable; norm_g
.. _doxid-structnls__subproblem__inform__type_1ae1bc0a751c6ede62421bbc49fbe7d9fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_g

the norm of the gradient of $\|c(x)\|_W$ of the objective function at the best estimate, x, of the solution determined by NLS_solve

.. index:: pair: variable; weight
.. _doxid-structnls__subproblem__inform__type_1adcd20aeaf7042e972ddab56f3867ce70:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight

the final regularization weight used

.. index:: pair: variable; time
.. _doxid-structnls__subproblem__inform__type_1a44dc03b1a33bf900f668c713cbac9498:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`nls_time_type<doxid-structnls__time__type>` time

timings (see above)

.. index:: pair: variable; rqs_inform
.. _doxid-structnls__subproblem__inform__type_1a68497e7bbd1695ac9b830fc8fe594d60:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` rqs_inform

inform parameters for RQS

.. index:: pair: variable; glrt_inform
.. _doxid-structnls__subproblem__inform__type_1aa5a47a840c1f9680ac8b9e4db3eb9e88:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` glrt_inform

inform parameters for GLRT

.. index:: pair: variable; psls_inform
.. _doxid-structnls__subproblem__inform__type_1a57ca5ed37882eb917736f845d3cdb8ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; bsc_inform
.. _doxid-structnls__subproblem__inform__type_1ab95c5e6786b9d93eb147f64fbf14da17:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` bsc_inform

inform parameters for BSC

.. index:: pair: variable; roots_inform
.. _doxid-structnls__subproblem__inform__type_1a68574d04a336f7be88a151fa8b975885:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`roots_inform_type<doxid-structroots__inform__type>` roots_inform

inform parameters for ROOTS

