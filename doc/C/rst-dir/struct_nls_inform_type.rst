.. index:: pair: struct; nls_inform_type
.. _doxid-structnls__inform__type:

nls_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_nls.h>
	
	struct nls_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structnls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structnls__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structnls__inform__type_bad_alloc>`[81];
		char :ref:`bad_eval<doxid-structnls__inform__type_bad_eval>`[13];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structnls__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structnls__inform__type_cg_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`c_eval<doxid-structnls__inform__type_c_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`j_eval<doxid-structnls__inform__type_j_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structnls__inform__type_h_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_max<doxid-structnls__inform__type_factorization_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structnls__inform__type_factorization_status>`;
		int64_t :ref:`max_entries_factors<doxid-structnls__inform__type_max_entries_factors>`;
		int64_t :ref:`factorization_integer<doxid-structnls__inform__type_factorization_integer>`;
		int64_t :ref:`factorization_real<doxid-structnls__inform__type_factorization_real>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorization_average<doxid-structnls__inform__type_factorization_average>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structnls__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_c<doxid-structnls__inform__type_norm_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_g<doxid-structnls__inform__type_norm_g>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight<doxid-structnls__inform__type_weight>`;
		struct :ref:`nls_time_type<doxid-structnls__time__type>` :ref:`time<doxid-structnls__inform__type_time>`;
		struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` :ref:`rqs_inform<doxid-structnls__inform__type_rqs_inform>`;
		struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` :ref:`glrt_inform<doxid-structnls__inform__type_glrt_inform>`;
		struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` :ref:`psls_inform<doxid-structnls__inform__type_psls_inform>`;
		struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` :ref:`bsc_inform<doxid-structnls__inform__type_bsc_inform>`;
		struct :ref:`roots_inform_type<doxid-structroots__inform__type>` :ref:`roots_inform<doxid-structnls__inform__type_roots_inform>`;
		struct :ref:`nls_subproblem_inform_type<doxid-structnls__subproblem__inform__type>` :ref:`subproblem_inform<doxid-structnls__inform__type_subproblem_inform>`;
	};
.. _details-structnls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structnls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See NLS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structnls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structnls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; bad_eval
.. _doxid-structnls__inform__type_bad_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_eval[13]

the name of the user-supplied evaluation routine for which an error occurred

.. index:: pair: variable; iter
.. _doxid-structnls__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structnls__inform__type_cg_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of CG iterations performed

.. index:: pair: variable; c_eval
.. _doxid-structnls__inform__type_c_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` c_eval

the total number of evaluations of the residual function c(x)

.. index:: pair: variable; j_eval
.. _doxid-structnls__inform__type_j_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` j_eval

the total number of evaluations of the Jacobian J(x) of c(x)

.. index:: pair: variable; h_eval
.. _doxid-structnls__inform__type_h_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the scaled Hessian H(x,y) of c(x)

.. index:: pair: variable; factorization_max
.. _doxid-structnls__inform__type_factorization_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; factorization_status
.. _doxid-structnls__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; max_entries_factors
.. _doxid-structnls__inform__type_max_entries_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structnls__inform__type_factorization_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structnls__inform__type_factorization_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; factorization_average
.. _doxid-structnls__inform__type_factorization_average:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorization_average

the average number of factorizations per sub-problem solve

.. index:: pair: variable; obj
.. _doxid-structnls__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function $\frac{1}{2}\|c(x)\|^2_W$ at the best estimate the solution, x, determined by NLS_solve

.. index:: pair: variable; norm_c
.. _doxid-structnls__inform__type_norm_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_c

the norm of the residual $\|c(x)\|_W$ at the best estimate of the solution x, determined by NLS_solve

.. index:: pair: variable; norm_g
.. _doxid-structnls__inform__type_norm_g:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_g

the norm of the gradient of $\|c(x)\|_W$ of the objective function at the best estimate, x, of the solution determined by NLS_solve

.. index:: pair: variable; weight
.. _doxid-structnls__inform__type_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight

the final regularization weight used

.. index:: pair: variable; time
.. _doxid-structnls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`nls_time_type<doxid-structnls__time__type>` time

timings (see above)

.. index:: pair: variable; rqs_inform
.. _doxid-structnls__inform__type_rqs_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` rqs_inform

inform parameters for RQS

.. index:: pair: variable; glrt_inform
.. _doxid-structnls__inform__type_glrt_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` glrt_inform

inform parameters for GLRT

.. index:: pair: variable; psls_inform
.. _doxid-structnls__inform__type_psls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; bsc_inform
.. _doxid-structnls__inform__type_bsc_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` bsc_inform

inform parameters for BSC

.. index:: pair: variable; roots_inform
.. _doxid-structnls__inform__type_roots_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`roots_inform_type<doxid-structroots__inform__type>` roots_inform

inform parameters for ROOTS

.. index:: pair: variable; subproblem_inform
.. _doxid-structnls__inform__type_subproblem_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`nls_subproblem_inform_type<doxid-structnls__subproblem__inform__type>` subproblem_inform

inform parameters for subproblem

