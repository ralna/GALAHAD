.. index:: pair: table; nrek_inform_type
.. _doxid-structnrek__inform__type:

nrek_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_nrek.h>
	
	struct nrek_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structnrek__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structnrek__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structnrek__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`n_vec<doxid-structnrek__inform__type_n_vec>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structnrek__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_regularized<doxid-structnrek__inform__type_obj_regularized>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structnrek__inform__type_x_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structnrek__inform__type_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight<doxid-structnrek__inform__type_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`next_weight<doxid-structnrek__inform__type_next_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`error<doxid-structnrek__inform__type_error>`;
		char :ref:`bad_alloc<doxid-structnrek__inform__type_bad_alloc>`[81];
		struct :ref:`nrek_time_type<doxid-structnrek__time__type>` :ref:`time<doxid-structnrek__inform__type_time>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structnrek__inform__type_sls_inform>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_s_inform<doxid-structnrek__inform__type_sls_s_inform>`;
		struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` :ref:`rqs_inform<doxid-structnrek__inform__type_rqs_inform>`;
	};
.. _details-structnrek__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structnrek__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

reported return status:

* **0**

  the solution has been found

* **-1**

  an array allocation has failed

* **-2**

  an array deallocation has failed

* **-3**

  n and/or $\Delta$ is not positive, or the requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' has been violated.

* **-9**

  the analysis phase of the factorization failed; the return status from the factorization package is given by inform.sls_inform.status or inform.sls_s_inform.status as appropriate

* **-10**

  the factorization failed; the return status from the factorization package is given by inform.sls_inform.status or inform.sls_s_inform.status as appropriate

* **-11**

  the solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given by inform.sls_inform.status or inform.sls_s_inform.status as appropriate

* **-15**

  $S$ does not appear to be strictly diagonally dominant

* **-16**

  ill-conditioning has prevented further progress

* **-18**

  too many iterations have been required. This may happen if control.eks max is too small, but may also be symptomatic of  a badly scaled problem.

* **-31** 

  a resolve call has been made before an initial call (see control.new_weight and control.new_values)

* **-38** 

  an error occurred in a call to an LAPACK subroutine


.. index:: pair: variable; alloc_status
.. _doxid-structnrek__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure

.. index:: pair: variable; iter
.. _doxid-structnrek__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; n_vec
.. _doxid-structnrek__inform__type_n_vec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` n_vec

the number of orthogonal vectors required (the dimension of the extended-Krylov subspace)

.. index:: pair: variable; obj
.. _doxid-structnrek__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the quadratic function $f + c^T x + \frac{1}{2} x^T H x$

.. index:: pair: variable; obj_regularized
.. _doxid-structnrek__inform__type_obj_regularized:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_regularized

the value of the regularized quadratic function $r(x)$

.. index:: pair: variable; x_norm
.. _doxid-structnrek__inform__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the $S$ -norm of $x$, $||x||_S$

.. index:: pair: variable; multiplier
.. _doxid-structnrek__inform__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the Lagrange multiplier associated with the regularization

.. index:: pair: variable; weight
.. _doxid-structnrek__inform__type_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight

the value of the current weight

.. index:: pair: variable; next_weight
.. _doxid-structnrek__inform__type_next_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` next_weight

the value of the the proposed next weight to be used if the current weight proves to be too small (see inform.increase).

.. index:: pair: variable; error
.. _doxid-structnrek__inform__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` error

the value of the norm of the maximum relative residual error, $\|(H+\lambda S) x + c\|/\max(1,\|c\|)$

.. index:: pair: variable; bad_alloc
.. _doxid-structnrek__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array that provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structnrek__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`nrek_time_type<doxid-structnrek__time__type>` time

time information

.. index:: pair: variable; sls_inform
.. _doxid-structnrek__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

Cholesky information for factorization and solves with $H$ (see sls_c documentation)

.. index:: pair: variable; sls_s_inform
.. _doxid-structnrek__inform__type_sls_s_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_s_inform

Cholesky information for factorization and solves with $S$ (see sls_c documentation)

.. index:: pair: variable; rqs_inform
.. _doxid-structnrek__inform__type_rqs_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` rqs_inform

diagonal subproblem solve information (see rqs_c documentation)
