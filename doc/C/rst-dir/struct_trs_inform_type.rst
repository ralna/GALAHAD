.. index:: pair: table; trs_inform_type
.. _doxid-structtrs__inform__type:

trs_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_trs.h>
	
	struct trs_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structtrs__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structtrs__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorizations<doxid-structtrs__inform__type_factorizations>`;
		int64_t :ref:`max_entries_factors<doxid-structtrs__inform__type_max_entries_factors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`len_history<doxid-structtrs__inform__type_len_history>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structtrs__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structtrs__inform__type_x_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structtrs__inform__type_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pole<doxid-structtrs__inform__type_pole>`;
		bool :ref:`dense_factorization<doxid-structtrs__inform__type_dense_factorization>`;
		bool :ref:`hard_case<doxid-structtrs__inform__type_hard_case>`;
		char :ref:`bad_alloc<doxid-structtrs__inform__type_bad_alloc>`[81];
		struct :ref:`trs_time_type<doxid-structtrs__time__type>` :ref:`time<doxid-structtrs__inform__type_time>`;
		struct :ref:`trs_history_type<doxid-structtrs__history__type>` :ref:`history<doxid-structtrs__inform__type_history>`[100];
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structtrs__inform__type_sls_inform>`;
		struct :ref:`ir_inform_type<doxid-structir__inform__type>` :ref:`ir_inform<doxid-structtrs__inform__type_ir_inform>`;
	};
.. _details-structtrs__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structtrs__inform__type_status:

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

  n and/or Delta is not positive

* **-9**

  the analysis phase of the factorization of $H + \lambda M$ failed

* **-10**

  the factorization of $H + \lambda M$ failed

* **-15**

  $M$ does not appear to be strictly diagonally dominant

* **-16**

  ill-conditioning has prevented further progress

.. index:: pair: variable; alloc_status
.. _doxid-structtrs__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structtrs__inform__type_factorizations:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorizations

the number of factorizations performed

.. index:: pair: variable; max_entries_factors
.. _doxid-structtrs__inform__type_max_entries_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; len_history
.. _doxid-structtrs__inform__type_len_history:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` len_history

the number of $(||x||_M,\lambda)$ pairs in the history

.. index:: pair: variable; obj
.. _doxid-structtrs__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the quadratic function

.. index:: pair: variable; x_norm
.. _doxid-structtrs__inform__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the $M$ -norm of $x$, $||x||_M$

.. index:: pair: variable; multiplier
.. _doxid-structtrs__inform__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the Lagrange multiplier corresponding to the trust-region constraint

.. index:: pair: variable; pole
.. _doxid-structtrs__inform__type_pole:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pole

a lower bound max $(0,-\lambda_1)$, where $\lambda_1$ is the left-most eigenvalue of $(H,M)$

.. index:: pair: variable; dense_factorization
.. _doxid-structtrs__inform__type_dense_factorization:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool dense_factorization

was a dense factorization used?

.. index:: pair: variable; hard_case
.. _doxid-structtrs__inform__type_hard_case:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hard_case

has the hard case occurred?

.. index:: pair: variable; bad_alloc
.. _doxid-structtrs__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array that provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structtrs__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trs_time_type<doxid-structtrs__time__type>` time

time information

.. index:: pair: variable; history
.. _doxid-structtrs__inform__type_history:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trs_history_type<doxid-structtrs__history__type>` history[100]

history information

.. index:: pair: variable; sls_inform
.. _doxid-structtrs__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

cholesky information (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structtrs__inform__type_ir_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

iterative_refinement information (see ir_c documentation)

