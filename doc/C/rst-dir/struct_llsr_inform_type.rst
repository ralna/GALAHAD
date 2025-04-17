.. index:: pair: table; llsr_inform_type
.. _doxid-structllsr__inform__type:

llsr_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_llsr.h>
	
	struct llsr_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structllsr__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structllsr__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorizations<doxid-structllsr__inform__type_factorizations>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`len_history<doxid-structllsr__inform__type_len_history>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`r_norm<doxid-structllsr__inform__type_r_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structllsr__inform__type_x_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structllsr__inform__type_multiplier>`;
		char :ref:`bad_alloc<doxid-structllsr__inform__type_bad_alloc>`[81];
		struct :ref:`llsr_time_type<doxid-structllsr__time__type>` :ref:`time<doxid-structllsr__inform__type_time>`;
		struct :ref:`llsr_history_type<doxid-structllsr__history__type>` :ref:`history<doxid-structllsr__inform__type_history>`[100];
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structllsr__inform__type_sbls_inform>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structllsr__inform__type_sls_inform>`;
		struct :ref:`ir_inform_type<doxid-structir__inform__type>` :ref:`ir_inform<doxid-structllsr__inform__type_ir_inform>`;
	};
.. _details-structllsr__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structllsr__inform__type_status:

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

* **-10**

  the factorization of $K(\lambda)$ failed

* **-15**

  $S$ does not appear to be strictly diagonally dominant

* **-16**

  ill-conditioning has prevented furthr progress

.. index:: pair: variable; alloc_status
.. _doxid-structllsr__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structllsr__inform__type_factorizations:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorizations

the number of factorizations performed

.. index:: pair: variable; len_history
.. _doxid-structllsr__inform__type_len_history:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` len_history

the number of ($\|x\|_S$, $\lambda$) pairs in the history

.. index:: pair: variable; r_norm
.. _doxid-structllsr__inform__type_r_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` r_norm

corresponding value of the two-norm of the residual, $\|A x(\lambda) - b\|$

.. index:: pair: variable; x_norm
.. _doxid-structllsr__inform__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the S-norm of x, $\|x\|_S$

.. index:: pair: variable; multiplier
.. _doxid-structllsr__inform__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the multiplier corresponding to the regularization term

.. index:: pair: variable; bad_alloc
.. _doxid-structllsr__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structllsr__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`llsr_time_type<doxid-structllsr__time__type>` time

time information

.. index:: pair: variable; history
.. _doxid-structllsr__inform__type_history:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`llsr_history_type<doxid-structllsr__history__type>` history[100]

history information

.. index:: pair: variable; sbls_inform
.. _doxid-structllsr__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

information from the symmetric factorization and related linear solves (see sbls_c documentation)

.. index:: pair: variable; sls_inform
.. _doxid-structllsr__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from the factorization of S and related linear solves (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structllsr__inform__type_ir_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

information from the iterative refinement for definite system solves (see ir_c documentation)

