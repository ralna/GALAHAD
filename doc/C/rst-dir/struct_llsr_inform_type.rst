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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structllsr__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structllsr__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorizations<doxid-structllsr__inform__type_1a9a6a5a0de7d7a6048b4170a768c0c86f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`len_history<doxid-structllsr__inform__type_1a2087c1ee7c5859aa738d2f07ba91b4a6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`r_norm<doxid-structllsr__inform__type_1ae908410fabf891cfd89626c3605c38ca>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structllsr__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structllsr__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0>`;
		char :ref:`bad_alloc<doxid-structllsr__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		struct :ref:`llsr_time_type<doxid-structllsr__time__type>` :ref:`time<doxid-structllsr__inform__type_1ace18e9a0877156e432cc23c7d5799dd6>`;
		struct :ref:`llsr_history_type<doxid-structllsr__history__type>` :ref:`history<doxid-structllsr__inform__type_1a13047d24b0cf3469a41cc14c364d3587>`[100];
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structllsr__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structllsr__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0>`;
		struct :ref:`ir_inform_type<doxid-structir__inform__type>` :ref:`ir_inform<doxid-structllsr__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5>`;
	};
.. _details-structllsr__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structllsr__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

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
.. _doxid-structllsr__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structllsr__inform__type_1a9a6a5a0de7d7a6048b4170a768c0c86f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorizations

the number of factorizations performed

.. index:: pair: variable; len_history
.. _doxid-structllsr__inform__type_1a2087c1ee7c5859aa738d2f07ba91b4a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` len_history

the number of ($\|x\|_S$, $\lambda$) pairs in the history

.. index:: pair: variable; r_norm
.. _doxid-structllsr__inform__type_1ae908410fabf891cfd89626c3605c38ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` r_norm

corresponding value of the two-norm of the residual, $\|A x(\lambda) - b\|$

.. index:: pair: variable; x_norm
.. _doxid-structllsr__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the S-norm of x, $\|x\|_S$

.. index:: pair: variable; multiplier
.. _doxid-structllsr__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the multiplier corresponding to the regularization term

.. index:: pair: variable; bad_alloc
.. _doxid-structllsr__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structllsr__inform__type_1ace18e9a0877156e432cc23c7d5799dd6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`llsr_time_type<doxid-structllsr__time__type>` time

time information

.. index:: pair: variable; history
.. _doxid-structllsr__inform__type_1a13047d24b0cf3469a41cc14c364d3587:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`llsr_history_type<doxid-structllsr__history__type>` history[100]

history information

.. index:: pair: variable; sbls_inform
.. _doxid-structllsr__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

information from the symmetric factorization and related linear solves (see sbls_c documentation)

.. index:: pair: variable; sls_inform
.. _doxid-structllsr__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from the factorization of S and related linear solves (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structllsr__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

information from the iterative refinement for definite system solves (see ir_c documentation)

