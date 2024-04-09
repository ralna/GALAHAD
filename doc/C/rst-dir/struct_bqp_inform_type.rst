.. index:: pair: struct; bqp_inform_type
.. _doxid-structbqp__inform__type:

bqp_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bqp.h>
	
	struct bqp_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structbqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structbqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structbqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structbqp__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structbqp__inform__type_1ad37cf7ad93af3413bc01b6515aad692a>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structbqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structbqp__inform__type_1acb02a4d1ae275a55874bb9897262b1fe>`;
		char :ref:`bad_alloc<doxid-structbqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		struct :ref:`bqp_time_type<doxid-structbqp__time__type>` :ref:`time<doxid-structbqp__inform__type_1a7f44be002389597b3f6c06e9a9b6eefa>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structbqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`;
	};
.. _details-structbqp__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structbqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

reported return status:

* **0**

  success

* **-1**

  allocation error

* **-2**

  deallocation error

* **-3**

  matrix data faulty (.n < 1, .ne < 0)

* **-20**

  alegedly +ve definite matrix is not

.. index:: pair: variable; alloc_status
.. _doxid-structbqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

Fortran STAT value after allocate failure.

.. index:: pair: variable; factorization_status
.. _doxid-structbqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

status return from factorization

.. index:: pair: variable; iter
.. _doxid-structbqp__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structbqp__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

number of CG iterations required

.. index:: pair: variable; obj
.. _doxid-structbqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

current value of the objective function

.. index:: pair: variable; norm_pg
.. _doxid-structbqp__inform__type_1acb02a4d1ae275a55874bb9897262b1fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

current value of the projected gradient

.. index:: pair: variable; bad_alloc
.. _doxid-structbqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structbqp__inform__type_1a7f44be002389597b3f6c06e9a9b6eefa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bqp_time_type<doxid-structbqp__time__type>` time

times for various stages

.. index:: pair: variable; sbls_inform
.. _doxid-structbqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform values from SBLS

