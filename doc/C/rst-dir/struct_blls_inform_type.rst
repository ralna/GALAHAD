.. index:: pair: struct; blls_inform_type
.. _doxid-structblls__inform__type:

blls_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_blls.h>
	
	struct blls_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structblls__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structblls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structblls__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structblls__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structblls__inform__type_1ad37cf7ad93af3413bc01b6515aad692a>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structblls__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structblls__inform__type_1acb02a4d1ae275a55874bb9897262b1fe>`;
		char :ref:`bad_alloc<doxid-structblls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		struct :ref:`blls_time_type<doxid-structblls__time__type>` :ref:`time<doxid-structblls__inform__type_1ac8d53b198a1597a4a9fe75d4c1191ec0>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structblls__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`;
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>` :ref:`convert_inform<doxid-structblls__inform__type_1a7006a98737e58bb90259d7705ef537ae>`;
	};
.. _details-structblls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structblls__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

reported return status.

.. index:: pair: variable; alloc_status
.. _doxid-structblls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

Fortran STAT value after allocate failure.

.. index:: pair: variable; factorization_status
.. _doxid-structblls__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

status return from factorization

.. index:: pair: variable; iter
.. _doxid-structblls__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structblls__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

number of CG iterations required

.. index:: pair: variable; obj
.. _doxid-structblls__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

current value of the objective function, r(x).

.. index:: pair: variable; norm_pg
.. _doxid-structblls__inform__type_1acb02a4d1ae275a55874bb9897262b1fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

current value of the Euclidean norm of projected gradient of r(x).

.. index:: pair: variable; bad_alloc
.. _doxid-structblls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structblls__inform__type_1ac8d53b198a1597a4a9fe75d4c1191ec0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`blls_time_type<doxid-structblls__time__type>` time

times for various stages

.. index:: pair: variable; sbls_inform
.. _doxid-structblls__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform values from SBLS

.. index:: pair: variable; convert_inform
.. _doxid-structblls__inform__type_1a7006a98737e58bb90259d7705ef537ae:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`convert_inform_type<doxid-structconvert__inform__type>` convert_inform

inform values for CONVERT

