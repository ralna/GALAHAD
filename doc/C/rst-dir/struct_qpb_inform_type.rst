.. index:: pair: struct; qpb_inform_type
.. _doxid-structqpb__inform__type:

qpb_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_qpb.h>
	
	struct qpb_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structqpb__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structqpb__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structqpb__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structqpb__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structqpb__inform__type_1ad37cf7ad93af3413bc01b6515aad692a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structqpb__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`;
		int64_t :ref:`factorization_integer<doxid-structqpb__inform__type_1a29cd3a5b0f30227170f825116d9ade9e>`;
		int64_t :ref:`factorization_real<doxid-structqpb__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structqpb__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nbacts<doxid-structqpb__inform__type_1a4b9a11ae940f04846c342978808696d6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nmods<doxid-structqpb__inform__type_1aedcbf93d59a135329f358f366e37cc94>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structqpb__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`non_negligible_pivot<doxid-structqpb__inform__type_1a827ddb7fead8e375404c9b770b67e771>`;
		bool :ref:`feasible<doxid-structqpb__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321>`;
		struct :ref:`qpb_time_type<doxid-structqpb__time__type>` :ref:`time<doxid-structqpb__inform__type_1ae4e944df2baf87107a291094002befb2>`;
		struct :ref:`lsqp_inform_type<doxid-structlsqp__inform__type>` :ref:`lsqp_inform<doxid-structqpb__inform__type_1acc149a8b0411baab4ddd0d9e4ccf28ff>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structqpb__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structqpb__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`;
		struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` :ref:`gltr_inform<doxid-structqpb__inform__type_1a27a98844f05f18669d3dd60d3e6a8e46>`;
		struct :ref:`fit_inform_type<doxid-structfit__inform__type>` :ref:`fit_inform<doxid-structqpb__inform__type_1ac6efa45e989564727014956bf3e00deb>`;
	};
.. _details-structqpb__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structqpb__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See QPB_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structqpb__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structqpb__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structqpb__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structqpb__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structqpb__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structqpb__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structqpb__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structqpb__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structqpb__inform__type_1a4b9a11ae940f04846c342978808696d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; nmods
.. _doxid-structqpb__inform__type_1aedcbf93d59a135329f358f366e37cc94:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nmods

the total number of factorizations which were modified to ensure that th matrix was an appropriate preconditioner

.. index:: pair: variable; obj
.. _doxid-structqpb__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by QPB_solve

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structqpb__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structqpb__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; time
.. _doxid-structqpb__inform__type_1ae4e944df2baf87107a291094002befb2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`qpb_time_type<doxid-structqpb__time__type>` time

timings (see above)

.. index:: pair: variable; lsqp_inform
.. _doxid-structqpb__inform__type_1acc149a8b0411baab4ddd0d9e4ccf28ff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lsqp_inform_type<doxid-structlsqp__inform__type>` lsqp_inform

inform parameters for LSQP

.. index:: pair: variable; fdc_inform
.. _doxid-structqpb__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structqpb__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structqpb__inform__type_1a27a98844f05f18669d3dd60d3e6a8e46:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

return information from GLTR

.. index:: pair: variable; fit_inform
.. _doxid-structqpb__inform__type_1ac6efa45e989564727014956bf3e00deb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fit_inform_type<doxid-structfit__inform__type>` fit_inform

return information from FIT

