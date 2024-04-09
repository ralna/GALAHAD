.. index:: pair: struct; clls_inform_type
.. _doxid-structclls__inform__type:

clls_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_clls.h>
	
	struct clls_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structclls__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structclls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structclls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structclls__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structclls__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`;
		int64_t :ref:`factorization_integer<doxid-structclls__inform__type_1a29cd3a5b0f30227170f825116d9ade9e>`;
		int64_t :ref:`factorization_real<doxid-structclls__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nfacts<doxid-structclls__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nbacts<doxid-structclls__inform__type_1a4b9a11ae940f04846c342978808696d6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`threads<doxid-structclls__inform__type_1a4f987a98d3e1221916748962e45399fe>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structclls__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`primal_infeasibility<doxid-structclls__inform__type_1a2bce6cd733ae08834689fa66747f53b9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`dual_infeasibility<doxid-structclls__inform__type_1a979cebdf2e5f1e043f48a615a46b0299>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`complementary_slackness<doxid-structclls__inform__type_1aa9bb6bfb5903021b1942fe5a02f23f06>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`init_primal_infeasibility<doxid-structclls__inform__type_1a85355c90bdd4cdd2e09bfed7fd9f66e1>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`init_dual_infeasibility<doxid-structclls__inform__type_1a8a03c79f840170d644609f1fb95d06e3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`init_complementary_slackness<doxid-structclls__inform__type_1adbb8b72850a5e57700268fc582064615>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`potential<doxid-structclls__inform__type_1a85f37aa42c9e051ea61ae035ff63059e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`non_negligible_pivot<doxid-structclls__inform__type_1a827ddb7fead8e375404c9b770b67e771>`;
		bool :ref:`feasible<doxid-structclls__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`checkpointsIter<doxid-structclls__inform__type_1acb0789a29239327ab8a4e929e0fbc65b>`[16];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`checkpointsTime<doxid-structclls__inform__type_1af2d3b92abc0ea9392d412ab45438eeb9>`[16];
		struct :ref:`clls_time_type<doxid-structclls__time__type>` :ref:`time<doxid-structclls__inform__type_1a437e235d4db7a908d3a2bdef228584a7>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structclls__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structclls__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`;
		struct :ref:`fit_inform_type<doxid-structfit__inform__type>` :ref:`fit_inform<doxid-structclls__inform__type_1ac6efa45e989564727014956bf3e00deb>`;
		struct :ref:`roots_inform_type<doxid-structroots__inform__type>` :ref:`roots_inform<doxid-structclls__inform__type_1a68574d04a336f7be88a151fa8b975885>`;
		struct :ref:`cro_inform_type<doxid-structcro__inform__type>` :ref:`cro_inform<doxid-structclls__inform__type_1a594c21272a0c3d75d5ffa712f1d8f971>`;
		struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` :ref:`rpd_inform<doxid-structclls__inform__type_1a823701505feea7615e9f8995769d8b60>`;
	};
.. _details-structclls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structclls__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See CLLS_solve for details.

.. index:: pair: variable; alloc_status
.. _doxid-structclls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; bad_alloc
.. _doxid-structclls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; iter
.. _doxid-structclls__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structclls__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structclls__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structclls__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structclls__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structclls__inform__type_1a4b9a11ae940f04846c342978808696d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; threads
.. _doxid-structclls__inform__type_1a4f987a98d3e1221916748962e45399fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` threads

the number of threads used

.. index:: pair: variable; obj
.. _doxid-structclls__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by CLLS_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structclls__inform__type_1a2bce6cd733ae08834689fa66747f53b9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; dual_infeasibility
.. _doxid-structclls__inform__type_1a979cebdf2e5f1e043f48a615a46b0299:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` dual_infeasibility

the value of the dual infeasibility

.. index:: pair: variable; complementary_slackness
.. _doxid-structclls__inform__type_1aa9bb6bfb5903021b1942fe5a02f23f06:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` complementary_slackness

the value of the complementary slackness

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structclls__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structclls__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; checkpointsIter
.. _doxid-structclls__inform__type_1acb0789a29239327ab8a4e929e0fbc65b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` checkpointsIter[16]

checkpoints(i) records the iteration at which the criticality measures first fall below $10^{-i-1}$, i = 0, ..., 15 (-1 means not achieved)

.. index:: pair: variable; checkpointsTime
.. _doxid-structclls__inform__type_1af2d3b92abc0ea9392d412ab45438eeb9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` checkpointsTime[16]

see checkpointsIter

.. index:: pair: variable; time
.. _doxid-structclls__inform__type_1a437e235d4db7a908d3a2bdef228584a7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`clls_time_type<doxid-structclls__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structclls__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sls_inform
.. _doxid-structclls__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsbls__inform__type>` sls_inform

inform parameters for SLS

.. index:: pair: variable; sls_pounce_inform
.. _doxid-structclls__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsbls__inform__type>` sls_pounce_inform

inform parameters for SLS_pounce
.. index:: pair: variable; fit_inform
.. _doxid-structclls__inform__type_1ac6efa45e989564727014956bf3e00deb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fit_inform_type<doxid-structfit__inform__type>` fit_inform

return information from FIT

.. index:: pair: variable; roots_inform
.. _doxid-structclls__inform__type_1a68574d04a336f7be88a151fa8b975885:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`roots_inform_type<doxid-structroots__inform__type>` roots_inform

return information from ROOTS

.. index:: pair: variable; cro_inform
.. _doxid-structclls__inform__type_1a594c21272a0c3d75d5ffa712f1d8f971:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`cro_inform_type<doxid-structcro__inform__type>` cro_inform

inform parameters for CRO

.. index:: pair: variable; rpd_inform
.. _doxid-structclls__inform__type_1a823701505feea7615e9f8995769d8b60:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

