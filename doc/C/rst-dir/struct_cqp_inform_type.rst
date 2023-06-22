.. index:: pair: table; cqp_inform_type
.. _doxid-structcqp__inform__type:

cqp_inform_type structure
-------------------------

.. toctree::
	:hidden:

inform derived type as a C struct :ref:`More...<details-structcqp__inform__type>`

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	cqp_inform_type = {
		-- fields
	
		:ref:`status<doxid-structcqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`,
		:ref:`alloc_status<doxid-structcqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`,
		:ref:`bad_alloc<doxid-structcqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`,
		:ref:`iter<doxid-structcqp__inform__type_1aab6f168571c2073e01e240524b8a3da0>`,
		:ref:`factorization_status<doxid-structcqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`,
		:ref:`factorization_integer<doxid-structcqp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e>`,
		:ref:`factorization_real<doxid-structcqp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc>`,
		:ref:`nfacts<doxid-structcqp__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f>`,
		:ref:`nbacts<doxid-structcqp__inform__type_1a4b9a11ae940f04846c342978808696d6>`,
		:ref:`threads<doxid-structcqp__inform__type_1a4f987a98d3e1221916748962e45399fe>`,
		:ref:`obj<doxid-structcqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`,
		:ref:`primal_infeasibility<doxid-structcqp__inform__type_1a2bce6cd733ae08834689fa66747f53b9>`,
		:ref:`dual_infeasibility<doxid-structcqp__inform__type_1a979cebdf2e5f1e043f48a615a46b0299>`,
		:ref:`complementary_slackness<doxid-structcqp__inform__type_1aa9bb6bfb5903021b1942fe5a02f23f06>`,
		:ref:`init_primal_infeasibility<doxid-structcqp__inform__type_1a85355c90bdd4cdd2e09bfed7fd9f66e1>`,
		:ref:`init_dual_infeasibility<doxid-structcqp__inform__type_1a8a03c79f840170d644609f1fb95d06e3>`,
		:ref:`init_complementary_slackness<doxid-structcqp__inform__type_1adbb8b72850a5e57700268fc582064615>`,
		:ref:`potential<doxid-structcqp__inform__type_1a85f37aa42c9e051ea61ae035ff63059e>`,
		:ref:`non_negligible_pivot<doxid-structcqp__inform__type_1a827ddb7fead8e375404c9b770b67e771>`,
		:ref:`feasible<doxid-structcqp__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321>`,
		:ref:`checkpointsIter<doxid-structcqp__inform__type_1acb0789a29239327ab8a4e929e0fbc65b>`,
		:ref:`checkpointsTime<doxid-structcqp__inform__type_1af2d3b92abc0ea9392d412ab45438eeb9>`,
		:ref:`time<doxid-structcqp__inform__type_1a437e235d4db7a908d3a2bdef228584a7>`,
		:ref:`fdc_inform<doxid-structcqp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4>`,
		:ref:`sbls_inform<doxid-structcqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`,
		:ref:`fit_inform<doxid-structcqp__inform__type_1ac6efa45e989564727014956bf3e00deb>`,
		:ref:`roots_inform<doxid-structcqp__inform__type_1a68574d04a336f7be88a151fa8b975885>`,
		:ref:`cro_inform<doxid-structcqp__inform__type_1a594c21272a0c3d75d5ffa712f1d8f971>`,
		:ref:`rpd_inform<doxid-structcqp__inform__type_1a823701505feea7615e9f8995769d8b60>`,
	}

.. _details-structcqp__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structcqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	status

return status. See CQP_solve for details.

.. index:: pair: variable; alloc_status
.. _doxid-structcqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; bad_alloc
.. _doxid-structcqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	bad_alloc

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; iter
.. _doxid-structcqp__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structcqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structcqp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structcqp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structcqp__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structcqp__inform__type_1a4b9a11ae940f04846c342978808696d6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; threads
.. _doxid-structcqp__inform__type_1a4f987a98d3e1221916748962e45399fe:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	threads

the number of threads used

.. index:: pair: variable; obj
.. _doxid-structcqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	obj

the value of the objective function at the best estimate of the solution determined by CQP_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structcqp__inform__type_1a2bce6cd733ae08834689fa66747f53b9:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; dual_infeasibility
.. _doxid-structcqp__inform__type_1a979cebdf2e5f1e043f48a615a46b0299:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	dual_infeasibility

the value of the dual infeasibility

.. index:: pair: variable; complementary_slackness
.. _doxid-structcqp__inform__type_1aa9bb6bfb5903021b1942fe5a02f23f06:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	complementary_slackness

the value of the complementary slackness

.. index:: pair: variable; init_primal_infeasibility
.. _doxid-structcqp__inform__type_1a85355c90bdd4cdd2e09bfed7fd9f66e1:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	init_primal_infeasibility

these values at the initial point (needed bg GALAHAD_CCQP)

.. index:: pair: variable; init_dual_infeasibility
.. _doxid-structcqp__inform__type_1a8a03c79f840170d644609f1fb95d06e3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	init_dual_infeasibility

see init_primal_infeasibility

.. index:: pair: variable; init_complementary_slackness
.. _doxid-structcqp__inform__type_1adbb8b72850a5e57700268fc582064615:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	init_complementary_slackness

see init_primal_infeasibility

.. index:: pair: variable; potential
.. _doxid-structcqp__inform__type_1a85f37aa42c9e051ea61ae035ff63059e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	potential

the value of the logarithmic potential function sum -log(distance to constraint boundary)

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structcqp__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structcqp__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	feasible

is the returned "solution" feasible?

.. index:: pair: variable; checkpointsIter
.. _doxid-structcqp__inform__type_1acb0789a29239327ab8a4e929e0fbc65b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	checkpointsIter

checkpoints(i) records the iteration at which the criticality measures first fall below :math:`10^{-i-1}`, i = 0, ..., 15 (-1 means not achieved)

.. index:: pair: variable; checkpointsTime
.. _doxid-structcqp__inform__type_1af2d3b92abc0ea9392d412ab45438eeb9:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	checkpointsTime

see checkpointsIter

.. index:: pair: variable; time
.. _doxid-structcqp__inform__type_1a437e235d4db7a908d3a2bdef228584a7:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structcqp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structcqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sbls_inform

inform parameters for SBLS

.. index:: pair: variable; fit_inform
.. _doxid-structcqp__inform__type_1ac6efa45e989564727014956bf3e00deb:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	fit_inform

return information from FIT

.. index:: pair: variable; roots_inform
.. _doxid-structcqp__inform__type_1a68574d04a336f7be88a151fa8b975885:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	roots_inform

return information from ROOTS

.. index:: pair: variable; cro_inform
.. _doxid-structcqp__inform__type_1a594c21272a0c3d75d5ffa712f1d8f971:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	cro_inform

inform parameters for CRO

.. index:: pair: variable; rpd_inform
.. _doxid-structcqp__inform__type_1a823701505feea7615e9f8995769d8b60:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	rpd_inform

inform parameters for RPD

